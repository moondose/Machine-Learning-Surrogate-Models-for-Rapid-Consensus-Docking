#!/usr/bin/env python3
"""Generate SUBMO embeddings and select a globally diverse molecular subset.

Example (run in an environment containing torch, DGL, dgllife, RDKit, pandas,
NumPy and threadpoolctl):
    python submo_diverse_sampling.py --input dataset.csv.gz --smiles-col smiles \
        --model save_pickle/qm9_normalize_relu_1_attentivefp_model.p \
        --output-dir results/submo --n-select 500000 --workers 8

This script uses the upstream GNNMultitask.forward(task=None), canonical featurizers and graph construction. 
It accepts a trained QM9 full-model SUBMO checkpoint for transfer to other datasets.

Outputs:
    embeddings.npy: all valid graph embeddings, in metadata row order.
    metadata.csv: original columns plus source_row and embedding_row.
    selected.csv: selected original rows in greedy order, plus selection_rank.
    selected_embeddings.npy: embeddings aligned with selected.csv.
    rejected.csv: invalid SMILES/graph failures and their original row indices.
    run.json: parameters, model hash, versions and output counts.

Source: https://github.com/tomotomonakanaka/SUBMO
The model and greedy objective follow Tomohiro Nakamura's MIT-licensed SUBMO.
Original copyright: Copyright <2022> <COPYRIGHT Tomohiro Nakamura>.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import hashlib
import inspect
import json
import pickle
from pathlib import Path
import tempfile

import numpy as np


def exact_logdet_greedy(x, k, workers=1, block_size=100000):
    """Return greedy indices for L=I+XX^T using the Woodbury identity.

    At each step A = I + X_S.T @ X_S. The marginal determinant factor is
    1 + x_i.T @ inv(A) @ x_i. For selected vector v, update inv(A) by
    Sherman-Morrison and subtract (x_i.T @ inv(A) @ v)^2 / denominator
    from every remaining candidate's factor. This is the same Schur
    complement recurrence used by upstream, without storing N*k factors.
    """
    n, d = x.shape
    if not 1 <= k <= n:
        raise ValueError(f'n_select must be between 1 and {n}, got {k}')
    inverse = np.eye(d, dtype=np.float64)
    scores = np.empty(n, dtype=np.float64)
    blocks = [(lo, min(lo + block_size, n)) for lo in range(0, n, block_size)]
    for lo, hi in blocks:
        block = np.asarray(x[lo:hi], dtype=np.float64)
        if not np.isfinite(block).all():
            raise ValueError('Embeddings contain non-finite values')
        scores[lo:hi] = 1.0 + np.einsum('ij,ij->i', block, block)
    selected = np.empty(k, dtype=np.int64)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for step in range(k):
            j = int(np.argmax(scores))
            selected[step] = j
            v = np.asarray(x[j], dtype=np.float64)
            direction = inverse @ v
            denominator = 1.0 + float(v @ direction)
            if not np.isfinite(denominator) or denominator <= 0:
                raise FloatingPointError('Non-positive logdet pivot; numerical failure')

            def update(bounds):
                lo, hi = bounds
                block = np.asarray(x[lo:hi], dtype=np.float64)
                cross = block @ direction
                scores[lo:hi] -= cross * cross / denominator

            # Exhausting map propagates worker exceptions and synchronizes the step.
            list(pool.map(update, blocks))
            inverse -= np.outer(direction, direction) / denominator
            inverse = (inverse + inverse.T) * 0.5
            scores[j] = -np.inf
            if step == 0 or (step + 1) % 1000 == 0 or step + 1 == k:
                print(f'Selected {step + 1}/{k}', flush=True)
    return selected


_FEATURIZERS = None


def build_graph(item):
    """Match upstream graph_featurizer while retaining failed-row identities."""
    global _FEATURIZERS
    from dgllife.utils import (CanonicalAtomFeaturizer, CanonicalBondFeaturizer,
                              smiles_to_bigraph)
    from rdkit import Chem
    row, smiles, model_type = item
    if _FEATURIZERS is None:
        _FEATURIZERS = (CanonicalAtomFeaturizer(), CanonicalBondFeaturizer())
    atom, bond = _FEATURIZERS
    try:
        if not isinstance(smiles, str) or not smiles.strip():
            raise ValueError('Missing or empty SMILES')
        mol = Chem.MolFromSmiles(smiles)
        if mol is None or mol.GetNumAtoms() == 0:
            raise ValueError('Invalid or empty molecule')
        graph = smiles_to_bigraph(
            smiles, node_featurizer=atom,
            edge_featurizer=None if model_type == 'gcn' else bond,
            add_self_loop=(model_type == 'gcn'), explicit_hydrogens=False)
        if graph is None:
            raise ValueError('Graph construction returned None')
        # Serialize inside the worker to avoid PyTorch's multiprocessing
        # tensor reducer, which requires a socket-based resource sharer.
        return row, pickle.dumps(graph, protocol=pickle.HIGHEST_PROTOCOL), None
    except Exception as exc:
        return row, None, f'{type(exc).__name__}: {exc}'


def load_submo_model(path, device):
    """Load the original architecture without silently ignoring missing weights."""
    import torch
    from model.gnn_models import GNNMultitask
    kwargs = {'map_location': device}
    if 'weights_only' in inspect.signature(torch.load).parameters:
        kwargs['weights_only'] = False
    model = torch.load(path, **kwargs)
    if not isinstance(model, GNNMultitask):
        raise TypeError('Expected a full SUBMO GNNMultitask checkpoint, not a '
                        'state_dict or AttentiveFPPredictor')
    if model.model_type not in ('gcn', 'attentivefp'):
        raise ValueError('Only upstream GCN and AttentiveFP models are supported')
    return model.to(device).eval()


def generate_embeddings(args, out):
    """Stream CSV chunks; never reconstruct metadata from unfiltered input."""
    import csv
    import multiprocessing
    import pandas as pd
    import torch
    import dgl
    model = load_submo_model(args.model, args.device)
    reserved = {'source_row', 'embedding_row', 'selection_rank'}
    columns = pd.read_csv(args.input, sep=args.sep, nrows=0).columns.tolist()
    if args.smiles_col not in columns:
        raise ValueError(f'Missing SMILES column: {args.smiles_col}')
    if reserved.intersection(columns):
        raise ValueError(f'Input columns must not include {sorted(reserved)}')
    offset = count = 0
    # Spawn workers so CUDA state is never inherited through fork.
    pool = (ProcessPoolExecutor(max_workers=args.workers,
            mp_context=multiprocessing.get_context('spawn'))
            if args.workers > 1 else None)
    try:
        with tempfile.TemporaryDirectory(prefix='.embedding-parts-', dir=out) as tmp:
            parts = []
            with (out / 'metadata.csv').open('w', newline='') as meta_file, \
                 (out / 'rejected.csv').open('w', newline='') as bad_file:
                pd.DataFrame(columns=columns + ['source_row', 'embedding_row']).to_csv(
                    meta_file, index=False)
                bad_writer = csv.writer(bad_file)
                bad_writer.writerow(['source_row', args.smiles_col, 'error'])
                for chunk in pd.read_csv(args.input, sep=args.sep,
                                         chunksize=args.chunk_size, dtype=str, keep_default_na=False):
                    items = [(i, s, model.model_type) for i, s in
                             enumerate(chunk[args.smiles_col].tolist())]
                    results = (pool.map(build_graph, items, chunksize=32)
                               if pool else map(build_graph, items))
                    graphs, rows = [], []

                    def flush():
                        nonlocal count
                        if not graphs:
                            return
                        bg = dgl.batch(graphs).to(args.device)
                        edge = None if model.model_type == 'gcn' else bg.edata['e']
                        try:
                            with torch.no_grad():
                                # task=None returns upstream graph descriptors,
                                # including checkpoint-defined ReLU/normalization.
                                emb = model(bg, bg.ndata['h'], edge_feats=edge)
                        except RuntimeError as exc:
                            raise RuntimeError('Embedding inference failed. The model '
                                'must use upstream canonical atom/bond features; '
                                'a checkpoint trained with AttentiveFP-specific '
                                '39/10 features is incompatible.') from exc
                        arr = emb.detach().cpu().numpy()
                        if arr.ndim != 2 or not np.isfinite(arr).all():
                            raise ValueError('Model produced invalid embeddings')
                        part = Path(tmp) / f'{len(parts):08d}.npy'
                        np.save(part, arr)
                        parts.append(part)
                        meta = chunk.iloc[rows].copy()
                        meta['source_row'] = np.asarray(rows) + offset
                        meta['embedding_row'] = np.arange(count, count + len(rows))
                        meta.to_csv(meta_file, header=False, index=False)
                        count += len(rows)
                        graphs.clear()
                        rows.clear()

                    for row, graph, error in results:
                        if error:
                            bad_writer.writerow([offset + row,
                                                 chunk.iloc[row][args.smiles_col], error])
                        else:
                            graphs.append(pickle.loads(graph))
                            rows.append(row)
                            if len(graphs) >= args.batch_size:
                                flush()
                    flush()
                    offset += len(chunk)
                    print(f'Embedded {count}/{offset} input rows', flush=True)
            if not parts:
                raise ValueError('No valid molecules were found')
            first = np.load(parts[0], mmap_mode='r')
            merged = np.lib.format.open_memmap(out / 'embeddings.npy', mode='w+',
                       dtype=first.dtype, shape=(count, first.shape[1]))
            start = 0
            for part in parts:
                arr = np.load(part, mmap_mode='r')
                merged[start:start + len(arr)] = arr
                start += len(arr)
            merged.flush()
            del merged
    finally:
        if pool:
            pool.shutdown(wait=True)
    return offset, count


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                         formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input', type=Path, required=True, help='CSV or compressed CSV')
    parser.add_argument('--smiles-col', default='smiles')
    parser.add_argument('--sep', default=',', help='Input delimiter')
    parser.add_argument('--model', type=Path, required=True,
                        help='Trusted full-model upstream SUBMO checkpoint')
    parser.add_argument('--output-dir', type=Path, required=True,
                        help='New directory; existing directories are never overwritten')
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument('--n-select', type=int)
    target.add_argument('--fraction', type=float,
                        help='Fraction of valid molecules, rounded down as upstream')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--chunk-size', type=int, default=10000)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--score-block-size', type=int, default=100000)
    parser.add_argument('--device', default='cpu', help='cpu, cuda, or cuda:0')
    args = parser.parse_args()
    for name in ('workers', 'chunk_size', 'batch_size', 'score_block_size'):
        if getattr(args, name) < 1:
            parser.error(f'{name} must be positive')
    if args.n_select is not None and args.n_select < 1:
        parser.error('--n-select must be positive')
    if args.fraction is not None and not 0 < args.fraction <= 1:
        parser.error('--fraction must be in (0, 1]')
    if not args.input.is_file() or not args.model.is_file():
        parser.error('Input and model files must exist')
    import pandas as pd
    import torch
    import dgl
    import dgllife
    from rdkit import rdBase
    from threadpoolctl import threadpool_limits
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with threadpool_limits(limits=1):
        total, valid = generate_embeddings(args, args.output_dir)
        x = np.load(args.output_dir / 'embeddings.npy', mmap_mode='r')
        k = args.n_select if args.n_select is not None else int(valid * args.fraction)
        selected = exact_logdet_greedy(x, k, args.workers, args.score_block_size)
    # Read metadata in chunks so the full source table need not fit in memory.
    ranks = {int(row): rank for rank, row in enumerate(selected)}
    selected_parts = []
    for chunk in pd.read_csv(args.output_dir / 'metadata.csv', chunksize=args.chunk_size,
                             dtype={column: str for column in pd.read_csv(args.input, sep=args.sep, nrows=0).columns},
                             keep_default_na=False):
        subset = chunk[chunk.embedding_row.isin(ranks)].copy()
        subset['selection_rank'] = subset.embedding_row.map(ranks)
        selected_parts.append(subset)
    result = pd.concat(selected_parts, ignore_index=True).sort_values('selection_rank')
    result.to_csv(args.output_dir / 'selected.csv', index=False)
    selected_x = np.lib.format.open_memmap(args.output_dir / 'selected_embeddings.npy',
                    mode='w+', dtype=x.dtype, shape=(k, x.shape[1]))
    for lo in range(0, k, args.batch_size):
        selected_x[lo:lo + args.batch_size] = x[selected[lo:lo + args.batch_size]]
    selected_x.flush()
    digest = hashlib.sha256()
    with args.model.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    manifest = {'arguments': {key: str(value) if isinstance(value, Path) else value
                             for key, value in vars(args).items()},
                'input_rows': total, 'valid_rows': valid, 'selected_rows': k,
                'model_sha256': digest.hexdigest(),
                'objective': 'logdet(I + X_S @ X_S.T)',
                'versions': {'numpy': np.__version__, 'pandas': pd.__version__,
                             'torch': torch.__version__, 'dgl': dgl.__version__,
                             'dgllife': dgllife.__version__, 'rdkit': rdBase.rdkitVersion}}
    (args.output_dir / 'run.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Completed: {args.output_dir}', flush=True)


if __name__ == '__main__':
    main()
