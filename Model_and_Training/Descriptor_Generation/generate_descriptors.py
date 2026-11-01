#!/usr/bin/env python3
"""Generate the trained Morgan + RDKit 2D/3D feature matrix from prepared SDF.

Requires Python 3.9+, RDKit and NumPy. The adjacent feature_names.csv fixes the
training column order: 217 RDKit descriptors, 2048 Morgan radius-2 bits, and
nine RDKit 3D descriptors (2274 float32 columns).

Example:
    python generate_descriptors.py --input /path/to/prepared/sdf \
        --output-dir /path/to/descriptors --workers 8 --batch-size 128

Inputs:
    One .sdf/.sd/.sdf.gz/.sd.gz file or a directory scanned recursively.
    Multi-record files are supported. Use prepared 3D structures. Explicit
    hydrogens and the supplied conformer are retained (removeHs=False).
    No conformer generation, optimization, normalization or scaling is applied.
    Auto IDs use the SDF _Name, falling back to the filename for uncompressed
    files with an empty name. --id-source filename uses the filename for each
    record; --id-source name requires a nonempty SDF name. IDs are uppercased.
    Keep input IDs consistent with those used to construct training targets.

Optional:
    --exclude-ids ids.csv --exclude-id-column zinc_id
    --id-source filename
    --feature-names /path/to/feature_names.csv
    --max-molecules 10  (limit the number of source records inspected)

Parallel computation uses bounded batches of serialized molecules. Results are
written in deterministic source-file/record order. The final matrix is created
with a memory map; the dataset is never assembled into a full in-memory matrix.
A temporary raw matrix is removed after conversion to NPY. Allow disk space
for both raw and NPY matrices during finalization.

Outputs:
    X_features.npy: float32 matrix in feature_names.csv column order.
    zinc_ids_aligned.csv: row_index, zinc_id, source_file, source_record.
    feature_names.csv: exact feature schema used for this run.
    excluded.csv: invalid records, duplicate IDs and explicit ID exclusions.
    descriptor_failures.csv: per-molecule descriptor failures replaced with zero.
    generation_summary.json: counts, parameters, versions and schema hash.

Nonfinite descriptor values and descriptor exceptions become zero, matching the
existing training workflow. Invalid molecules, missing 3D conformers and
nonfinite coordinates are excluded. Duplicate IDs retain the first valid input
record. Morgan settings follow GetMorganGenerator(radius=2, fpSize=2048), with
RDKit's defaults for the other fingerprint options.
"""
import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import csv
import gzip
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import tempfile

# Avoid numerical-library thread multiplication across worker processes.
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import Descriptors, Descriptors3D, rdFingerprintGenerator

THREE_D = ['RadiusOfGyration', 'Eccentricity', 'InertialShapeFactor', 'NPR1',
           'NPR2', 'PMI1', 'PMI2', 'PMI3', 'SpherocityIndex']
_FUNCTIONS = None
_GENERATOR = None


def load_schema(path):
    with path.open(newline='') as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ['feature_name']:
            raise ValueError('Feature CSV must have one column named feature_name')
        names = [row['feature_name'] for row in reader]
    available = dict(Descriptors.descList)
    if len(names) != 2274 or len(set(names)) != len(names):
        raise ValueError('Expected 2274 unique training feature names')
    functions = []
    for name in names:
        if name.startswith('rdkit_'):
            key = name[len('rdkit_'):]
            if key not in available:
                raise ValueError(f'Installed RDKit lacks training descriptor: {key}')
            functions.append((name, available[key]))
        elif name.startswith('rdkit3d_') and name[len('rdkit3d_'):] in THREE_D:
            functions.append((name, getattr(Descriptors3D, name[len('rdkit3d_'):])))
        elif name.startswith('morgan_r2_'):
            functions.append((name, int(name[len('morgan_r2_'):])))
        else:
            raise ValueError(f'Unsupported training feature: {name}')
    if [value for name, value in functions if name.startswith('morgan_')] != list(range(2048)):
        raise ValueError('Expected all 2048 Morgan bits in training order')
    if sum(name.startswith('rdkit_') for name in names) != 217:
        raise ValueError('Expected 217 RDKit 2D descriptor columns')
    if sum(name.startswith('rdkit3d_') for name in names) != 9:
        raise ValueError('Expected nine RDKit 3D descriptor columns')
    return names, functions


def initialize(schema):
    global _FUNCTIONS, _GENERATOR
    _, _FUNCTIONS = load_schema(Path(schema))
    _GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def compute_batch(batch):
    """Compute the original descriptor formula and retain failure audit details."""
    values, metadata, failures = [], [], []
    for identifier, source, record, serialized in batch:
        mol = Chem.Mol(serialized)
        fp = _GENERATOR.GetFingerprintAsNumPy(mol)
        vector = []
        failed = []
        for name, function in _FUNCTIONS:
            if name.startswith('morgan_'):
                vector.append(fp[function])
            else:
                try:
                    value = function(mol)
                    value = float(value) if value is not None else 0.0
                except Exception:
                    value = 0.0
                    failed.append(name)
                vector.append(value)
        with np.errstate(over='ignore', invalid='ignore'):
            array = np.asarray(vector, dtype=np.float32)
        nonfinite = ~np.isfinite(array)
        failed.extend(_FUNCTIONS[i][0] for i in np.flatnonzero(nonfinite))
        array[nonfinite] = 0.0
        values.append(array)
        metadata.append((identifier, source, record))
        if failed:
            failures.append((identifier, source, record, '|'.join(failed)))
    return np.asarray(values, dtype=np.float32), metadata, failures


def records(paths, args, excluded_ids, report, counts):
    """Parse once, preserve hydrogens, and stream valid molecules to workers."""
    seen = set()
    for path in paths:
        opener = gzip.open if path.suffix.lower() == '.gz' else open
        with opener(path, 'rb') as handle:
            supplier = Chem.ForwardSDMolSupplier(handle, sanitize=True, removeHs=False)
            for index, mol in enumerate(supplier):
                if args.max_molecules is not None and counts['records'] >= args.max_molecules:
                    return
                counts['records'] += 1
                identifier = ''
                try:
                    if mol is None:
                        raise ValueError('SDF parsing failed')
                    name = mol.GetProp('_Name').strip() if mol.HasProp('_Name') else ''
                    filename = path.with_suffix('').stem if path.suffix.lower() == '.gz' else path.stem
                    if args.id_source == 'filename':
                        # Remove the two suffixes explicitly for compressed SD files.
                        filename = path.with_suffix('').stem if path.suffix.lower() == '.gz' else path.stem
                        identifier = filename
                    else:
                        identifier = name
                        if not identifier and args.id_source == 'auto' and path.suffix.lower() != '.gz':
                            identifier = filename
                    identifier = identifier.strip().upper()
                    if not identifier:
                        raise ValueError('Missing molecule ID')
                    if identifier in excluded_ids:
                        raise ValueError('ID excluded by --exclude-ids')
                    if mol.GetNumAtoms() == 0 or not mol.GetNumConformers() or not mol.GetConformer().Is3D():
                        raise ValueError('Missing 3D conformer')
                    if not np.isfinite(mol.GetConformer().GetPositions()).all():
                        raise ValueError('Nonfinite coordinates')
                    if identifier in seen:
                        raise ValueError('Duplicate molecule ID; first valid record retained')
                    seen.add(identifier)
                    # Preserve double-precision coordinates across process boundaries.
                    # RDKit's default binary serialization stores float32 coordinates.
                    flags = Chem.PropertyPickleOptions.AllProps | Chem.PropertyPickleOptions.CoordsAsDouble
                    yield identifier, str(path), index, mol.ToBinary(flags)
                except ValueError as exc:
                    counts['excluded'] += 1
                    report.writerow([identifier, str(path), index, str(exc)])


def batches(items, size):
    batch = []
    for item in items:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def ordered_results(items, args, schema):
    """Bound pending work to twice the worker count, retaining source row order."""
    if args.workers == 1:
        initialize(schema)
        for batch in items:
            yield compute_batch(batch)
        return
    with ProcessPoolExecutor(max_workers=args.workers,
             mp_context=multiprocessing.get_context('spawn'),
             initializer=initialize, initargs=(str(schema),)) as pool:
        pending = deque()
        for batch in items:
            pending.append(pool.submit(compute_batch, batch))
            if len(pending) >= 2 * args.workers:
                yield pending.popleft().result()
        while pending:
            yield pending.popleft().result()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                    formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--feature-names', type=Path,
                        default=Path(__file__).with_name('feature_names.csv'))
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--exclude-ids', type=Path)
    parser.add_argument('--exclude-id-column', default='zinc_id')
    parser.add_argument('--id-source', choices=['auto', 'name', 'filename'], default='auto')
    parser.add_argument('--max-molecules', type=int)
    args = parser.parse_args()
    if args.workers < 1 or args.batch_size < 1 or (args.max_molecules is not None and args.max_molecules < 1):
        parser.error('Workers, batch size and molecule limit must be positive')
    if args.output_dir.exists():
        parser.error('Choose a new output directory')
    names, _ = load_schema(args.feature_names)
    source = args.input.resolve()
    candidates = source.rglob('*') if source.is_dir() else [source]
    paths = sorted(path for path in candidates if path.is_file() and
                   path.name.lower().endswith(('.sdf', '.sd', '.sdf.gz', '.sd.gz')))
    if not paths:
        parser.error('No supported SDF/SD files found')
    excluded_ids = set()
    if args.exclude_ids:
        with args.exclude_ids.open(newline='') as handle:
            reader = csv.DictReader(handle)
            if args.exclude_id_column not in (reader.fieldnames or []):
                parser.error('ID column missing from exclusion CSV')
            excluded_ids = {row[args.exclude_id_column].strip().upper() for row in reader}
    counts = dict(records=0, generated=0, excluded=0, molecules_with_zero_filled_descriptors=0)
    args.output_dir.mkdir(parents=True)
    with (args.output_dir / 'excluded.csv').open('w', newline='') as rejects, \
         (args.output_dir / 'zinc_ids_aligned.csv').open('w', newline='') as aligned, \
         (args.output_dir / 'descriptor_failures.csv').open('w', newline='') as failure_log, \
         tempfile.TemporaryDirectory(prefix='.descriptor-work-', dir=args.output_dir) as tmp:
        report = csv.writer(rejects)
        report.writerow(['zinc_id', 'source_file', 'source_record', 'reason'])
        writer = csv.writer(aligned)
        writer.writerow(['row_index', 'zinc_id', 'source_file', 'source_record'])
        failure_writer = csv.writer(failure_log)
        failure_writer.writerow(['zinc_id', 'source_file', 'source_record', 'zero_filled_features'])
        tasks = batches(records(paths, args, excluded_ids, report, counts), args.batch_size)
        raw_path = Path(tmp) / 'features.bin'
        with raw_path.open('wb') as raw:
            for matrix, metadata, failures in ordered_results(tasks, args, args.feature_names.resolve()):
                matrix.tofile(raw)
                writer.writerows((counts['generated'] + i, *row) for i, row in enumerate(metadata))
                failure_writer.writerows(failures)
                counts['molecules_with_zero_filled_descriptors'] += len(failures)
                counts['generated'] += len(matrix)
                print(f"Generated {counts['generated']} rows", flush=True)
        if not counts['generated']:
            raise ValueError('No valid molecules with 3D conformers; inspect excluded.csv')
        shape = (counts['generated'], len(names))
        raw_matrix = np.memmap(raw_path, mode='r', dtype=np.float32, shape=shape)
        output = np.lib.format.open_memmap(args.output_dir / 'X_features.npy',
                                          mode='w+', dtype=np.float32, shape=shape)
        for start in range(0, shape[0], args.batch_size):
            output[start:start + args.batch_size] = raw_matrix[start:start + args.batch_size]
        output.flush()
        del output, raw_matrix
    with (args.output_dir / 'feature_names.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['feature_name'])
        writer.writerows((name,) for name in names)
    summary = dict(counts, features=len(names), source_files=len(paths),
                   rdkit_version=rdBase.rdkitVersion, numpy_version=np.__version__,
                   morgan_radius=2, morgan_bits=2048,
                   hydrogen_policy='removeHs=False; explicit hydrogens retained',
                   conformer_policy='supplied 3D conformer; no generation or optimization',
                   feature_schema_sha256=hashlib.sha256(('\n'.join(names)+'\n').encode()).hexdigest(),
                   arguments={key: str(value) if isinstance(value, Path) else value
                              for key, value in vars(args).items()})
    (args.output_dir / 'generation_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(counts, indent=2))


if __name__ == '__main__':
    main()
