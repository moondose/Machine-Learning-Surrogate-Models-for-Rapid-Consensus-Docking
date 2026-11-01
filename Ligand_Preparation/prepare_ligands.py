#!/usr/bin/env python3
"""Prepare 3D ligand structures from a sampled molecular CSV.

Dependencies:
    Python 3.9+, RDKit, and Open Babel when --mol2 is enabled.
    Open Babel's obabel executable must be on PATH or supplied via --obabel.

Input:
    An uncompressed CSV with unique zincid values and a smiles column.
    Molecule IDs must follow the ZINC identifier format: ZINC followed by digits.

Workflow:
    Parse SMILES, add explicit hydrogens, and generate one conformer with
    ETKDGv3 using a fixed seed. Retry with random coordinates if needed.
    Optimize with MMFF, or UFF when MMFF parameters are unavailable.
    Process molecules in parallel and optionally convert SDF to MOL2 using
    Open Babel. Record preparation failures in the output report.

Outputs:
    sdf/<zincid>.sdf                 Prepared 3D structures.
    mol2/<zincid>.mol2               Optional MOL2 structures.
    preparation_report.csv          Per-molecule status and failure details.
    sdf_ligands.txt                  Absolute paths for successful molecules.
    mol2_ligands.txt                 Optional MOL2 paths for successful molecules.
    summary.json                    Requested, successful, and failed counts.

Usage:
    python prepare_ligands.py --input selected.csv --output-dir ligands \
        --workers 8 --seed 42

    python prepare_ligands.py --input selected.csv --output-dir ligands_mol2 \
        --workers 8 --seed 42 --mol2 --obabel /path/to/obabel

Choose a new output directory for each run. The process exits with an error
if any molecule fails; inspect preparation_report.csv for individual results.
"""
import argparse
import csv
import json
import re
import subprocess
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem


def prepare(task):
    row, root, seed, mol2, obabel = task
    zid = row['zincid']
    result = {'zincid': zid, 'status': 'failed', 'detail': ''}
    produced = []
    try:
        if not re.fullmatch(r'ZINC\d+', zid):
            raise ValueError('Unsafe or unexpected ZINC ID')
        mol = Chem.MolFromSmiles(row['smiles'])
        if mol is None:
            raise ValueError('Invalid SMILES')
        mol = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = seed
        params.numThreads = 1
        if AllChem.EmbedMolecule(mol, params) != 0:
            params.useRandomCoords = True
            if AllChem.EmbedMolecule(mol, params) != 0:
                raise ValueError('3D embedding failed')
        if AllChem.MMFFHasAllMoleculeParams(mol):
            # Allow up to 1000 MMFF optimization iterations per molecule.
            optimization = AllChem.MMFFOptimizeMolecule(mol, maxIters=1000)
        elif AllChem.UFFHasAllMoleculeParams(mol):
            # UFF fallback also allows up to 1000 optimization iterations.
            optimization = AllChem.UFFOptimizeMolecule(mol, maxIters=1000)
        else:
            raise ValueError('No complete MMFF or UFF parameters')
        if optimization != 0:
            raise ValueError(f'Optimization did not converge: {optimization}')
        mol.SetProp('_Name', zid)
        mol.SetProp('source_smiles', row['smiles'])
        sdf = Path(root)/'sdf'/f'{zid}.sdf'
        produced.append(sdf)
        with Chem.SDWriter(str(sdf)) as writer:
            writer.write(mol)
        check = Chem.SDMolSupplier(str(sdf), removeHs=False)[0]
        if check is None or not check.GetConformer().Is3D():
            raise ValueError('SDF validation failed')
        if mol2:
            target = Path(root)/'mol2'/f'{zid}.mol2'
            produced.append(target)
            proc = subprocess.run([obabel, str(sdf), '-O', str(target)], capture_output=True, text=True, timeout=120)
            if proc.returncode or not target.exists() or target.stat().st_size == 0:
                raise ValueError(f'MOL2 conversion failed: {proc.stderr[-500:]}')
            converted = Chem.MolFromMol2File(str(target), removeHs=False)
            if converted is None or converted.GetNumHeavyAtoms() != mol.GetNumHeavyAtoms():
                raise ValueError('MOL2 parsing or heavy atom count validation failed')
        result['status'] = 'ok'
    except Exception as exc:
        for path in produced:
            path.unlink(missing_ok=True)
        result['detail'] = str(exc)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, default=Path(__file__).parent/'selected.csv')
    p.add_argument('--output-dir', type=Path, default=Path(__file__).parent/'ligands')
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--mol2', action='store_true')
    p.add_argument('--obabel', default='obabel')
    args = p.parse_args()
    if args.output_dir.exists():
        p.error('Output directory exists; choose a new --output-dir')
    if args.workers < 1 or not 0 <= args.seed < 2147483647:
        p.error('Require positive workers and seed in [0, 2147483647)')
    if args.mol2:
        executable = shutil.which(args.obabel)
        if executable is None:
            p.error('Open Babel executable not found; install it or specify --obabel')
        args.obabel = executable
    with args.input.open(newline='') as handle:
        reader = csv.DictReader(handle)
        if not {'zincid', 'smiles'}.issubset(reader.fieldnames or []):
            p.error('Input CSV requires zincid and smiles columns')
        rows = list(reader)
    if not rows or len({r['zincid'] for r in rows}) != len(rows):
        p.error('Input must contain unique molecule IDs')
    (args.output_dir/'sdf').mkdir(parents=True)
    if args.mol2:
        (args.output_dir/'mol2').mkdir()
    tasks = ((r, str(args.output_dir), args.seed, args.mol2, args.obabel) for r in rows)
    successful = []
    with (args.output_dir/'preparation_report.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['zincid', 'status', 'detail'])
        writer.writeheader()
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for i, result in enumerate(pool.map(prepare, tasks, chunksize=10), 1):
                writer.writerow(result)
                if result['status'] == 'ok':
                    successful.append(result['zincid'])
                if i % 1000 == 0:
                    handle.flush()
                    print(f'{i}/{len(rows)} processed; {len(successful)} successful', flush=True)
    for fmt in (['sdf', 'mol2'] if args.mol2 else ['sdf']):
        (args.output_dir/f'{fmt}_ligands.txt').write_text(''.join(
            str((args.output_dir/fmt/f'{zid}.{fmt}').resolve())+'\n' for zid in successful))
    summary = {'requested': len(rows), 'successful': len(successful), 'failed': len(rows)-len(successful)}
    (args.output_dir/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary))
    if summary['failed']:
        raise SystemExit('Preparation incomplete: inspect preparation_report.csv before docking.')


if __name__ == '__main__':
    main()
