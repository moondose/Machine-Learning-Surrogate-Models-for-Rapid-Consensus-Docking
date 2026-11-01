#!/usr/bin/env python3
"""Extract the lowest-SCORE rDock pose for consensus target construction.

Requires Python 3.9+ and RDKit. The input is either a rdock_batch.py output
(work/<original ligand filename>/poses.sd) or a historical directory containing
<molecule_id>_rdock.sd/.sdf files. Input directories are scanned recursively.

For each molecule, select the valid SD record with the lowest finite SCORE.
Ties retain the first record. Original record indices are preserved even when
other records have missing scores. Missing scores, unreadable records and
nonfinite coordinates are excluded from consideration. Molecules without a
usable scored pose are recorded in errors.csv. Scores use rDock's native
SCORE field, not RFScore rescoring or SCORE.INTER.

Example:
    python preprocess_rdock.py --base-dir /path/to/rdock_output \
        --output-dir /path/to/new_rdock_preprocessed

Outputs:
    molecules/<id>_pose0.sdf: one selected pose with its original coordinates.
    csvs/Results_scoring.csv: score;1;rdock;pose_index;id;/molecules/<id>_pose0.sdf
    pose_selection.csv: source file, score, original zero-based record index,
        and counts of complete and usable scored records.
    errors.csv: excluded molecules and reasons.
    preparation_manifest.json: parameters and processing counts.

The SCORE property and all other selected-record properties are retained.
The output directory must be new. Source docking files are never modified.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import re

from rdkit import Chem


def index_sources(base):
    """Recognize batch and historical output without ingesting input ligands."""
    sources = {}
    for path in sorted(base.rglob('*')):
        if not path.is_file() or path.is_symlink() or path.suffix.lower() not in {'.sd', '.sdf'}:
            continue
        if path.name == 'poses.sd' and path.parent.suffix.lower() in {'.sdf', '.sd'}:
            identifier = path.parent.stem
        else:
            match = re.fullmatch(r'(.+)_rdock', path.stem, flags=re.IGNORECASE)
            if not match:
                continue
            identifier = match.group(1)
        if identifier in sources:
            raise ValueError(f'Duplicate molecule ID: {identifier}')
        sources[identifier] = path
    config_path = base / 'run_config.json'
    if config_path.is_file():
        config = json.loads(config_path.read_text())
        for filename in config.get('ligand_files', []):
            ligand = Path(filename)
            relative = ligand.relative_to(Path(config['ligands']))
            sources.setdefault(ligand.stem, base / 'work' / relative / 'poses.sd')
    if not sources:
        raise ValueError('No batch poses.sd or historical *_rdock.sd/.sdf files found')
    return sources


def select_pose(path):
    """Keep source record indices and validate score/coordinates together."""
    best = None
    complete = usable = 0
    with path.open('rb') as handle:
        # Count terminated records independently of RDKit parse success.
        complete = sum(line.strip() == b'$$$$' for line in handle)
    supplier = Chem.SDMolSupplier(str(path), removeHs=False, sanitize=False)
    for index, mol in enumerate(supplier):
        if index >= complete or mol is None or not mol.HasProp('SCORE'):
            continue
        try:
            score = float(mol.GetProp('SCORE'))
            if not math.isfinite(score) or mol.GetNumAtoms() == 0 or mol.GetNumConformers() == 0:
                continue
            xyz = mol.GetConformer().GetPositions()
            if not all(math.isfinite(float(v)) for row in xyz for v in row):
                continue
        except (ValueError, RuntimeError):
            continue
        usable += 1
        if best is None or score < best[0]:
            best = score, index, mol
    if best is None:
        raise ValueError('No complete readable pose with finite SCORE and coordinates')
    return best, complete, usable



def failed_batch_ids(base):
    """Read batch completion states before considering saved pose artifacts."""
    report = Path(base) / 'results.csv'
    if not report.is_file():
        return set()
    with report.open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    return {Path(row['ligand']).stem for row in rows if row['status'] != 'success'}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                    formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--base-dir', '-base-dir', type=Path, required=True)
    parser.add_argument('--output-dir', '-output-dir', type=Path, required=True)
    args = parser.parse_args()
    base = args.base_dir.resolve()
    if not base.is_dir():
        parser.error('Input directory does not exist')
    if args.output_dir.exists():
        parser.error('Choose a new output directory')
    sources = index_sources(base)
    (args.output_dir / 'molecules').mkdir(parents=True)
    (args.output_dir / 'csvs').mkdir()
    failed = failed_batch_ids(base)
    counts = dict(input_molecules=len(sources), selected=0, excluded=0)
    with (args.output_dir / 'csvs/Results_scoring.csv').open('w') as scoring, \
         (args.output_dir / 'pose_selection.csv').open('w', newline='') as audit, \
         (args.output_dir / 'errors.csv').open('w', newline='') as errors:
        writer = csv.writer(audit)
        writer.writerow(['molecule_id', 'source_sd', 'pose_index', 'score',
                         'complete_records', 'usable_scored_records'])
        err = csv.writer(errors)
        err.writerow(['molecule_id', 'source_sd', 'reason'])
        for identifier, path in sorted(sources.items()):
            target = args.output_dir / 'molecules' / f'{identifier}_pose0.sdf'
            try:
                if identifier in failed:
                    raise ValueError('Docking job did not complete successfully')
                (score, index, mol), complete, usable = select_pose(path)
                with Chem.SDWriter(str(target)) as handle:
                    handle.write(mol)
                writer.writerow([identifier, str(path), index, score, complete, usable])
                scoring.write(f'{score};1;rdock;{index};{identifier};/molecules/{target.name}\n')
                counts['selected'] += 1
            except (OSError, ValueError, RuntimeError) as exc:
                target.unlink(missing_ok=True)
                counts['excluded'] += 1
                err.writerow([identifier, str(path), str(exc)])
    manifest = dict(counts, input_directory=str(base), output_directory=str(args.output_dir.resolve()),
                    selection='lowest finite SCORE; first saved record wins ties',
                    pose_index='zero-based original SD record index')
    (args.output_dir / 'preparation_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(counts, indent=2))


if __name__ == '__main__':
    main()
