#!/usr/bin/env python3
"""Select the first usable, in-pocket DiffDock pose in ascending rank order.

Input layouts are detected automatically:
    diffdock_batch.py: run_config.json plus shard_*/poses/VS_DD_*.sdf.
    MetaScreener: VS_DD_VS_FOLDER_*/molecules/VS_DD_ZINC*_rank*.sdf.

For batch output, run_config.json maps complex names back to ligand filename
stems. Molecule identifiers must be unique. Rank 1 is checked first, followed
by rank 2 and subsequent ranks. A pose must parse, have finite coordinates,
lie within the centroid distance threshold, and have finite confidence > -999.
Molecules with no qualifying pose are excluded and recorded in errors.csv.
The centroid uses RDKit's default hydrogen removal, matching the original
preprocess_diffdock_rank_fallback.py. Coordinates are never aligned or fitted.

Requires Python 3.9+, RDKit and NumPy.
Example:
    python preprocess_diffdock.py --base-dir /path/to/diffdock_output \
        --output-dir /path/to/new_preprocessed --pocket-center 23.176 5.742 13.697 \
        --distance-threshold 20

Outputs:
    molecules/<id>_pose0.sdf: the selected pose, copied without coordinate changes.
    csvs/Results_scoring.csv: legacy consensus score rows (confidence is retained).
    pose_selection.csv: source path, actual selected rank, confidence and distance.
    errors.csv: excluded molecules and pose-level rejection reasons.
    preparation_manifest.json: arguments and processing counts.

The compatibility scoring table keeps rank placeholders at 1; the actual
DiffDock rank is stored in pose_selection.csv. Pocket coordinates must use the
same receptor frame as the input poses. All comments and messages are English.
"""
import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import re
import shutil

import numpy as np
from rdkit import Chem

PATTERN = re.compile(r'VS_DD_(.+)_rank(\d+)_confidence(.+)\.sdf$')


def index_poses(base):
    """Map source poses to original molecule IDs without merging duplicate ranks."""
    index = defaultdict(dict)
    config = base / 'run_config.json'
    if config.is_file():
        entries = json.loads(config.read_text())['ligands']
        identifiers = set()
        sources = []
        for entry in entries:
            identifier = Path(entry['ligand']).stem
            if identifier in identifiers:
                raise ValueError(f'Duplicate ligand ID: {identifier}')
            identifiers.add(identifier)
            index[identifier]  # Include molecules with no saved poses in the report.
            sources.append((base / f"shard_{int(entry['shard']):04d}" / 'poses',
                            entry['complex_name'], identifier))
        # Scan each shard once, even for large libraries.
        mapping = {(folder, complex_name): identifier
                   for folder, complex_name, identifier in sources}
        folders = sorted({folder for folder, _, _ in sources})
        layout = 'diffdock_batch'
    else:
        folders = sorted(base.glob('VS_DD_VS_FOLDER_*/molecules'))
        mapping = None
        layout = 'metascreener'
    for folder in folders:
        for path in sorted(folder.glob('VS_DD_*.sdf')):
            match = PATTERN.fullmatch(path.name)
            if not match:
                continue
            name, rank, _ = match.groups()
            identifier = mapping.get((folder, name)) if mapping is not None else name
            if identifier is None:
                continue
            if not re.fullmatch(r'[A-Za-z0-9_.-]+', identifier) or identifier in ('.', '..'):
                raise ValueError(f'Unsupported molecule ID: {identifier}')
            rank = int(rank)
            if rank < 1 or rank in index[identifier]:
                raise ValueError(f'Duplicate or invalid rank {rank}: {identifier}')
            index[identifier][rank] = path
    if not index:
        raise ValueError('No DiffDock molecules found in the supported input layouts')
    return index, layout


def select_pose(poses, center, threshold):
    """Return the first qualified rank and the rejection audit accumulated so far."""
    rejected = []
    for rank, path in sorted(poses.items()):
        try:
            mol = next(iter(Chem.SDMolSupplier(str(path))))
            if mol is None or mol.GetNumConformers() == 0 or not mol.GetConformer().Is3D():
                raise ValueError('Unreadable molecule or missing coordinates')
            coordinates = mol.GetConformer().GetPositions()
            if not len(coordinates) or not np.isfinite(coordinates).all():
                raise ValueError('Empty or nonfinite coordinates')
            distance = float(np.linalg.norm(coordinates.mean(axis=0) - center))
            if distance > threshold:
                raise ValueError(f'Centroid outside pocket: {distance:.6f} Angstrom')
            confidence = float(PATTERN.fullmatch(path.name).group(3))
            if not math.isfinite(confidence) or confidence <= -999:
                raise ValueError('Invalid confidence')
            return dict(rank=rank, source_sdf=str(path), confidence=confidence,
                        distance=distance), rejected
        except (OSError, ValueError, StopIteration, RuntimeError) as exc:
            rejected.append(f'rank {rank}: {exc}')
    return None, rejected



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
    parser.add_argument('--pocket-center', '-pocket-center', type=float, nargs=3,
                        required=True, metavar=('X', 'Y', 'Z'))
    parser.add_argument('--distance-threshold', '-distance-threshold', type=float, default=20.0)
    args = parser.parse_args()
    if not all(math.isfinite(v) for v in args.pocket_center):
        parser.error('Pocket center coordinates must be finite')
    if not math.isfinite(args.distance_threshold) or args.distance_threshold <= 0:
        parser.error('Distance threshold must be finite and positive')
    args.base_dir = args.base_dir.resolve()
    if not args.base_dir.is_dir():
        parser.error('Input directory does not exist')
    if args.output_dir.exists():
        parser.error('Choose a new output directory')
    index, layout = index_poses(args.base_dir)
    (args.output_dir / 'molecules').mkdir(parents=True)
    (args.output_dir / 'csvs').mkdir()
    failed = failed_batch_ids(args.base_dir)
    counts = dict(input_molecules=len(index), selected=0, lower_rank_selected=0, excluded=0)
    with (args.output_dir / 'pose_selection.csv').open('w', newline='') as audit, \
         (args.output_dir / 'errors.csv').open('w', newline='') as errors, \
         (args.output_dir / 'csvs/Results_scoring.csv').open('w') as scores:
        writer = csv.DictWriter(audit, fieldnames=['molecule_id', 'rank', 'source_sdf',
                                                  'confidence', 'distance'])
        writer.writeheader()
        err = csv.writer(errors)
        err.writerow(['molecule_id', 'reason'])
        for identifier, poses in sorted(index.items()):
            if identifier in failed:
                counts['excluded'] += 1
                err.writerow([identifier, 'Docking job did not complete successfully'])
                continue
            selected, rejected = select_pose(poses, np.array(args.pocket_center), args.distance_threshold)
            if selected is None:
                counts['excluded'] += 1
                err.writerow([identifier, '; '.join(rejected) or 'No saved poses'])
                continue
            filename = f'{identifier}_pose0.sdf'
            shutil.copyfile(selected['source_sdf'], args.output_dir / 'molecules' / filename)
            writer.writerow(dict(molecule_id=identifier, **selected))
            scores.write(f"{selected['confidence']};1;diffdock;1;{identifier};/molecules/{filename}\n")
            counts['selected'] += 1
            counts['lower_rank_selected'] += selected['rank'] > 1
    manifest = dict(counts, layout=layout, arguments={k: str(v) if isinstance(v, Path) else v
                                                    for k, v in vars(args).items()},
                    selection='first usable pose in ascending rank within centroid distance threshold')
    (args.output_dir / 'preparation_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(counts, indent=2))


if __name__ == '__main__':
    main()
