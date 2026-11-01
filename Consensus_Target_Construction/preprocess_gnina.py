#!/usr/bin/env python3
"""Select the lowest-minimizedAffinity GNINA pose and its matching score.

Supports gnina_batch.py output (poses/ plus run_config.json/results.csv) and
historical MetaScreener output (energies/<batch>/*.json and matching SDFs).
Failed batch jobs are excluded even if a partial pose file exists. Among saved
complete poses, select the lowest finite minimizedAffinity; ties keep
the first record. Missing or invalid scored poses are excluded without fallback.
CNNscore and CNNaffinity are optional audit fields, not selection criteria.

Usage:
    python preprocess_gnina.py --base-dir /path/to/gnina_output \
        --output-dir /path/to/gnina_preprocessed

Optional legacy consensus update:
    --consensus-input /path/to/old_input.txt --output-input /path/to/new_input.txt
    Updates only the GN column and processes the old input's molecule IDs.

Outputs: molecules/<id>_pose0.sdf, csvs/Results_scoring.csv, pose_selection.csv,
errors.csv and preparation_manifest.json. Preserve original SD record indices.
For duplicate historical runs, choose the lowest numeric source folder, then
lexical path. Batch ligand IDs must be unique. Requires Python 3.9+ and RDKit.
"""
import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import re

from rdkit import Chem


def fix_sdf_header(content):
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if 'V2000' in line or 'V3000' in line:
            return '\n'.join(lines[i-3:]) if i > 3 else content
    return content


def value(block, tag):
    # SD headers may contain a record identifier after <tag>, e.g. "(1)".
    match = re.search(r'^>\s*<' + re.escape(tag) + r'>[^\r\n]*\r?\n([^\r\n]+)', block, re.MULTILINE)
    if not match or not math.isfinite(result := float(match[1].strip())):
        raise ValueError(f'Missing/nonfinite {tag}')
    return result


def optional_value(block, tag):
    try:
        return value(block, tag)
    except ValueError:
        return None


def select_pose(content):
    """Select the minimum score first; exclude an unreadable winning pose."""
    candidates = []
    for index, block in enumerate(content.split('$$$$')[:-1]):
        fixed = fix_sdf_header(block.rstrip())
        try:
            energy = value(fixed, 'minimizedAffinity')
        except ValueError:
            continue
        candidates.append((energy, index, fixed))
    if not candidates:
        raise ValueError('No complete pose with finite minimizedAffinity; excluded')
    energy, index, block = min(candidates, key=lambda item: (item[0], item[1]))
    mol = Chem.MolFromMolBlock(block, sanitize=False, removeHs=False)
    if mol is None or not mol.GetNumAtoms() or not mol.GetNumConformers():
        raise ValueError('Minimum-affinity pose is unreadable; excluded without fallback')
    if not all(math.isfinite(float(v)) for xyz in mol.GetConformer().GetPositions() for v in xyz):
        raise ValueError('Minimum-affinity pose has nonfinite coordinates; excluded')
    return (block+'\n$$$$\n', index, energy,
            optional_value(block, 'CNNscore'), optional_value(block, 'CNNaffinity'))


def index_sources(base):
    index = defaultdict(list)
    failures = {}
    config_path = base/'run_config.json'
    if config_path.is_file() and (base/'poses').is_dir():
        config = json.loads(config_path.read_text())
        rows = {}
        if (base/'results.csv').is_file():
            with (base/'results.csv').open(newline='') as handle:
                rows = {row['ligand']: row for row in csv.DictReader(handle)}
        seen = set()
        for filename in config['ligand_files']:
            ligand = Path(filename)
            identifier = ligand.stem
            if identifier in seen:
                raise ValueError(f'Duplicate ligand ID: {identifier}')
            seen.add(identifier)
            index[identifier]
            if (base/'results.csv').is_file() and (filename not in rows or rows[filename]['status'] != 'success'):
                failures[identifier] = 'Docking job did not complete successfully'
                continue
            relative = ligand.relative_to(Path(config['ligands']))
            index[identifier].append(base/'poses'/relative)
        return index, failures, 'gnina_batch'
    for path in (base/'energies').glob('*/*.json'):
        index[path.stem].append(base/'molecules'/path.parent.name/(path.stem+'.sdf'))
    if not index:
        raise ValueError('No supported GNINA batch or MetaScreener inputs found')
    return index, failures, 'metascreener'


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--base-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--consensus-input', type=Path)
    parser.add_argument('--output-input', type=Path)
    args = parser.parse_args()
    if bool(args.consensus_input) != bool(args.output_input):
        parser.error('Supply --consensus-input and --output-input together')
    if args.output_dir.exists() or (args.output_input and args.output_input.exists()):
        parser.error('Choose new output paths')
    base = args.base_dir.resolve()
    index, failures, layout = index_sources(base)
    consensus = {}
    header = None
    if args.consensus_input:
        with args.consensus_input.open() as handle:
            header = next(handle).strip().split(',')
            if header.count('GN') != 1:
                raise ValueError('Expected one GN column')
            for line in handle:
                parts = line.split()
                if not parts:
                    continue
                if len(parts) != len(header)*3+1 or parts[-1] in consensus:
                    raise ValueError('Malformed or duplicate consensus input row')
                consensus[parts[-1]] = parts
    identifiers = sorted(consensus if header else index)
    (args.output_dir/'molecules').mkdir(parents=True)
    (args.output_dir/'csvs').mkdir()
    summary = dict(layout=layout, input_compounds=len(identifiers), written=0, errors=0,
                   selected_nonfirst_pose=0, duplicate_ids=0)
    updates = []
    with (args.output_dir/'csvs/Results_scoring.csv').open('w') as scores, \
         (args.output_dir/'pose_selection.csv').open('w', newline='') as audit, \
         (args.output_dir/'errors.csv').open('w', newline='') as errors:
        writer = csv.writer(audit)
        writer.writerow(['zinc_id','raw_sdf','raw_candidates','selected_pose_1based',
                         'minimizedAffinity','CNNscore','CNNaffinity'])
        err = csv.writer(errors); err.writerow(['zinc_id','error'])
        for identifier in identifiers:
            output = args.output_dir/'molecules'/f'{identifier}_pose0.sdf'
            try:
                if not re.fullmatch(r'[A-Za-z0-9_.-]+', identifier) or identifier in {'.','..'}:
                    raise ValueError('Unsupported molecule ID')
                if identifier in failures:
                    raise ValueError(failures[identifier])
                candidates = sorted(index[identifier], key=lambda p: (not p.parent.name.isdigit(), int(p.parent.name) if p.parent.name.isdigit() else 0, str(p)))
                if not candidates:
                    raise ValueError('No indexed source pose file')
                path = candidates[0]
                selected, pose, energy, cnn, affinity = select_pose(path.read_text())
                if header:
                    parts = consensus[identifier].copy()
                    gn = header.index('GN')
                    for i in range(len(header)):
                        if i != gn:
                            old_path = Path(parts[i*3+2])
                            if not old_path.is_absolute():
                                old_path = args.consensus_input.resolve().parent/old_path
                            parts[i*3+2] = str(old_path.resolve())
                    parts[gn*3:gn*3+3] = ['1', str(energy), str(output.resolve())]
                    updates.append(' '.join(parts)+'\n')
                output.write_text(selected)
                scores.write(f'{energy};1;gnina;{pose};{identifier};/molecules/{output.name}\n')
                writer.writerow([identifier,str(path),len(candidates),pose+1,energy,cnn,affinity])
                summary['written'] += 1
                summary['selected_nonfirst_pose'] += pose != 0
                summary['duplicate_ids'] += len(candidates)>1
            except (OSError, ValueError, RuntimeError) as exc:
                output.unlink(missing_ok=True)
                err.writerow([identifier,str(exc)])
                summary['errors'] += 1
    if header:
        args.output_input.parent.mkdir(parents=True,exist_ok=True)
        args.output_input.write_text(','.join(header)+'\n'+''.join(updates))
    summary.update(selection='lowest finite minimizedAffinity; unreadable winning pose excluded without fallback',
                   reported_energy='selected pose minimizedAffinity',
                   duplicate_policy='lowest numeric historical source folder, then lexical path',
                   raw_base=str(base))
    (args.output_dir/'preparation_manifest.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__ == '__main__':
    main()
