#!/usr/bin/env python3
"""Build a configurable mean-score / mean binding energy (MBE) target.

Choose one or more engines by repeating --engine NAME=PATH. Defaults:
    gnina, ledock, rdock: six-column preprocessed score tables; lower is better.
    diffdock: preprocessed confidence table; higher is better.
    dock37: header CSV with zincid/dockscore; lower is better.
    Custom engine names: preprocessed table; lower is better.

Six-column preprocessed rows have no header:
    score;rank;software;pose_index;molecule_id;pose_path

For other header CSVs, specify --engine-format NAME=csv, then configure
--engine-id-column NAME=COLUMN, --engine-score-column NAME=COLUMN and
optionally --engine-sep NAME=DELIMITER. Use --engine-direction NAME=lower
or NAME=higher to set the preference direction explicitly.

The target is the arithmetic mean of direction-aligned raw scores: lower-is-
better scores are unchanged; higher-is-better scores are negated. For GNINA
minimizedAffinity and LeDock binding scores, this is the original MBE formula.
Column Mean_Binding_Energy(MBE) and y_mbe.npy retain their existing names.

Example (GNINA + LeDock + rDock):
    python build_mean_binding_energy_target.py \
        --engine gnina=/path/to/gnina/csvs/Results_scoring.csv \
        --engine ledock=/path/to/ledock/csvs/Results_scoring.csv \
        --engine rdock=/path/to/rdock/csvs/Results_scoring.csv \
        --output-dir targets/mean

Example (custom header CSV):
    --engine vina=/path/to/vina.csv --engine-format vina=csv \
    --engine-id-column vina=ligand_id --engine-score-column vina=affinity

Legacy --gnina-csv, --ledock-csv and --dock37-csv are supported and may be
combined with other named engines. --dock37-id-column and
--dock37-score-column configure the legacy DOCK3.7 input.

Molecules must have finite scores from ALL chosen engines. The engine set is
fixed for the run. IDs are stripped/uppercased; identical score duplicates
collapse and conflicting finite scores raise an error.

Outputs: targets.csv, y_mbe.npy, target_ids.csv, excluded.csv, summary.json.
The summary records the engine set, input schemas, signs and formula.
--ids-csv FILE --id-column zinc_id optionally aligns target arrays to descriptor
row order, with NaN for unavailable targets. Without it, array rows follow
targets.csv sorted by molecule ID. Choose a new output directory.
Requires Python 3.9+, NumPy and pandas. Keep this file beside the percentile
builder, which imports its shared engine-reading and alignment helpers.
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


def normalize_ids(values):
    """Use the same whitespace/case normalization as existing target construction."""
    ids = values.astype('string').str.strip().str.upper()
    return ids.mask(ids.eq(''))


def unique_scores(frame, id_column, score_column, name, source):
    """Merge identical duplicate scores and reject conflicting finite scores."""
    ids = normalize_ids(frame[id_column])
    if ids.isna().any():
        raise ValueError(f'{source}: missing molecule IDs')
    numeric = pd.to_numeric(frame[score_column], errors='coerce')
    numeric = numeric.where(np.isfinite(numeric))
    data = pd.DataFrame({'molecule_id': ids, name: numeric})
    conflicts = data.groupby('molecule_id')[name].nunique()
    conflicts = conflicts[conflicts > 1]
    if len(conflicts):
        raise ValueError(f'{source}: conflicting scores for IDs {conflicts.index[:10].tolist()}')
    result = data.groupby('molecule_id', as_index=False, sort=True)[name].first()
    audit = dict(path=str(Path(source).resolve()), input_rows=len(data),
                 unique_ids=len(result), duplicate_rows=int(data.molecule_id.duplicated().sum()),
                 missing_numeric_scores=int(result[name].isna().sum()))
    return result, audit


def read_preprocessed(path, name, engine=None):
    """Read the common six-column schema; retain the source software label."""
    columns = ['score', 'rank', 'software', 'pose_index', 'molecule_id', 'pose_path']
    data = pd.read_csv(path, sep=';', header=None, dtype='string')
    if data.shape[1] != 6:
        raise ValueError(f'{path}: expected six semicolon-delimited columns')
    data.columns = columns
    software = data.software.str.strip().str.lower()
    if software.isna().any() or software.eq('').any() or software.nunique() != 1:
        raise ValueError(f'{path}: preprocessed input must contain one software label')
    if engine is not None and engine in {'gnina', 'ledock', 'rdock', 'diffdock'}:
        if not software.eq(engine).all():
            raise ValueError(f'{path}: expected only {engine} rows')
    result, audit = unique_scores(data, 'molecule_id', 'score', name, path)
    audit['source_software'] = str(software.iloc[0])
    return result, audit


def paired_scores(gnina_path, ledock_path):
    """Retain the original two-engine helper for existing importers."""
    gnina, ga = read_preprocessed(gnina_path, 'GN_Binding_Energy', 'gnina')
    ledock, la = read_preprocessed(ledock_path, 'LeDock_Binding_Energy', 'ledock')
    return gnina.merge(ledock, on='molecule_id', how='outer', validate='one_to_one'), {'gnina': ga, 'ledock': la}


def engine_name(value):
    name = value.strip().lower()
    name = {'dock3.7': 'dock37', 'dock3_7': 'dock37', 'le_dock': 'ledock', 'r_dock': 'rdock'}.get(name, name)
    if not re.fullmatch(r'[a-z][a-z0-9_]*', name):
        raise ValueError(f'Invalid engine name: {value}')
    # These names would collide with consensus output column names.
    if name in {'consensus_mean', 'consensus_median'}:
        raise ValueError(f'Reserved engine name: {name}')
    return name


def assignments(items, option):
    """Parse repeated NAME=VALUE options while rejecting duplicate definitions."""
    result = {}
    for item in items or []:
        name, separator, value = item.partition('=')
        if not separator or not value:
            raise ValueError(f'{option} requires NAME=VALUE: {item}')
        name = engine_name(name)
        if name in result:
            raise ValueError(f'{option}: duplicate engine {name}')
        result[name] = value
    return result


def resolve_engines(args):
    """Resolve legacy aliases and configurable source formats into engine specs."""
    paths = assignments(args.engine, '--engine')
    for name in ['gnina', 'ledock', 'dock37']:
        value = getattr(args, name + '_csv')
        if value is not None:
            if name in paths:
                raise ValueError(f'{name}: choose --engine or --{name}-csv, not both')
            paths[name] = str(value)
    if not paths:
        raise ValueError('Select at least one engine with --engine NAME=PATH or a legacy CSV option')
    options = {}
    for key in ['format', 'id_column', 'score_column', 'sep', 'direction']:
        options[key] = assignments(getattr(args, 'engine_' + key), '--engine-' + key.replace('_', '-'))
        unknown = set(options[key]) - set(paths)
        if unknown:
            raise ValueError(f'Engine option supplied without input: {sorted(unknown)}')
    score_names = {'gnina': 'GN_Binding_Energy', 'ledock': 'LeDock_Binding_Energy',
                   'dock37': 'DOCK37_Binding_Energy'}
    specs = []
    for name, path in paths.items():
        fmt = options['format'].get(name, 'csv' if name == 'dock37' else 'preprocessed')
        direction = options['direction'].get(name, 'higher' if name == 'diffdock' else 'lower')
        if fmt not in {'csv', 'preprocessed'} or direction not in {'lower', 'higher'}:
            raise ValueError(f'{name}: format must be csv/preprocessed and direction lower/higher')
        if fmt == 'preprocessed' and any(name in options[key] for key in ['id_column','score_column','sep']):
            raise ValueError(f'{name}: column/delimiter options require --engine-format {name}=csv')
        default_id = args.dock37_id_column if name == 'dock37' else 'molecule_id'
        default_score = args.dock37_score_column if name == 'dock37' else 'score'
        sep = options['sep'].get(name, ',')
        if sep == r'\t':
            sep = '\t'
        if len(sep) != 1:
            raise ValueError(f'{name}: delimiter must be one character')
        specs.append(dict(name=name, path=str(Path(path).resolve()), format=fmt,
                          id_column=options['id_column'].get(name, default_id) if fmt == 'csv' else 'molecule_id',
                          score_column=options['score_column'].get(name, default_score) if fmt == 'csv' else 'score',
                          delimiter=sep if fmt == 'csv' else ';', direction=direction,
                          mean_multiplier=1 if direction == 'lower' else -1,
                          output_score_column=score_names.get(name, name + '_score'),
                          output_percentile_column=name + '_percentile'))
    return specs


def engine_scores(args):
    """Outer-join source scores by ID; callers select the full finite intersection."""
    specs = resolve_engines(args)
    joined, audit = None, {}
    for spec in specs:
        path, column = spec['path'], spec['output_score_column']
        if spec['format'] == 'preprocessed':
            data, details = read_preprocessed(path, column, spec['name'])
        else:
            raw = pd.read_csv(path, sep=spec['delimiter'], dtype='string')
            missing = {spec['id_column'], spec['score_column']} - set(raw.columns)
            if missing:
                raise ValueError(f'{path}: missing columns {sorted(missing)}')
            data, details = unique_scores(raw, spec['id_column'], spec['score_column'], column, path)
        joined = data if joined is None else joined.merge(data, on='molecule_id', how='outer', validate='one_to_one')
        audit[spec['name']] = dict(details, **{key: value for key, value in spec.items() if key != 'path'})
    return joined, audit, specs


def export_arrays(complete, target_columns, output, ids_csv=None, id_column='zinc_id'):
    """Write explicitly paired IDs and arrays; preserve optional descriptor order."""
    if ids_csv is not None:
        frame = pd.read_csv(ids_csv, dtype='string')
        if id_column not in frame:
            raise ValueError(f'Missing ID column: {id_column}')
        ids = normalize_ids(frame[id_column])
    else:
        ids = complete.molecule_id.reset_index(drop=True)
    if ids.isna().any() or ids.duplicated().any():
        raise ValueError('Array alignment IDs must be nonempty and unique')
    lookup = complete.set_index('molecule_id')
    arrays = {filename: ids.map(lookup[column]).to_numpy(dtype=np.float32, na_value=np.nan)
              for column, filename in target_columns.items()}
    pd.DataFrame({'molecule_id': ids}).to_csv(output / 'target_ids.csv', index=False)
    for filename, values in arrays.items():
        np.save(output / filename, values)
    first = next(iter(arrays.values()))
    return dict(array_rows=len(ids), array_rows_with_target=int(np.isfinite(first).sum()),
                order='ids_csv row order' if ids_csv else 'targets.csv row order',
                ids_csv=str(Path(ids_csv).resolve()) if ids_csv else None)


def add_common_arguments(parser):
    parser.add_argument('--engine', action='append', default=[], metavar='NAME=PATH',
                        help='Repeat for each desired engine; see module examples')
    parser.add_argument('--engine-format', action='append', default=[], metavar='NAME=FORMAT')
    parser.add_argument('--engine-id-column', action='append', default=[], metavar='NAME=COLUMN')
    parser.add_argument('--engine-score-column', action='append', default=[], metavar='NAME=COLUMN')
    parser.add_argument('--engine-sep', action='append', default=[], metavar='NAME=DELIMITER')
    parser.add_argument('--engine-direction', action='append', default=[], metavar='NAME=lower|higher')
    parser.add_argument('--gnina-csv', type=Path)
    parser.add_argument('--ledock-csv', type=Path)
    parser.add_argument('--dock37-csv', type=Path)
    parser.add_argument('--dock37-id-column', default='zincid')
    parser.add_argument('--dock37-score-column', default='dockscore')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--ids-csv', type=Path, help='Optional descriptor-aligned ID CSV')
    parser.add_argument('--id-column', default='zinc_id')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                    formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_arguments(parser)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error('Choose a new output directory')
    try:
        joined, audit, specs = engine_scores(args)
    except (ValueError, OSError, pd.errors.ParserError) as exc:
        parser.error(str(exc))
    columns = [spec['output_score_column'] for spec in specs]
    complete = joined.dropna(subset=columns).sort_values('molecule_id').copy()
    if complete.empty:
        parser.error('No molecules have finite scores from all selected engines')
    signs = np.asarray([spec['mean_multiplier'] for spec in specs], dtype=np.float64)
    # Raw score columns remain unchanged; only the target uses preference signs.
    complete['Mean_Binding_Energy(MBE)'] = (complete[columns] * signs).mean(axis=1)
    args.output_dir.mkdir(parents=True)
    complete.to_csv(args.output_dir / 'targets.csv', index=False)
    joined[joined[columns].isna().any(axis=1)].to_csv(args.output_dir / 'excluded.csv', index=False)
    alignment = export_arrays(complete, {'Mean_Binding_Energy(MBE)': 'y_mbe.npy'},
                             args.output_dir, args.ids_csv, args.id_column)
    formula = 'mean(' + ', '.join(('' if spec['mean_multiplier'] == 1 else '-') + spec['output_score_column'] for spec in specs) + ')'
    summary = dict(formula=formula, engines=[spec['name'] for spec in specs],
                   engine_definitions=specs, engine_count=len(specs),
                   target_definition='Arithmetic mean of direction-aligned raw engine scores',
                   population='complete selected-engine intersection',
                   target_artifacts={'y_mbe.npy': {'column': 'Mean_Binding_Energy(MBE)',
                       'definition': formula, 'engines': [spec['name'] for spec in specs]}},
                   lower_is_better=True, inputs=audit, source_union=len(joined),
                   matched_molecules=len(complete), excluded=len(joined)-len(complete),
                   alignment=alignment)
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
