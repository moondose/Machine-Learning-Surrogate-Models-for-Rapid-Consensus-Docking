#!/usr/bin/env python3
"""Build percentile-consensus targets for any selected set of docking engines.

Repeat --engine NAME=PATH to choose engines. Read source definitions and
configuration options with --help or in build_mean_binding_energy_target.py.
Preprocessed tables use score;rank;software;pose_index;molecule_id;pose_path.
Header CSVs support per-engine ID columns, score columns and delimiters.

Example (GNINA + LeDock + rDock):
    python build_percentile_consensus_target.py \
        --engine gnina=/path/to/gnina/csvs/Results_scoring.csv \
        --engine ledock=/path/to/ledock/csvs/Results_scoring.csv \
        --engine rdock=/path/to/rdock/csvs/Results_scoring.csv \
        --output-dir targets/percentile

Example (GNINA + a custom higher-is-better score in a header CSV):
    --engine gnina=gnina_scores.csv --engine custom=custom_scores.csv \
    --engine-format custom=csv --engine-id-column custom=compound_id \
    --engine-score-column custom=confidence --engine-direction custom=higher

Known defaults: GNINA, LeDock, rDock and DOCK3.7 prefer lower scores;
DiffDock prefers higher confidence. --engine-direction NAME=lower|higher
can override defaults. Legacy --gnina-csv, --ledock-csv and --dock37-csv
remain supported, including GNINA + LeDock + DOCK3.7 reproduction.

Use only molecules with finite scores from ALL selected engines. For the
complete intersection of N molecules, rank each engine in its preference
direction using average ranks for ties. p = (rank - 0.5) / N; lower p is better.
The primary target is the mean of all engine percentiles. All ranks are
computed BEFORE optional descriptor alignment. For N=1, p=0.5.

Outputs:
    targets.csv: source scores, engine percentiles and consensus targets.
    y_percentile_consensus.npy: mean percentile (primary target).
    y_percentile_median.npy: median percentile.
    y_percentile_rank_product.npy: geometric mean of percentiles.
    y_percentile_penalized.npy: mean + weight * population SD (default 0.5).
    target_ids.csv: array row IDs; excluded.csv: incomplete source-union IDs.
    summary.json: engine set, source formats, directions, formulas and counts.

--ids-csv FILE --id-column zinc_id optionally aligns arrays to descriptor
order; unmatched IDs receive NaN. Otherwise arrays follow targets.csv, sorted
by molecule ID. Requires Python 3.9+, NumPy, pandas and the adjacent mean-target
builder. Choose a new output directory.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from build_mean_binding_energy_target import (add_common_arguments, export_arrays,
                                              engine_scores)


def percentile_rank(values, ascending=True):
    """Use the existing project's average-tie mid-rank percentile definition."""
    return (values.rank(method='average', ascending=ascending) - 0.5) / len(values)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                    formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_arguments(parser)
    parser.add_argument('--disagreement-weight', type=float, default=0.5)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error('Choose a new output directory')
    if not math.isfinite(args.disagreement_weight) or args.disagreement_weight < 0:
        parser.error('Disagreement weight must be finite and nonnegative')
    try:
        joined, audit, specs = engine_scores(args)
    except (ValueError, OSError, pd.errors.ParserError) as exc:
        parser.error(str(exc))
    score_columns = [spec['output_score_column'] for spec in specs]
    percentile_columns = [spec['output_percentile_column'] for spec in specs]
    complete = joined.dropna(subset=score_columns).sort_values('molecule_id').copy()
    if complete.empty:
        parser.error('No molecules have finite scores from every requested engine')
    for spec in specs:
        complete[spec['output_percentile_column']] = percentile_rank(
            complete[spec['output_score_column']], ascending=spec['direction'] == 'lower')
    percentiles = complete[percentile_columns]
    complete['consensus_mean_percentile'] = percentiles.mean(axis=1)
    complete['consensus_median_percentile'] = percentiles.median(axis=1)
    complete['consensus_rank_product'] = np.exp(np.log(percentiles).mean(axis=1))
    complete['consensus_disagreement_sd'] = percentiles.std(axis=1, ddof=0)
    complete['consensus_penalized'] = (complete.consensus_mean_percentile
                         + args.disagreement_weight * complete.consensus_disagreement_sd)
    targets = {'consensus_mean_percentile': 'y_percentile_consensus.npy',
               'consensus_median_percentile': 'y_percentile_median.npy',
               'consensus_rank_product': 'y_percentile_rank_product.npy',
               'consensus_penalized': 'y_percentile_penalized.npy'}
    args.output_dir.mkdir(parents=True)
    complete.to_csv(args.output_dir / 'targets.csv', index=False)
    joined[joined[score_columns].isna().any(axis=1)].to_csv(args.output_dir / 'excluded.csv', index=False)
    alignment = export_arrays(complete, targets, args.output_dir, args.ids_csv, args.id_column)
    definitions = {
        'consensus_mean_percentile': 'arithmetic mean of selected-engine mid-rank percentiles',
        'consensus_median_percentile': 'median of selected-engine mid-rank percentiles',
        'consensus_rank_product': 'geometric mean of selected-engine mid-rank percentiles',
        'consensus_penalized': f'mean percentile + {args.disagreement_weight} * population percentile SD'}
    artifacts = {filename: {'column': column, 'definition': definitions[column],
                           'engines': [spec['name'] for spec in specs]}
                 for column, filename in targets.items()}
    summary = dict(inputs=audit, target_artifacts=artifacts, engines=[spec['name'] for spec in specs],
                   engine_definitions=specs, engine_count=len(specs),
                   score_columns=score_columns, source_union=len(joined),
                   matched_molecules=len(complete), excluded=len(joined)-len(complete),
                   ranking_population='complete requested-engine intersection',
                   percentile_formula='(average preference-direction rank - 0.5) / N',
                   primary_target='mean of engine percentiles',
                   disagreement_weight=args.disagreement_weight,
                   lower_is_better=True, alignment=alignment)
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
