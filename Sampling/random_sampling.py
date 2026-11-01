#!/usr/bin/env python3
"""Sample CSV rows uniformly without replacement using reservoir sampling.

Example:
    python random_sampling.py --input dataset.csv.gz --output random.csv \
        --n-select 500000 --seed 42

CSV, gzip, bzip2 and xz inputs are supported. The header and original column
values are preserved. Sampling treats each data row as one population member.
Memory usage is proportional to the requested sample size.
"""
import argparse
import bz2
import csv
import gzip
import lzma
from pathlib import Path
import random


def reservoir_sampling(input_path, output_path, n_select, seed=42, sep=',',
                       encoding='utf-8-sig'):
    """Apply Algorithm R in one streaming pass and return input/sample counts."""
    input_path, output_path = Path(input_path), Path(output_path)
    if n_select < 1:
        raise ValueError('n_select must be positive')
    if len(sep) != 1:
        raise ValueError('The input delimiter must be a single character')
    if input_path.resolve() == output_path.resolve():
        raise ValueError('Input and output must be different files')
    if output_path.exists():
        raise FileExistsError(f'Output already exists: {output_path}')
    opener = {'.gz': gzip.open, '.bz2': bz2.open, '.xz': lzma.open}.get(
        input_path.suffix.lower(), open)
    rng = random.Random(seed)
    reservoir = []
    total = 0
    with opener(input_path, 'rt', newline='', encoding=encoding) as handle:
        reader = csv.reader(handle, delimiter=sep)
        header = next(reader, None)
        if not header:
            raise ValueError('Input must contain a header row')
        for row in reader:
            if not row:
                continue
            if len(row) != len(header):
                raise ValueError(f'Unexpected column count at CSV line {reader.line_num}')
            # Each of the first i+1 rows has probability k/(i+1) of retention.
            i = total
            total += 1
            if i < n_select:
                reservoir.append(row)
            else:
                slot = rng.randrange(i + 1)
                if slot < n_select:
                    reservoir[slot] = row
    if total < n_select:
        raise ValueError(f'Requested {n_select} rows, but the dataset contains {total}')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('x', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(reservoir)
    return total, len(reservoir)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                         formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='Output CSV path')
    parser.add_argument('--n-select', type=int, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--sep', default=',', help='Input delimiter; output uses commas')
    parser.add_argument('--encoding', default='utf-8-sig', help='Input text encoding')
    args = parser.parse_args()
    try:
        total, selected = reservoir_sampling(args.input, args.output, args.n_select,
                                            args.seed, args.sep, args.encoding)
    except (OSError, ValueError, csv.Error) as exc:
        parser.exit(1, f'Error: {exc}\n')
    print(f'Sampled {selected} of {total} rows (seed={args.seed}): {args.output}')


if __name__ == '__main__':
    main()
