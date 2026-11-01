#!/usr/bin/env python3
"""Shared model definitions, fold-local preprocessing and input alignment.

Model architectures, cached batching, scaler formulas and ranking metrics are
retained from the project's final MBE and percentile trainers. This module is
also the model-definition dependency for loading the exported checkpoints.
Dependencies: NumPy, pandas, PyTorch, SciPy, scikit-learn and Matplotlib.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, Dataset, BatchSampler, RandomSampler, SequentialSampler

ACTIVATIONS = {
    "relu": nn.ReLU,
    "leaky_relu": lambda: nn.LeakyReLU(negative_slope=0.01),
    "elu": nn.ELU, "gelu": nn.GELU, "silu": nn.SiLU, "tanh": nn.Tanh,
}

class DockingMLP(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dims=(2048, 1024, 512),
        dropout_rate=0.4,
        activation="gelu",
    ):
        super().__init__()
        if activation not in ACTIVATIONS:
            raise ValueError(f"Unsupported activation: {activation}")
        activation_factory = ACTIVATIONS[activation]
        layers = []
        curr_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(curr_dim, h_dim),
                    nn.BatchNorm1d(h_dim),
                    activation_factory(),
                    nn.Dropout(dropout_rate),
                ]
            )
            curr_dim = h_dim
        layers.append(nn.Linear(curr_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class CachedDescriptorDataset(Dataset):
    """Normalize once per fold, in bounded chunks, into a contiguous float32 cache."""
    def __init__(self, x_array, y_array, rows, x_mean, x_std, y_mean, y_std,
                 chunk_size):
        self.features = np.empty((len(rows), x_array.shape[1]), dtype=np.float32)
        for start in range(0, len(rows), chunk_size):
            stop = min(start + chunk_size, len(rows))
            batch = np.array(x_array[rows[start:stop]], dtype=np.float32, copy=True)
            np.nan_to_num(batch, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            batch -= x_mean
            batch /= x_std
            self.features[start:stop] = batch
        self.targets = np.asarray(
            (y_array[rows] - np.float32(y_mean)) / np.float32(y_std), dtype=np.float32
        )

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, indices):
        # The sampler supplies a whole batch: no Python work per individual row.
        return torch.from_numpy(self.features[indices]), torch.from_numpy(self.targets[indices])


class DeviceBatchLoader:
    """Keep normalized data on the GPU and gather batches there."""
    def __init__(self, dataset, batch_size, shuffle, drop_last, device):
        self.features = torch.from_numpy(dataset.features).to(device)
        self.targets = torch.from_numpy(dataset.targets).to(device)
        self.batch_size, self.shuffle, self.drop_last = batch_size, shuffle, drop_last

    def __iter__(self):
        count = len(self.targets)
        order = torch.randperm(count, device=self.targets.device) if self.shuffle else None
        stop = count - count % self.batch_size if self.drop_last else count
        for start in range(0, stop, self.batch_size):
            end = min(start + self.batch_size, stop)
            if order is None:
                yield self.features[start:end], self.targets[start:end]
            else:
                indices = order[start:end]
                yield self.features[indices], self.targets[indices]


def make_cached_loader(dataset, args, device, shuffle):
    drop_last = shuffle and len(dataset) % args.batch_size == 1
    if args.cache_device == "cuda":
        return DeviceBatchLoader(dataset, args.batch_size, shuffle, drop_last, device)
    sampler = RandomSampler(dataset) if shuffle else SequentialSampler(dataset)
    return DataLoader(
        dataset, batch_size=None,
        sampler=BatchSampler(sampler, args.batch_size, drop_last),
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    pearson = np.nan
    if len(y_true) > 1 and np.std(y_true) > 0 and np.std(y_pred) > 0:
        pearson = pearsonr(y_true, y_pred)[0]
    return {
        "RMSE": float(np.sqrt(mse)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
        "Pearson": float(pearson),
    }


def compute_feature_scaler(x_array, row_indices, chunk_size):
    row_indices = np.asarray(row_indices, dtype=np.int64)
    n_features = x_array.shape[1]
    total = 0
    sum_x = np.zeros(n_features, dtype=np.float64)
    sum_x2 = np.zeros(n_features, dtype=np.float64)

    for start in range(0, len(row_indices), chunk_size):
        batch_idx = row_indices[start : start + chunk_size]
        batch = np.asarray(x_array[batch_idx], dtype=np.float32)
        batch = np.nan_to_num(batch, nan=0.0, posinf=0.0, neginf=0.0)
        sum_x += batch.sum(axis=0, dtype=np.float64)
        sum_x2 += np.square(batch, dtype=np.float64).sum(axis=0, dtype=np.float64)
        total += batch.shape[0]

    mean = sum_x / total
    var = np.maximum(sum_x2 / total - np.square(mean), 0.0)
    std = np.sqrt(var)
    std[std < 1e-8] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)
def compute_scaler(x, rows, chunk_size):
    total = np.zeros(x.shape[1], np.float64)
    squared = np.zeros(x.shape[1], np.float64)
    count = 0
    for start in range(0, len(rows), chunk_size):
        block = np.asarray(x[rows[start:start + chunk_size]], dtype=np.float32)
        block = np.nan_to_num(block, nan=0.0, posinf=0.0, neginf=0.0)
        total += block.sum(axis=0, dtype=np.float64)
        squared += np.square(block, dtype=np.float64).sum(axis=0)
        count += len(block)
    mean = total / count
    variance = np.maximum(squared / count - mean * mean, 0.0)
    std = np.sqrt(variance)
    std[std < 1e-8] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def tail_metrics(y, prediction, fraction):
    n = len(y)
    k = max(1, int(np.ceil(n * fraction)))
    actual_order = np.argsort(y, kind="stable")
    predicted_order = np.argsort(prediction, kind="stable")
    actual_hits = np.zeros(n, dtype=np.float32)
    actual_hits[actual_order[:k]] = 1.0
    selected = predicted_order[:k]
    overlap = int(actual_hits[selected].sum())
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = float(np.sum(actual_hits[selected] * discounts))
    idcg = float(np.sum(discounts))
    recall = overlap / k
    return {
        f"recall_at_{fraction:g}": recall,
        f"ndcg_at_{fraction:g}": dcg / idcg,
        f"enrichment_at_{fraction:g}": recall / fraction,
        f"overlap_at_{fraction:g}": overlap,
    }


def all_metrics(y, prediction, fractions):
    result = {
        "RMSE": float(np.sqrt(mean_squared_error(y, prediction))),
        "MAE": float(mean_absolute_error(y, prediction)),
        "R2": float(r2_score(y, prediction)),
        "Pearson": float(pearsonr(y, prediction).statistic),
        "Spearman": float(spearmanr(y, prediction).statistic),
    }
    for fraction in fractions:
        result.update(tail_metrics(y, prediction, fraction))
    return result
class SearchConsensusMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout, activation):
        super().__init__()
        layers = []
        previous = input_dim
        activation_factory = ACTIVATIONS[activation]
        for width in hidden_dims:
            layers.extend(
                [nn.Linear(previous, width), nn.LayerNorm(width),
                 activation_factory(), nn.Dropout(dropout)]
            )
            previous = width
        layers.append(nn.Linear(previous, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze(1)


def make_optimizer(name, parameters, lr, weight_decay, momentum):
    if name == "adamw":
        return torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(
            parameters, lr=lr, weight_decay=weight_decay,
            momentum=momentum, nesterov=True,
        )
    if name == "rmsprop":
        return torch.optim.RMSprop(
            parameters, lr=lr, weight_decay=weight_decay, momentum=momentum
        )
    raise ValueError(f"Unknown optimizer: {name}")


def load_config(path):
    """Read portable model hyperparameters and validate their numeric domains."""
    config = json.loads(Path(path).read_text())
    required = {'hidden_dims', 'activation', 'optimizer', 'lr', 'dropout', 'weight_decay', 'batch_size'}
    if required - config.keys():
        raise ValueError(f'Missing config fields: {sorted(required - config.keys())}')
    widths = [int(value) for value in str(config['hidden_dims']).split(',')]
    if not widths or min(widths) < 1:
        raise ValueError('Hidden dimensions must be positive')
    if config['activation'] not in ACTIVATIONS or config['optimizer'] not in {'sgd','adam','adamw','rmsprop'}:
        raise ValueError('Unsupported activation or optimizer')
    for key in ['lr', 'dropout', 'weight_decay']:
        if not math.isfinite(float(config[key])):
            raise ValueError(f'Nonfinite {key}')
    if float(config['lr']) <= 0 or not 0 <= float(config['dropout']) < 1 or float(config['weight_decay']) < 0:
        raise ValueError('Invalid learning rate, dropout or weight decay')
    if int(config['batch_size']) < 1:
        raise ValueError('Batch size must be positive')
    return config


def read_ids(path, column):
    """Normalize molecule IDs while preserving row order and leading zeros."""
    frame = pd.read_csv(path, dtype='string')
    if column not in frame:
        raise ValueError(f'{path}: missing {column} column')
    ids = frame[column].str.strip().str.upper()
    if ids.isna().any() or ids.eq('').any() or ids.duplicated().any():
        raise ValueError(f'{path}: IDs must be nonempty and unique')
    return ids.to_numpy(str)


def load_training_inputs(descriptor_dir, target_path, target_ids=None,
                         target_id_column='molecule_id', x_file='X_features.npy',
                         target_key=None):
    """Align targets by molecule ID; feature values stay in their original order."""
    descriptor_dir, target_path = Path(descriptor_dir), Path(target_path)
    x = np.load(descriptor_dir / x_file, mmap_mode='r', allow_pickle=False)
    ids = read_ids(descriptor_dir / 'zinc_ids_aligned.csv', 'zinc_id')
    names = pd.read_csv(descriptor_dir / 'feature_names.csv', dtype='string')
    if 'feature_name' not in names or names.feature_name.isna().any() or names.feature_name.duplicated().any():
        raise ValueError('Feature names must be nonempty and unique')
    id_frame = pd.read_csv(descriptor_dir / 'zinc_ids_aligned.csv')
    if 'row_index' in id_frame and not np.array_equal(id_frame.row_index.to_numpy(), np.arange(len(ids))):
        raise ValueError('Descriptor row_index does not match matrix order')
    if x.ndim != 2 or x.shape != (len(ids), len(names)) or not np.issubdtype(x.dtype, np.number):
        raise ValueError('Descriptor matrix shape/type does not match IDs and feature names')
    if target_path.suffix.lower() == '.npz':
        with np.load(target_path, allow_pickle=False) as archive:
            raw_y = np.asarray(archive[target_key or 'consensus_mean_percentile'], dtype=np.float32)
            source_ids = pd.Series(archive['zinc_id'].astype(str), dtype='string').str.strip().str.upper()
            if source_ids.isna().any() or source_ids.eq('').any() or source_ids.duplicated().any():
                raise ValueError('Target archive IDs must be nonempty and unique')
            source_ids = source_ids.to_numpy(str)
    else:
        raw_y = np.asarray(np.load(target_path, allow_pickle=False), dtype=np.float32)
        target_ids = Path(target_ids) if target_ids else target_path.with_name('target_ids.csv')
        source_ids = read_ids(target_ids, target_id_column)
    if raw_y.ndim != 1 or len(raw_y) != len(source_ids):
        raise ValueError('Target must be a one-dimensional array aligned to its ID table')
    lookup = pd.Series(raw_y, index=source_ids)
    y = pd.Series(ids).map(lookup).to_numpy(dtype=np.float32, na_value=np.nan)
    audit = dict(descriptor_dir=str(descriptor_dir.resolve()), target_path=str(target_path.resolve()),
                 target_ids=str(Path(target_ids).resolve()) if target_ids else None,
                 descriptor_rows=len(ids), source_target_rows=len(source_ids),
                 descriptor_rows_with_finite_target=int(np.isfinite(y).sum()),
                 target_only_ids=int((~pd.Index(source_ids).isin(ids)).sum()),
                 preprocessing='nonfinite_features_to_zero_then_fold_training_standardization',
                 feature_schema_sha256=hashlib.sha256(('\n'.join(names.feature_name)+'\n').encode()).hexdigest())
    recipe_path = target_path.with_name('summary.json')
    if recipe_path.is_file():
        recipe = json.loads(recipe_path.read_text())
        artifact = recipe.get('target_artifacts', {}).get(target_path.name)
        if artifact:
            if target_key and artifact['column'] != target_key:
                raise ValueError('Selected target file does not match the requested target name')
            audit['target_definition'] = artifact['definition']
            audit['target_recipe'] = dict(artifact=artifact,
                engine_definitions=recipe.get('engine_definitions'),
                population=recipe.get('ranking_population', recipe.get('population')),
                matched_molecules=recipe.get('matched_molecules'))
    return x, y, ids, names, audit


def save_training_inputs(output, names, ids, alignment, config):
    """Save schema and row provenance beside ensemble weights."""
    output = Path(output)
    names.to_csv(output / 'feature_names.csv', index=False)
    pd.DataFrame({'row_index': np.arange(len(ids)), 'zinc_id': ids}).to_csv(
        output / 'input_ids.csv', index=False)
    (output / 'input_manifest.json').write_text(json.dumps(alignment, indent=2) + '\n')
    saved_config = dict(config)
    if 'target_definition' in alignment:
        saved_config['target_definition'] = alignment['target_definition']
        saved_config['target_recipe'] = alignment['target_recipe']
    (output / 'model_config.json').write_text(json.dumps(saved_config, indent=2) + '\n')
    (output / 'versions.json').write_text(json.dumps(dict(torch=torch.__version__,
        numpy=np.__version__, pandas=pd.__version__), indent=2) + '\n')
