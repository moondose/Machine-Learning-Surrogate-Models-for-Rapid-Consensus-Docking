#!/usr/bin/env python3
"""Train the final MBE descriptor MLP with fold-local scalers and MSE selection.

Uses the project's BatchNorm MLP and original final-fold training loop: scaled
MSE loss, ReduceLROnPlateau on validation MSE and early stopping on validation
MSE. Default hyperparameters come from configs/mbe.json (batch 1024, SGD,
LR 0.01, SiLU, hidden layers 2048/1024/512, 10 folds, seed 42).

Example:
    python train_mbe.py --descriptor-dir /path/to/descriptors \
        --targets /path/to/targets/mbe/y_mbe.npy --output-dir results/mbe

Target IDs default to target_ids.csv beside the target NPY. Molecule IDs align
targets to zinc_ids_aligned.csv; unmatched/nonfinite targets are excluded.
Use --target-ids and --target-id-column for a different alignment CSV.
Input descriptors are X_features.npy with the adjacent feature_names.csv.

Outputs: fold_N/best_model.pt and metadata.npz (exact scalers, architecture,
train/validation source rows); fold histories/plots; fold_assignments.csv;
out_of_fold_predictions.csv; feature_names.csv; input/config manifests; CV
metrics and summary. Checkpoint weights use the model_utils.DockingMLP class.

Dependencies: Python 3.9+, torch >=2.3, NumPy, pandas, SciPy, scikit-learn,
Matplotlib. Keep model_utils.py and configs/ alongside this script.
Use a new output directory. --cpu selects CPU; --precision fp32 disables AMP.
Normalized fold caches occupy approximately 4*N*(features+1) bytes in RAM;
--cache-device cuda also stores the cache on the GPU.
"""
import argparse
import json
import os
from pathlib import Path
import time
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from model_utils import (DockingMLP, make_cached_loader, CachedDescriptorDataset,
                         DeviceBatchLoader, set_seed, compute_metrics,
                         compute_feature_scaler, load_training_inputs,
                         save_training_inputs, load_config)

def save_fold_loss_plot(history_df, fold_dir, fold_num):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history_df["epoch"], history_df["train_loss"], label="Train loss", linewidth=1.8)
    ax.plot(history_df["epoch"], history_df["val_loss"], label="Validation loss", linewidth=1.8)
    best_idx = int(history_df["val_loss"].idxmin())
    best_epoch = int(history_df.loc[best_idx, "epoch"])
    best_val = float(history_df.loc[best_idx, "val_loss"])
    ax.scatter([best_epoch], [best_val], color="tab:red", s=28, zorder=3, label="Best validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss (scaled target)")
    ax.set_title(f"Fold {fold_num} Training Curve")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(fold_dir, "loss_curve.png"), dpi=200)
    plt.close(fig)


def save_oof_scatter(oof_actual, oof_pred, oof_metrics, output_dir, target_name):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(oof_actual, oof_pred, alpha=0.25, s=1)
    min_v = min(float(oof_actual.min()), float(oof_pred.min()))
    max_v = max(float(oof_actual.max()), float(oof_pred.max()))
    ax.plot([min_v, max_v], [min_v, max_v], "r--", linewidth=1)
    ax.set_xlabel(f"Actual {target_name}")
    ax.set_ylabel(f"Predicted {target_name}")
    ax.set_title(f"{target_name} OOF Prediction (R2={oof_metrics['R2']:.3f})")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "out_of_fold_scatter.png"), dpi=200)
    plt.close(fig)


def save_residual_histogram(oof_actual, oof_pred, output_dir, target_name):
    residuals = oof_pred - oof_actual
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(residuals, bins=80, color="tab:blue", alpha=0.8)
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel(f"Prediction error ({target_name})")
    ax.set_ylabel("Count")
    ax.set_title("Out-of-Fold Residual Distribution")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "out_of_fold_residuals.png"), dpi=200)
    plt.close(fig)


def save_residual_scatter(oof_actual, oof_pred, output_dir, target_name):
    residuals = oof_pred - oof_actual
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(oof_pred, residuals, alpha=0.25, s=1)
    ax.axhline(0.0, color="tab:red", linestyle="--", linewidth=1)
    ax.set_xlabel(f"Predicted {target_name}")
    ax.set_ylabel(f"Residual (predicted - actual {target_name})")
    ax.set_title("Out-of-Fold Residual Plot")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "out_of_fold_residual_scatter.png"), dpi=200)
    plt.close(fig)


def save_metric_summary_plots(fold_df, output_dir):
    metric_order = ["R2", "Pearson", "RMSE", "MAE"]
    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    axes = axes.ravel()
    for ax, metric in zip(axes, metric_order):
        ax.bar(fold_df["fold"].astype(str), fold_df[metric], color="tab:blue", alpha=0.85)
        ax.axhline(float(fold_df[metric].mean()), color="tab:red", linestyle="--", linewidth=1)
        ax.set_title(metric)
        ax.set_xlabel("Fold")
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Cross-Validation Metrics by Fold")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fold_metrics_summary.png"), dpi=200)
    plt.close(fig)


def save_combined_loss_plot(output_dir):
    histories = []
    for fold_dir_name in sorted(os.listdir(output_dir)):
        if not fold_dir_name.startswith("fold_"):
            continue
        history_path = os.path.join(output_dir, fold_dir_name, "history.csv")
        if not os.path.exists(history_path):
            continue
        history_df = pd.read_csv(history_path)
        history_df["fold"] = int(fold_dir_name.split("_", 1)[1])
        histories.append(history_df)

    if not histories:
        return

    all_history = pd.concat(histories, ignore_index=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    for fold, fold_history in all_history.groupby("fold"):
        ax.plot(
            fold_history["epoch"],
            fold_history["val_loss"],
            linewidth=1,
            alpha=0.55,
            label=f"Fold {fold}",
        )
    mean_history = all_history.groupby("epoch", as_index=False)["val_loss"].mean()
    ax.plot(mean_history["epoch"], mean_history["val_loss"], color="black", linewidth=2.2, label="Mean")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation MSE loss (scaled target)")
    ax.set_title("Validation Loss Across Folds")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "validation_loss_by_fold.png"), dpi=200)
    plt.close(fig)


def train_one_fold(args, fold_num, x_array, y_array, train_rows, val_rows, device,
                   epoch_observer=None):
    fold_started = time.monotonic()
    print(f"\n{'=' * 22} Fold {fold_num}/{args.n_folds} {'=' * 22}")
    print(f"Train rows: {len(train_rows)} | Validation rows: {len(val_rows)}")

    print("Computing fold feature scaler...")
    x_mean, x_std = compute_feature_scaler(x_array, train_rows, args.scaler_chunk_size)
    y_mean = float(y_array[train_rows].mean())
    y_std = float(y_array[train_rows].std() + 1e-8)

    cache_started = time.monotonic()
    cache_gib = (len(train_rows) + len(val_rows)) * (x_array.shape[1] + 1) * 4 / 2**30
    print(f"Preparing normalized fold cache: {cache_gib:.2f} GiB on RAM"
          + (" and GPU" if args.cache_device == "cuda" else ""))
    train_ds = CachedDescriptorDataset(x_array, y_array, train_rows, x_mean, x_std,
                                       y_mean, y_std, args.scaler_chunk_size)
    val_ds = CachedDescriptorDataset(x_array, y_array, val_rows, x_mean, x_std,
                                     y_mean, y_std, args.scaler_chunk_size)
    train_loader = make_cached_loader(train_ds, args, device, True)
    val_loader = make_cached_loader(val_ds, args, device, False)
    cache_seconds = time.monotonic() - cache_started

    def timestamp():
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        return time.monotonic()

    if len(train_rows) < 2:
        raise ValueError("Each training fold needs at least two rows for BatchNorm")
    hidden_dims = tuple(int(x) for x in args.hidden_dims.split(",") if x.strip())
    model = DockingMLP(
        x_array.shape[1],
        hidden_dims=hidden_dims,
        dropout_rate=args.dropout,
        activation=args.activation,
    ).to(device)
    if args.optimizer == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            momentum=0.9,
            nesterov=True,
        )
    elif args.optimizer == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "rmsprop":
        optimizer = optim.RMSprop(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9
        )
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")
    criterion = nn.MSELoss()
    amp_enabled = device.type == "cuda" and not args.no_amp
    grad_scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=args.lr_patience, factor=0.5
    )

    fold_dir = os.path.join(args.output_dir, f"fold_{fold_num}")
    os.makedirs(fold_dir, exist_ok=True)
    model_path = os.path.join(fold_dir, "best_model.pt")
    metadata_path = os.path.join(fold_dir, "metadata.npz")
    history = []
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        epoch_started = timestamp()
        model.train()
        train_loss_sum = torch.zeros((), device=device)
        train_item_count = 0
        for bx, by in train_loader:
            bx = bx.to(device, non_blocking=True)
            by = by.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=amp_enabled
            ):
                pred = model(bx).squeeze(-1)
                loss = criterion(pred, by)
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()
            train_loss_sum += loss.detach() * by.numel()
            train_item_count += by.numel()

        training_seconds = timestamp() - epoch_started
        validation_started = timestamp()
        model.eval()
        val_loss_sum = torch.zeros((), device=device)
        val_item_count = 0
        with torch.no_grad():
            for bx, by in val_loader:
                bx = bx.to(device, non_blocking=True)
                by = by.to(device, non_blocking=True)
                with torch.autocast(
                    device_type=device.type, dtype=torch.float16, enabled=amp_enabled
                ):
                    pred = model(bx).squeeze(-1)
                    loss = criterion(pred, by)
                val_loss_sum += loss.detach() * by.numel()
                val_item_count += by.numel()

        train_loss = (train_loss_sum / train_item_count).item()
        val_loss = (val_loss_sum / val_item_count).item()
        validation_seconds = timestamp() - validation_started
        scheduler.step(val_loss)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                        "training_seconds": training_seconds,
                        "validation_seconds": validation_seconds})

        if epoch_observer is not None:
            epoch_observer(epoch, model, train_ds, device)

        if epoch == 1 or epoch % args.print_every == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"Fold {fold_num} | Epoch {epoch:03d} | "
                f"Train {train_loss:.6f} | Val {val_loss:.6f} | LR {lr:.2e} | "
                f"Train {training_seconds:.2f}s | Val {validation_seconds:.2f}s"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path)
            np.savez(
                metadata_path,
                x_mean=x_mean,
                x_std=x_std,
                y_mean=np.array([y_mean], dtype=np.float32),
                y_std=np.array([y_std], dtype=np.float32),
                input_dim=np.array([x_array.shape[1]], dtype=np.int64),
                target_file=np.array([args.target_file]),
                target_name=np.array([args.target_name]),
                hidden_dims=np.asarray(hidden_dims, dtype=np.int64),
                activation=np.array([args.activation]),
                optimizer=np.array([args.optimizer]),
                lr=np.array([args.lr], dtype=np.float64),
                dropout=np.array([args.dropout], dtype=np.float64),
                weight_decay=np.array([args.weight_decay], dtype=np.float64),
                batch_size=np.array([args.batch_size], dtype=np.int64),
            )
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.early_stop_patience:
                print(f"Early stopping at epoch {epoch}.")
                break

    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(fold_dir, "history.csv"), index=False)
    save_fold_loss_plot(history_df, fold_dir, fold_num)

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    preds = []
    actuals = []
    with torch.no_grad():
        for bx, by in val_loader:
            bx = bx.to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=amp_enabled
            ):
                pred_norm = model(bx).squeeze(-1).float().cpu().numpy()
            pred = pred_norm * y_std + y_mean
            true = by.cpu().numpy() * y_std + y_mean
            preds.append(pred)
            actuals.append(true)

    preds = np.concatenate(preds).astype(np.float32)
    actuals = np.concatenate(actuals).astype(np.float32)
    metrics = compute_metrics(actuals, preds)
    metrics["cache_seconds"] = cache_seconds
    metrics["training_seconds"] = sum(row["training_seconds"] for row in history)
    metrics["validation_seconds"] = sum(row["validation_seconds"] for row in history)
    metrics["runtime_seconds"] = time.monotonic() - fold_started
    print(
        f"Fold {fold_num} Results | "
        f"R2={metrics['R2']:.4f} RMSE={metrics['RMSE']:.4f} "
        f"MAE={metrics['MAE']:.4f} Pearson={metrics['Pearson']:.4f} "
        f"Runtime={metrics['runtime_seconds']:.1f}s"
    )
    return metrics, preds, actuals


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--descriptor-dir', type=Path, required=True)
    parser.add_argument('--targets', type=Path, required=True)
    parser.add_argument('--target-ids', type=Path)
    parser.add_argument('--target-id-column', default='molecule_id')
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('configs') / 'mbe.json')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--x-file', default='X_features.npy')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--folds', type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--hidden-dims', help='Optional comma-separated hidden widths')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--scaler-chunk-size', type=int, default=20000)
    parser.add_argument('--cache-device', choices=['cpu','cuda'], default='cpu')
    parser.add_argument('--precision', choices=['auto','fp32','amp'], default='auto')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--max-rows', type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    for name in ['epochs','batch_size','hidden_dims','seed']:
        if getattr(args, name) is None:
            setattr(args, name, config[name])
    args.n_folds = args.folds if args.folds is not None else config['n_folds']
    args.hidden_dims = str(args.hidden_dims)
    args.lr = float(config['lr'])
    args.weight_decay = float(config['weight_decay'])
    args.dropout = float(config['dropout'])
    args.activation = config['activation']
    args.optimizer = config['optimizer']
    args.early_stop_patience = int(config['early_stop_patience'])
    args.lr_patience = int(config['lr_patience'])
    args.print_every = 5
    args.target_file = args.targets.name
    args.target_name = 'Mean_Binding_Energy(MBE)'
    if min(args.threads, args.epochs, args.scaler_chunk_size) < 1 or args.num_workers < 0 or args.batch_size < 2 or args.n_folds < 2:
        raise ValueError('Require positive controls, batch size >=2 and folds >=2')
    if args.max_rows is not None and args.max_rows < 1:
        raise ValueError('--max-rows must be positive')
    if args.early_stop_patience < 1 or args.lr_patience < 0:
        raise ValueError('Invalid patience settings')
    if Path(args.output_dir).exists():
        raise FileExistsError('Choose a new output directory')
    torch.set_num_threads(args.threads)
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    if args.cache_device == 'cuda' and device.type != 'cuda':
        raise ValueError('CUDA cache requires a CUDA device')
    if args.precision == 'auto':
        args.precision = config.get('precision','fp32') if device.type == 'cuda' else 'fp32'
    if args.precision == 'amp' and device.type != 'cuda':
        raise ValueError('AMP requires a CUDA device')
    args.no_amp = args.precision == 'fp32'
    widths = [int(value) for value in args.hidden_dims.split(',')]
    if not widths or min(widths) < 1:
        raise ValueError('Hidden dimensions must be positive')
    config.update(hidden_dims=args.hidden_dims, epochs=args.epochs, batch_size=args.batch_size,
                  seed=args.seed, n_folds=args.n_folds, precision=args.precision)
    x, y, ids, names, alignment = load_training_inputs(args.descriptor_dir, args.targets,
              args.target_ids, args.target_id_column, args.x_file, 'Mean_Binding_Energy(MBE)')
    if int(config.get('input_dim',x.shape[1])) != x.shape[1]:
        raise ValueError('Config input_dim does not match descriptors')
    valid = np.flatnonzero(np.isfinite(y))
    if args.max_rows is not None:
        valid = valid[:args.max_rows]
    if len(valid) < max(4, 2 * args.n_folds):
        raise ValueError('Require at least two validation rows and two training rows per fold')
    Path(args.output_dir).mkdir(parents=True)
    save_training_inputs(args.output_dir, names, ids, alignment, config)
    # Convert paths to strings so saved run configuration remains portable JSON.
    for key, value in vars(args).copy().items():
        if isinstance(value, Path):
            setattr(args, key, str(value))
    (Path(args.output_dir) / 'training_config.json').write_text(json.dumps(vars(args), indent=2)+'\n')
    kfold = KFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    predictions = np.full(len(valid), np.nan, np.float32)
    actual = np.full(len(valid), np.nan, np.float32)
    fold_ids = np.zeros(len(valid), np.int16)
    fold_rows = []
    assignments = []
    started = time.monotonic()
    for fold, (train_pos, val_pos) in enumerate(kfold.split(valid), 1):
        train_rows, val_rows = valid[train_pos], valid[val_pos]
        metrics, pred, true = train_one_fold(args, fold, x, y, train_rows, val_rows, device)
        predictions[val_pos], actual[val_pos], fold_ids[val_pos] = pred, true, fold
        fold_rows.append(dict(fold=fold, **metrics))
        assignments.extend((int(row), ids[row], fold) for row in val_rows)
        meta_path = Path(args.output_dir) / f'fold_{fold}/metadata.npz'
        with np.load(meta_path, allow_pickle=False) as archive:
            metadata = {key: archive[key] for key in archive.files}
        metadata.update(train_rows=train_rows, validation_rows=val_rows,
                        fold=np.asarray(fold), folds=np.asarray(args.n_folds),
                        seed=np.asarray(args.seed), precision=np.asarray(args.precision),
                        selection_metric=np.asarray('validation_mse'))
        np.savez_compressed(meta_path, **metadata)
    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(Path(args.output_dir) / 'fold_metrics.csv', index=False)
    pd.DataFrame(dict(zinc_id=ids[valid], fold=fold_ids, actual_target=actual,
                      predicted_target=predictions)).to_csv(Path(args.output_dir) / 'out_of_fold_predictions.csv', index=False)
    pd.DataFrame(assignments, columns=['row_index','zinc_id','validation_fold']).sort_values('row_index').to_csv(
        Path(args.output_dir) / 'fold_assignments.csv', index=False)
    summary = dict(vars(args))
    summary.update(OOF=compute_metrics(actual,predictions),
                   selection_metric='validation_mse', valid_rows=len(valid),
                   elapsed_seconds=time.monotonic()-started)
    (Path(args.output_dir) / 'cv_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    oof_metrics = summary['OOF']
    save_oof_scatter(actual,predictions,oof_metrics,args.output_dir,args.target_name)
    save_residual_histogram(actual,predictions,args.output_dir,args.target_name)
    save_residual_scatter(actual,predictions,args.output_dir,args.target_name)
    save_metric_summary_plots(fold_df,args.output_dir)
    save_combined_loss_plot(args.output_dir)
    print(json.dumps(oof_metrics,indent=2))


if __name__ == '__main__':
    main()
