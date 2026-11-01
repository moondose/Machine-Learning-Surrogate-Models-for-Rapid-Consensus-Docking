#!/usr/bin/env python3
"""Train the final percentile-consensus LayerNorm MLP.

Retains the original model, Smooth L1 loss, gradient clipping and validation
binary top-tail NDCG checkpoint selection. ReduceLROnPlateau and early stopping
follow validation NDCG. Defaults come from configs/percentile_consensus.json:
1024/512/256/128 ReLU layers, AdamW LR 0.0001, batch 1024, 10 folds, seed 42.

Example:
    python train_percentile_consensus.py --descriptor-dir /path/to/descriptors \
        --targets /path/to/targets/percentile/y_percentile_consensus.npy \
        --output-dir results/percentile

Input NPY targets use target_ids.csv beside the array, or --target-ids. Legacy
NPZ targets with zinc_id and a --target key are also supported. Targets are
aligned by molecule ID to descriptor rows; unavailable targets are excluded.
No percentile ranks are recomputed by the trainer. --config selects model
hyperparameters; explicit CLI training controls override configuration values.
CPU uses FP32 automatically; CUDA auto precision follows the configuration.

Outputs: fold_N_best_model.pt, fold_N_metadata.npz with exact fold scalers and
architecture, feature_names.csv, fold_assignments.csv, aligned IDs, manifests,
training histories, learning curves, OOF predictions and CV/ranking metrics.
Checkpoint weights use model_utils.SearchConsensusMLP.

Requires Python 3.9+, torch >=2.3, NumPy, pandas, SciPy, scikit-learn and
Matplotlib. Keep model_utils.py and configs/ alongside this script. A new
output directory is required. RAM cache is approximately 4*N*(features+1)
bytes; --cache-device cuda additionally retains the cache on the GPU.
"""
from __future__ import annotations
import argparse
import json
import time
import resource
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib
matplotlib.use("Agg")
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, BatchSampler, RandomSampler, SequentialSampler
from model_utils import (all_metrics, compute_scaler, set_seed, tail_metrics,
                         SearchConsensusMLP, make_optimizer, CachedDescriptorDataset,
                         DeviceBatchLoader, load_training_inputs, save_training_inputs, load_config)

def make_loader(dataset, args, device, batch_size, shuffle):
    # Percentile model uses LayerNorm: retain singleton final batches.
    if args.cache_device == "cuda":
        return DeviceBatchLoader(dataset, batch_size, shuffle, False, device)
    sampler = RandomSampler(dataset) if shuffle else SequentialSampler(dataset)
    return DataLoader(dataset, batch_size=None,
                      sampler=BatchSampler(sampler, batch_size, False),
                      num_workers=args.num_workers, pin_memory=device.type == "cuda",
                      persistent_workers=args.num_workers > 0)


def predict(model, loader, device, y_mean, y_std, amp_enabled=False):
    model.eval()
    values = []
    with torch.no_grad():
        for features, _ in loader:
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                prediction = model(features.to(device, non_blocking=True))
            values.append(prediction.float())
    # One GPU-to-CPU transfer per validation pass, rather than per batch.
    return torch.cat(values).cpu().numpy() * y_std + y_mean


def save_history_plot(history, path, fold):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history.epoch, history.train_loss)
    axes[0].set_ylabel("Smooth L1 training loss (scaled target)")
    axes[1].plot(history.epoch, history.val_ndcg)
    best = history.loc[history.val_ndcg.idxmax()]
    axes[1].scatter(best.epoch, best.val_ndcg, color="red", label="Best checkpoint")
    axes[1].set_ylabel("Validation top-tail NDCG (higher is better)")
    axes[1].legend()
    for ax in axes:
        ax.set_xlabel("Epoch"); ax.grid(alpha=.25)
    fig.suptitle(f"Percentile fold {fold}"); fig.tight_layout()
    fig.savefig(path, dpi=180); plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--descriptor-dir", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--target-ids", type=Path)
    parser.add_argument("--target-id-column", default="molecule_id")
    parser.add_argument("--config", "--grid-config", dest="grid_config", type=Path,
                        default=Path(__file__).with_name("configs") / "percentile_consensus.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target", default="consensus_mean_percentile")
    parser.add_argument("--x-file", default="X_features.npy")
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr-patience", type=int, default=4)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--tail-fraction", type=float, default=0.01)
    parser.add_argument("--report-fractions", type=float, nargs="+", default=[0.001,0.005,0.01,0.05])
    parser.add_argument("--smooth-l1-beta", type=float, default=0.25)
    parser.add_argument("--clip-grad-norm", type=float, default=5.0)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--scaler-chunk-size", type=int, default=20000)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--cache-device", choices=("cpu","cuda"), default="cpu")
    parser.add_argument("--precision", choices=("auto","fp32","amp"), default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.grid_config)
    # New search outputs persist training controls; explicit CLI choices win.
    for name in ("epochs", "patience", "lr_patience", "min_delta", "smooth_l1_beta", "clip_grad_norm", "tail_fraction", "folds", "seed"):
        option = "--" + name.replace("_", "-")
        explicit = any(token == option or token.startswith(option + "=") for token in sys.argv[1:])
        if name in config and not explicit:
            setattr(args, name, config[name])
    if args.precision == "auto":
        args.precision = config.get("precision", "fp32") if torch.cuda.is_available() and not args.cpu else "fp32"
    if min(args.threads, args.epochs, args.patience, args.scaler_chunk_size) < 1 or args.num_workers < 0:
        raise ValueError("Threads, epochs, patience and chunk size must be positive; workers nonnegative")
    if args.folds < 2 or not 0 < args.tail_fraction <= 1 or any(not 0 < f <= 1 for f in args.report_fractions):
        raise ValueError("Need folds >= 2 and fractions in (0, 1]")
    if args.smooth_l1_beta <= 0 or args.clip_grad_norm <= 0 or (args.max_rows is not None and args.max_rows < 1):
        raise ValueError("Beta, clipping norm and max rows must be positive")
    if args.lr_patience < 0 or args.min_delta < 0:
        raise ValueError("LR patience and min delta must be nonnegative")
    if args.patience <= args.lr_patience + 1:
        raise ValueError("Early-stop patience must exceed lr-patience + 1 to allow training after LR reduction")
    torch.set_num_threads(args.threads)
    started = time.perf_counter()
    cpu_started = resource.getrusage(resource.RUSAGE_SELF)
    set_seed(args.seed)
    x, y, ids, names, alignment = load_training_inputs(
        args.descriptor_dir, args.targets, args.target_ids, args.target_id_column,
        args.x_file, args.target)
    if int(config.get("input_dim", x.shape[1])) != x.shape[1]:
        raise ValueError("Config input_dim does not match descriptors")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    config.update(epochs=args.epochs, folds=args.folds, seed=args.seed, precision=args.precision,
                  patience=args.patience, lr_patience=args.lr_patience, min_delta=args.min_delta,
                  tail_fraction=args.tail_fraction, smooth_l1_beta=args.smooth_l1_beta,
                  clip_grad_norm=args.clip_grad_norm)
    save_training_inputs(args.output_dir, names, ids, alignment, config)
    assignments = []
    (args.output_dir / "training_config.json").write_text(
        json.dumps(vars(args), default=str, indent=2) + "\n"
    )
    required = {"hidden_dims", "activation", "optimizer", "lr", "dropout", "weight_decay", "batch_size"}
    if required - config.keys():
        raise ValueError(f"Missing tuned settings: {sorted(required - config.keys())}")
    if int(config["batch_size"]) < 1:
        raise ValueError("Grid batch_size must be positive")
    hidden_dims = [int(value) for value in config["hidden_dims"].split(",")]
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    def gpu_start():
        if device.type != 'cuda':
            return None
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        return event
    def gpu_finish(event):
        if event is None:
            return None
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        end.synchronize()
        return event.elapsed_time(end) / 1000.0
    if args.cache_device == "cuda" and device.type != "cuda":
        raise ValueError("CUDA cache requested but CUDA is unavailable or --cpu was supplied")
    def stamp():
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        return time.perf_counter()
    amp_enabled = device.type == "cuda" and args.precision == "amp"
    if args.precision == "amp" and not amp_enabled:
        raise ValueError("AMP precision requires CUDA; use --precision fp32 for CPU runs")
    tuned_input_dim = int(config.get("input_dim", x.shape[1]))
    feature_transfer = False
    if "target" in config and config["target"] != args.target:
        raise ValueError("Config target does not match --target")
    valid = np.flatnonzero(np.isfinite(y))
    if args.max_rows:
        valid = valid[:args.max_rows]
    if len(valid) < 2 * args.folds:
        raise ValueError("Require at least two validation rows per fold")
    print(f"Device={device}; rows={len(valid):,}; config={config}")

    splitter = KFold(args.folds, shuffle=True, random_state=args.seed)
    oof_prediction = np.full(len(valid), np.nan, np.float32)
    oof_fold = np.zeros(len(valid), np.int16)
    fold_rows = []
    runtime_rows = []
    for fold, (train_pos, val_pos) in enumerate(splitter.split(valid), 1):
        scaler_started = stamp()
        train_rows, val_rows = valid[train_pos], valid[val_pos]
        assignments.extend((int(row), ids[row], fold) for row in val_rows)
        x_mean, x_std = compute_scaler(x, train_rows, args.scaler_chunk_size)
        y_mean, y_std = float(y[train_rows].mean()), float(y[train_rows].std() + 1e-8)
        scaler_seconds = stamp() - scaler_started
        if not np.isfinite(x_mean).all() or not np.isfinite(x_std).all() or (x_std <= 0).any() or not np.isfinite(y_std):
            raise ValueError("Nonfinite or invalid fold scalers")
        # Save the exact fold-local transformation alongside the checkpoint.
        np.savez_compressed(
            args.output_dir / f"fold_{fold}_metadata.npz",
            x_mean=x_mean, x_std=x_std,
            y_mean=np.asarray(y_mean), y_std=np.asarray(y_std),
            input_dim=np.asarray(x.shape[1]), target=np.asarray(args.target),
            seed=np.asarray(args.seed), folds=np.asarray(args.folds),
            fold=np.asarray(fold), valid_rows=np.asarray(len(valid)),
            train_rows=train_rows, validation_rows=val_rows,
            hidden_dims=np.asarray(hidden_dims),
            activation=np.asarray(config["activation"]),
            dropout=np.asarray(float(config["dropout"])),
            optimizer=np.asarray(config["optimizer"]), lr=np.asarray(float(config["lr"])),
            weight_decay=np.asarray(float(config["weight_decay"])),
            batch_size=np.asarray(int(config["batch_size"])),
            x_file=np.asarray(args.x_file), precision=np.asarray(args.precision),
            hyperparameters_transferred=np.asarray(feature_transfer),
            grid_input_dim=np.asarray(tuned_input_dim if tuned_input_dim is not None else -1),
            selection_metric=np.asarray(f"ndcg_at_{args.tail_fraction:g}"),
            scheduler_metric=np.asarray(f"ndcg_at_{args.tail_fraction:g}"),
            lr_patience=np.asarray(args.lr_patience), min_delta=np.asarray(args.min_delta),
            early_stop_patience=np.asarray(args.patience),
            preprocessing=np.asarray("nonfinite_to_zero_then_standardize"),
        )
        cache_started = stamp()
        print(f"Fold {fold}: normalized cache {(len(train_rows)+len(val_rows))*(x.shape[1]+1)*4/2**30:.2f} GiB; "
              f"cache={args.cache_device}", flush=True)
        train_ds = CachedDescriptorDataset(x, y, train_rows, x_mean, x_std, y_mean, y_std, args.scaler_chunk_size)
        val_ds = CachedDescriptorDataset(x, y, val_rows, x_mean, x_std, y_mean, y_std, args.scaler_chunk_size)
        train_loader = make_loader(train_ds, args, device, int(config["batch_size"]), True)
        val_loader = make_loader(val_ds, args, device, int(config["batch_size"]), False)
        cache_seconds = stamp() - cache_started
        model = SearchConsensusMLP(
            x.shape[1], hidden_dims, float(config["dropout"]), config["activation"]
        ).to(device)
        optimizer = make_optimizer(
            config["optimizer"], model.parameters(), float(config["lr"]),
            float(config["weight_decay"]), args.momentum,
        )
        grad_scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
        criterion = nn.SmoothL1Loss(beta=args.smooth_l1_beta)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=args.lr_patience,
            threshold=args.min_delta, threshold_mode="abs",
        )
        best_ndcg, best_epoch, stale = -np.inf, 0, 0
        model_path = args.output_dir / f"fold_{fold}_best_model.pt"
        history = []
        for epoch in range(1, args.epochs + 1):
            model.train()
            epoch_started = stamp()
            training_gpu_started = gpu_start()
            loss_sum = torch.zeros((), device=device)
            batches = 0
            for features, target in train_loader:
                features, target = features.to(device, non_blocking=True), target.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                    loss = criterion(model(features), target)
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                grad_scaler.step(optimizer)
                grad_scaler.update()
                loss_sum += loss.detach()
                batches += 1
            training_cuda_elapsed_seconds = gpu_finish(training_gpu_started)
            train_loss = (loss_sum / batches).item()
            training_seconds = stamp() - epoch_started
            validation_started = stamp()
            validation_gpu_started = gpu_start()
            val_prediction = predict(model, val_loader, device, y_mean, y_std, amp_enabled)
            validation_cuda_elapsed_seconds = gpu_finish(validation_gpu_started)
            validation_metrics = tail_metrics(y[val_rows], val_prediction, args.tail_fraction)
            recall = validation_metrics[f"recall_at_{args.tail_fraction:g}"]
            ndcg = validation_metrics[f"ndcg_at_{args.tail_fraction:g}"]
            validation_seconds = stamp() - validation_started
            lr_used = optimizer.param_groups[0]["lr"]
            scheduler.step(ndcg)
            history.append({"epoch": epoch, "train_loss": train_loss,
                            "val_recall": recall, "val_ndcg": ndcg,
                            "lr": lr_used, "next_lr": optimizer.param_groups[0]["lr"],
                            "training_seconds": training_seconds,
                            "training_cuda_elapsed_seconds": training_cuda_elapsed_seconds,
                            "validation_cuda_elapsed_seconds": validation_cuda_elapsed_seconds,
                            "validation_seconds": validation_seconds})
            print(f"Fold {fold} epoch {epoch}: loss={train_loss:.6f}; recall={recall:.6f}; NDCG={ndcg:.6f}; LR={lr_used:.2e}; "
                  f"train={training_seconds:.2f}s; val={history[-1]['validation_seconds']:.2f}s", flush=True)
            if ndcg > best_ndcg + args.min_delta:
                best_ndcg, best_epoch, stale = ndcg, epoch, 0
                torch.save(model.state_dict(), model_path)
            else:
                stale += 1
                if stale >= args.patience:
                    break
        save_history_plot(pd.DataFrame(history), args.output_dir / f"fold_{fold}_learning_curve.png", fold)
        pd.DataFrame(history).to_csv(args.output_dir / f"fold_{fold}_history.csv", index=False)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        fold_prediction = predict(model, val_loader, device, y_mean, y_std, amp_enabled)
        oof_prediction[val_pos], oof_fold[val_pos] = fold_prediction, fold
        metrics = all_metrics(y[val_rows], fold_prediction, args.report_fractions)
        fold_rows.append({"fold": fold, **metrics})
        runtime_rows.append({"fold": fold, "scaler_seconds": scaler_seconds, "cache_seconds": cache_seconds,
                             "epochs": len(history), "best_epoch": best_epoch, "best_ndcg": best_ndcg, "training_seconds": sum(h["training_seconds"] for h in history),
                             "training_cuda_elapsed_seconds": sum(h["training_cuda_elapsed_seconds"] for h in history) if device.type == 'cuda' else None,
                             "validation_cuda_elapsed_seconds": sum(h["validation_cuda_elapsed_seconds"] for h in history) if device.type == 'cuda' else None,
                             "validation_seconds": sum(h["validation_seconds"] for h in history)})
        del model, optimizer, scheduler, train_loader, val_loader, train_ds, val_ds
        print(f"Fold {fold}: best NDCG={best_ndcg:.6f} at epoch {best_epoch}; RMSE={metrics['RMSE']:.6f}")

    pd.DataFrame(assignments, columns=["row_index", "zinc_id", "validation_fold"]).sort_values("row_index").to_csv(
        args.output_dir / "fold_assignments.csv", index=False)
    fold_metrics = pd.DataFrame(fold_rows)
    fold_metrics.to_csv(args.output_dir / "fold_metrics.csv", index=False)
    pd.DataFrame({
        "zinc_id": ids[valid], "fold": oof_fold, "actual_target": y[valid],
        "predicted_target": oof_prediction,
    }).to_csv(args.output_dir / "out_of_fold_predictions.csv", index=False)
    summary = {
        "grid_config": config, "valid_rows": int(len(valid)),
        "hyperparameters_transferred": bool(feature_transfer),
        "grid_input_dim": tuned_input_dim, "training_input_dim": int(x.shape[1]),
        "grid_x_file": config.get("x_file"), "training_x_file": args.x_file,
        "selection_metric": f"ndcg_at_{args.tail_fraction:g}",
        "scheduler": {"name": "ReduceLROnPlateau", "metric": "validation_ndcg",
                      "mode": "max", "factor": 0.5, "patience": args.lr_patience,
                      "threshold": args.min_delta, "threshold_mode": "abs"},
        "grid_selection_metric": config.get("selection_metric", "NDCG (legacy grid search)"),
        "scalers_saved": True, "folds": args.folds, "seed": args.seed,
        "training_config": vars(args),
        "OOF": all_metrics(y[valid], oof_prediction, args.report_fractions),
        "fold_metric_means": {
            column: float(fold_metrics[column].mean())
            for column in fold_metrics if column != "fold"
        },
    }
    with (args.output_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, default=str)
    usage = resource.getrusage(resource.RUSAGE_SELF)
    runtime = {"main_wall_seconds": stamp() - started, "device": str(device),
               "threads": args.threads, "num_workers": args.num_workers, "cache_device": args.cache_device,
               "user_cpu_seconds": usage.ru_utime - cpu_started.ru_utime,
               "system_cpu_seconds": usage.ru_stime - cpu_started.ru_stime,
               "total_cpu_seconds": usage.ru_utime - cpu_started.ru_utime + usage.ru_stime - cpu_started.ru_stime,
               "gpu_model": torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
               "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
               "precision": args.precision,
               "peak_gpu_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == 'cuda' else None,
               "training_cuda_elapsed_seconds": sum(r['training_cuda_elapsed_seconds'] for r in runtime_rows) if device.type == 'cuda' else None,
               "validation_cuda_elapsed_seconds": sum(r['validation_cuda_elapsed_seconds'] for r in runtime_rows) if device.type == 'cuda' else None,
               "cuda_timing_scope": "CUDA events on the current stream around training batches and validation prediction; elapsed intervals include host submission gaps and transfers, not a sum of GPU kernel busy time. Excludes scaler/cache preparation, validation ranking metrics, final fold prediction and checkpoint I/O. CPU and GPU intervals overlap and must not be added.",
               "scope": "Main function excluding imports; CPU times exclude loader child workers. Use /usr/bin/time -v for whole-job CPU time.",
               "folds": runtime_rows}
    (args.output_dir / "runtime_summary.json").write_text(json.dumps(runtime, indent=2) + "\n")
    print(json.dumps(summary["OOF"], indent=2))
    print(f"Wrote results to {args.output_dir}")


if __name__ == "__main__":
    main()
