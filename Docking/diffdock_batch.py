#!/usr/bin/env python3
"""Run SDF ligand batches with the DiffDockHPC DiffDock-L inference interface.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def worker(config_path):
    """Load models once per shard using the selected inference environment."""
    config = json.loads(Path(config_path).read_text())
    os.chdir(config["diffdock_dir"])
    sys.path.insert(0, config["diffdock_dir"])
    import numpy as np
    import torch
    import yaml

    if config["device"] == "gpu" and not torch.cuda.is_available():
        raise RuntimeError("GPU requested but CUDA is unavailable; refusing CPU fallback")
    if config["device"] == "cpu" and torch.cuda.is_available():
        raise RuntimeError("CPU requested but CUDA is still visible; check container environment settings")
    seed = config["seed"]
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if config["device"] == "gpu":
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(config["cpus_per_job"])
    spec = importlib.util.spec_from_file_location("batch_diffdock_inference", config["inference"])
    inference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(inference)
    args = inference.get_parser().parse_args(["--config", config["inference_config"]])
    with Path(config["inference_config"]).open() as handle:
        settings = yaml.safe_load(handle)
    args.config.close()
    if not isinstance(settings, dict):
        raise ValueError("Inference configuration must be a YAML mapping")
    args.__dict__.update(settings)
    # Apply wrapper settings last: upstream main() otherwise lets YAML override CLI.
    args.config = None
    args.protein_ligand_csv = config["inputs"]
    args.esm_embeddings_path = config["embedding"]
    args.out_dir = config["poses"]
    args.samples_per_complex = config["samples"]
    args.batch_size = config["batch_size"]
    args.cores = config["cpus_per_job"]
    args.seperate_dirs = False
    args.save_visualisation = False
    args.gnina_minimize = False
    args.gnina_full_dock = False
    args.remove_output_hs = config["remove_output_hs"]
    for key in ("model_dir", "confidence_model_dir"):
        selected = config.get(key) or getattr(args, key, None)
        if not selected:
            raise ValueError(f"A local {key} is required")
        path = Path(selected).resolve()
        setattr(args, key, str(path))
        checkpoint = args.ckpt if key == "model_dir" else args.confidence_ckpt
        for required in (path / "model_parameters.yml", path / checkpoint):
            if not required.is_file():
                raise FileNotFoundError(f"Missing model file: {required}")
    effective = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    Path(config["effective_config"]).write_text(json.dumps(effective, indent=2) + "\n")
    inference.main(args)
    if config["device"] == "gpu":
        torch.cuda.synchronize()


def run_shard(command, env, cwd, log, timeout):
    started = time.monotonic()
    code, error = None, ""
    try:
        with log.open("w") as handle:
            result = subprocess.run(command, env=env, cwd=cwd, stdout=handle,
                                    stderr=subprocess.STDOUT, timeout=timeout)
        code = result.returncode
        if code:
            error = "Worker failed; inspect its log"
    except subprocess.TimeoutExpired:
        error = "Worker exceeded the shard timeout"
    except OSError as exc:
        error = str(exc)
    return dict(exit_code=code, error=error, elapsed_seconds=round(time.monotonic() - started, 3))


def has_record(path):
    with path.open(errors="replace") as handle:
        return any(line.strip() == "$$$$" for line in handle)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receptor", required=True, type=Path, help="Prepared PDB receptor")
    parser.add_argument("--ligands", required=True, type=Path, help="Directory of single-molecule SDF files")
    parser.add_argument("--embedding", required=True, type=Path, help="Matching DiffDockHPC ESM embedding (.pt)")
    parser.add_argument("--diffdock-dir", required=True, type=Path, help="DiffDockHPC installation directory")
    parser.add_argument("--output", required=True, type=Path, help="New output directory (must not exist)")
    parser.add_argument("--config", type=Path, help="Inference YAML (default: installation's default_inference_args.yaml)")
    parser.add_argument("--model-dir", type=Path, help="Override score model directory")
    parser.add_argument("--confidence-model-dir", type=Path, help="Override confidence model directory")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--jobs", type=positive_int, default=1, help="Model-loaded worker processes (default: 1)")
    parser.add_argument("--cpus-per-job", type=positive_int, default=1)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--gpu-ids", nargs="+", help="Distinct GPU IDs from the current CUDA_VISIBLE_DEVICES allocation")
    parser.add_argument("--samples", type=positive_int, default=3, help="Samples generated per ligand (default: 3)")
    parser.add_argument("--batch-size", type=positive_int, default=3, help="Samples per diffusion minibatch, not ligand count")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--remove-output-hs", action="store_true")
    parser.add_argument("--python", default="python3", help="Python executable in the inference environment")
    parser.add_argument("--container", type=Path, help="Optional DiffDockHPC Apptainer/Singularity image")
    parser.add_argument("--container-runtime", default="apptainer")
    parser.add_argument("--timeout", type=positive_int, help="Maximum seconds per worker shard, including model loading")
    parser.add_argument("--prepare-only", action="store_true", help="Write manifests and commands without starting inference")
    args = parser.parse_args()
    for key in ("receptor", "ligands", "embedding", "diffdock_dir", "output", "config",
                "model_dir", "confidence_model_dir", "container"):
        value = getattr(args, key)
        if value is not None:
            setattr(args, key, value.resolve())
    if not args.receptor.is_file() or args.receptor.suffix.lower() != ".pdb":
        parser.error("--receptor must be an existing PDB file")
    if not args.embedding.is_file() or not args.ligands.is_dir():
        parser.error("embedding file and ligand directory must exist")
    if args.output.exists():
        parser.error("--output already exists; choose a new directory")
    inference = args.diffdock_dir / "inference.py"
    config = args.config or args.diffdock_dir / "default_inference_args.yaml"
    if not inference.is_file() or not config.is_file():
        parser.error("DiffDockHPC inference.py and inference YAML must exist")
    for directory in (args.model_dir, args.confidence_model_dir):
        if directory is not None and not directory.is_dir():
            parser.error(f"Model directory does not exist: {directory}")
    allocated = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    if args.jobs * args.cpus_per_job > allocated:
        parser.error(f"requested more CPU threads than the available {allocated}")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    allowed = visible.split(",") if visible is not None else None
    if args.device == "gpu":
        gpu_ids = args.gpu_ids or (allowed[:args.jobs] if allowed else [str(i) for i in range(args.jobs)])
        if len(gpu_ids) != args.jobs or len(set(gpu_ids)) != len(gpu_ids):
            parser.error("GPU mode requires one distinct GPU ID per worker")
        if any(not re.fullmatch(r"[A-Za-z0-9_.:/-]+", item) or item == "-1" for item in gpu_ids):
            parser.error("Invalid GPU ID")
        if allowed is not None and any(item not in allowed for item in gpu_ids):
            parser.error("Requested GPU is outside CUDA_VISIBLE_DEVICES")
    else:
        if args.gpu_ids:
            parser.error("--gpu-ids is only supported in GPU mode")
        gpu_ids = [""] * args.jobs
    iterator = args.ligands.rglob("*") if args.recursive else args.ligands.iterdir()
    ligands = sorted(p for p in iterator if p.is_file() and p.suffix.lower() == ".sdf")
    if not ligands:
        parser.error("No SDF files found")
    if len({path.stem for path in ligands}) != len(ligands):
        parser.error("Ligand filename stems must be unique across the batch")
    for ligand in ligands:
        with ligand.open(errors="replace") as handle:
            records = sum(line.strip() == "$$$$" for line in handle)
        if records != 1:
            parser.error(f"Expected one complete SDF record per file: {ligand}")
    prefix = []
    if args.container:
        runtime = shutil.which(args.container_runtime)
        if not args.container.is_file() or not runtime:
            parser.error("Container image and runtime must exist")
        # This image's runscript selects the DiffDockHPC conda environment.
        prefix = [runtime, "run"]
        if args.device == "gpu":
            prefix.append("--nv")
        binds = {args.diffdock_dir, args.receptor.parent, args.embedding.parent,
                 args.ligands, args.output, Path(__file__).resolve().parent}
        binds.update(p for p in (args.model_dir, args.confidence_model_dir) if p)
        binds.add(config.parent)
        if any("," in str(p) or ":" in str(p) for p in binds):
            parser.error("Container bind paths cannot contain commas or colons")
        for path in sorted(binds):
            prefix += ["--bind", f"{path}:{path}"]
        prefix += ["--pwd", str(args.diffdock_dir), str(args.container), args.python]
    else:
        python = shutil.which(args.python)
        if not python:
            parser.error("Inference Python executable not found")
        prefix = [str(Path(python).resolve())]
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(config, output / "inference_args.yaml")
    # Freeze the inference entry point; imported upstream utility modules remain external.
    shutil.copyfile(inference, output / "inference_source.py")
    run = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    run.update(receptor_sha256=digest(args.receptor), embedding_sha256=digest(args.embedding),
               inference_sha256=digest(inference), inference_config_sha256=digest(config),
               search_mode="whole_protein", gpu_ids=gpu_ids)
    entries = [dict(complex_name=f"mol_{i:08d}", ligand=str(p), shard=i % args.jobs)
               for i, p in enumerate(ligands)]
    run["ligands"] = entries
    (output / "run_config.json").write_text(json.dumps(run, indent=2) + "\n")
    shards = []
    for index in range(min(args.jobs, len(ligands))):
        folder = output / f"shard_{index:04d}"
        folder.mkdir()
        inputs = folder / "inputs.csv"
        with inputs.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["complex_name", "protein_path", "ligand_description"])
            for entry in entries[index::args.jobs]:
                writer.writerow([entry["complex_name"], str(args.receptor), entry["ligand"]])
        settings = dict(run, inference=str(output / "inference_source.py"),
                        inference_config=str(output / "inference_args.yaml"), inputs=str(inputs),
                        poses=str(folder / "poses"), effective_config=str(folder / "effective_config.json"),
                        seed=args.seed + index)
        worker_config = folder / "worker_config.json"
        worker_config.write_text(json.dumps(settings, indent=2) + "\n")
        command = prefix + ["-u", str(Path(__file__).resolve()), "--worker-config", str(worker_config)]
        env = os.environ.copy()
        overrides = dict(CUDA_VISIBLE_DEVICES=gpu_ids[index], OMP_NUM_THREADS=str(args.cpus_per_job),
                         MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        env.update(overrides)
        if args.container:
            for name, value in overrides.items():
                env[f"APPTAINERENV_{name}"] = value
                env[f"SINGULARITYENV_{name}"] = value
        (folder / "command.json").write_text(json.dumps(dict(command=command, environment=overrides), indent=2) + "\n")
        shards.append((command, env, args.diffdock_dir, folder / "inference.log", args.timeout))
    if args.prepare_only:
        print(f"Prepared {len(ligands)} ligands in {len(shards)} shards at {output}; inference was not started")
        return 0
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=len(shards)) as pool:
        futures = [pool.submit(run_shard, *values) for values in shards]
        statuses = [future.result() for future in futures]
    for index, status in enumerate(statuses):
        (output / f"shard_{index:04d}" / "process_status.json").write_text(json.dumps(status, indent=2) + "\n")
    # Scan each output directory once, rather than once per ligand in large libraries.
    saved_ranks = {}
    pattern = re.compile(r"VS_DD_(mol_[0-9]+)_rank([0-9]+)_confidence.*\.sdf$")
    for index in range(len(shards)):
        for path in (output / f"shard_{index:04d}" / "poses").glob("VS_DD_*.sdf"):
            match = pattern.fullmatch(path.name)
            if match and has_record(path):
                saved_ranks.setdefault((index, match.group(1)), set()).add(int(match.group(2)))
    failures = 0
    with (output / "results.csv").open("w", newline="") as handle:
        fields = ["complex_name", "ligand", "shard", "status", "poses_saved", "output_dir", "log", "error"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for entry in entries:
            folder = output / f"shard_{entry['shard']:04d}"
            poses = folder / "poses"
            ranks = saved_ranks.get((entry["shard"], entry["complex_name"]), set())
            success = ranks == set(range(1, args.samples + 1))
            failures += not success
            error = "" if success else (statuses[entry["shard"]]["error"] or "Missing or incomplete predicted poses; inspect the worker log")
            writer.writerow(dict(entry, status="success" if success else "failed", poses_saved=len(ranks),
                                 output_dir=str(poses), log=str(folder / "inference.log"), error=error))
    worker_failures = sum(status["exit_code"] != 0 for status in statuses)
    summary = dict(files=len(entries), successes=len(entries) - failures, failures=failures,
                   worker_failures=worker_failures, elapsed_seconds=round(time.monotonic() - started, 3))
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 1 if failures or worker_failures else 0


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--worker-config":
        worker(sys.argv[2])
    else:
        sys.exit(main())
