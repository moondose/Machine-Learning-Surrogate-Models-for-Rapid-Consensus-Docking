#!/usr/bin/env python3
"""Dock a directory of SDF files into a specified binding pocket using GNINA.

"""

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import csv
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def has_record(path):
    """Check for a complete SDF record without loading the output into memory."""
    if not path.is_file():
        return False
    with path.open(errors="replace") as handle:
        return any(line.strip() == "$$$$" for line in handle)


def dock(ligand, relative, args, prefix, output):
    # Preserve the input directory structure to avoid duplicate filename collisions.
    pose = output / "poses" / relative
    log = output / "logs" / relative.with_suffix(".log")
    pose.parent.mkdir(parents=True, exist_ok=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    command = prefix + [
        "--receptor", str(args.receptor), "--ligand", str(ligand),
        "--out", str(pose), "--cpu", str(args.cpus_per_job),
        "--exhaustiveness", str(args.exhaustiveness),
        "--num_modes", str(args.num_modes), "--seed", str(args.seed),
        "--cnn_scoring", args.cnn_scoring,
    ]
    for axis, center, size in zip("xyz", args.center, args.size):
        command += [f"--center_{axis}", str(center), f"--size_{axis}", str(size)]
    if args.device == "cpu":
        command.append("--no_gpu")
    else:
        command += ["--device", str(args.gpu_id)]
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS=str(args.cpus_per_job), MKL_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1")
    started = time.monotonic()
    error = ""
    code = None
    try:
        with ligand.open(errors="replace") as handle:
            if sum(line.strip() == "$$$$" for line in handle) != 1:
                raise ValueError("Expected exactly one complete molecule per SDF file")
        with log.open("w") as handle:
            result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT,
                                    env=env, timeout=args.timeout)
        code = result.returncode
        success = code == 0 and has_record(pose)
        if not success:
            error = "GNINA failed or produced no complete SDF record; inspect the log"
    except subprocess.TimeoutExpired:
        success, error = False, "GNINA exceeded the per-file timeout"
    except (OSError, ValueError) as exc:
        success, error = False, str(exc)
    return dict(ligand=str(ligand), output=str(pose), log=str(log),
                status="success" if success else "failed", exit_code=code,
                wall_seconds=round(time.monotonic() - started, 3), error=error,
                command=command)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receptor", required=True, type=Path, help="Prepared PDB or PDBQT receptor")
    parser.add_argument("--ligands", required=True, type=Path, help="Directory containing SDF files")
    parser.add_argument("--output", required=True, type=Path, help="New output directory (must not exist)")
    parser.add_argument("--center", required=True, nargs=3, type=float, metavar=("X", "Y", "Z"),
                        help="Binding pocket center in Angstroms")
    parser.add_argument("--size", required=True, nargs=3, type=float, metavar=("X", "Y", "Z"),
                        help="Full box dimensions in Angstroms")
    parser.add_argument("--recursive", action="store_true", help="Include ligand subdirectories")
    parser.add_argument("--jobs", type=positive_int, default=1, help="Concurrent GNINA processes (default: 1)")
    parser.add_argument("--cpus-per-job", type=positive_int, default=1, help="CPUs per GNINA process (default: 1)")
    parser.add_argument("--gnina", default="gnina", help="GNINA executable path or command name")
    parser.add_argument("--container", type=Path, help="Optional Apptainer/Singularity image")
    parser.add_argument("--container-runtime", default="apptainer", help="Container runtime (default: apptainer)")
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--gpu-id", type=int, default=0, help="GNINA GPU device index (default: 0)")
    parser.add_argument("--exhaustiveness", type=positive_int, default=8)
    parser.add_argument("--num-modes", type=positive_int, default=9)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--cnn-scoring", choices=("none", "rescore", "refinement", "all"), default="rescore")
    parser.add_argument("--timeout", type=positive_int, help="Maximum seconds per SDF file")
    args = parser.parse_args()
    args.receptor = args.receptor.resolve()
    args.ligands = args.ligands.resolve()
    output = args.output.resolve()
    if not args.receptor.is_file() or args.receptor.suffix.lower() not in (".pdb", ".pdbqt"):
        parser.error("--receptor must be an existing PDB or PDBQT file")
    if not args.ligands.is_dir():
        parser.error("--ligands must be an existing directory")
    if output.exists():
        parser.error("--output already exists; choose a new directory")
    if not all(math.isfinite(x) for x in args.center + args.size) or any(x <= 0 for x in args.size):
        parser.error("box values must be finite and all dimensions must be positive")
    if args.gpu_id < 0:
        parser.error("--gpu-id must be nonnegative")
    if args.device == "gpu" and args.jobs != 1:
        parser.error("GPU mode requires --jobs 1 to avoid competing processes on one GPU")
    iterator = args.ligands.rglob("*") if args.recursive else args.ligands.iterdir()
    ligands = sorted(p for p in iterator if p.is_file() and p.suffix.lower() == ".sdf")
    if not ligands:
        parser.error("no SDF files found")
    if len({path.stem for path in ligands}) != len(ligands):
        parser.error("Ligand filename stems must be unique across the batch")
    # Retain relative names rather than flattening nested ligand directories.
    relatives = [p.relative_to(args.ligands) for p in ligands]
    if args.container:
        args.container = args.container.resolve()
        if not args.container.is_file():
            parser.error("container image does not exist")
        runtime = shutil.which(args.container_runtime)
        if not runtime:
            parser.error("container runtime not found")
        prefix = [runtime, "exec"]
        if args.device == "gpu":
            prefix.append("--nv")
        # Bind the same absolute paths inside the container, including external files.
        binds = {args.receptor.parent, args.ligands, output}
        if "/" in args.gnina:
            args.gnina = str(Path(args.gnina).resolve())
            if not Path(args.gnina).is_file():
                parser.error("path-based --gnina must exist on the host; use a command name for an image-provided executable")
            binds.add(Path(args.gnina).parent)
        if any("," in str(p) or ":" in str(p) for p in binds):
            parser.error("container bind paths cannot contain commas or colons")
        for folder in sorted(binds):
            prefix += ["--bind", f"{folder}:{folder}"]
        prefix += [str(args.container), args.gnina]
    else:
        executable = shutil.which(args.gnina)
        if not executable:
            parser.error("GNINA executable not found or not executable")
        prefix = [str(Path(executable).resolve())]
    allocated = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    requested = args.jobs * args.cpus_per_job
    if requested > allocated:
        parser.error(f"requested {requested} CPU threads but only {allocated} are available; reduce jobs or CPUs per job")
    output.mkdir(parents=True, exist_ok=False)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config["ligand_files"] = [str(p) for p in ligands]
    config["command_prefix"] = prefix
    (output / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")
    failures = 0
    started = time.monotonic()
    fields = ["ligand", "output", "log", "status", "exit_code", "wall_seconds", "error"]
    # Keep at most jobs futures in memory, even for very large ligand libraries.
    items = iter(zip(ligands, relatives))
    with (output / "results.csv").open("w", newline="") as csv_handle, \
            (output / "commands.jsonl").open("w") as commands, \
            ThreadPoolExecutor(max_workers=args.jobs) as pool:
        writer = csv.DictWriter(csv_handle, fieldnames=fields)
        writer.writeheader()
        pending = set()
        for _ in range(min(args.jobs, len(ligands))):
            ligand, relative = next(items)
            pending.add(pool.submit(dock, ligand, relative, args, prefix, output))
        completed = 0
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                row = future.result()
                commands.write(json.dumps({"ligand": row["ligand"], "command": row.pop("command")}) + "\n")
                commands.flush()
                writer.writerow(row)
                csv_handle.flush()
                completed += 1
                failures += row["status"] != "success"
                print(f"[{completed}/{len(ligands)}] {row['status']}: {row['ligand']}", flush=True)
                item = next(items, None)
                if item:
                    pending.add(pool.submit(dock, *item, args, prefix, output))
    summary = dict(files=len(ligands), successes=len(ligands) - failures,
                   failures=failures, elapsed_seconds=round(time.monotonic() - started, 3))
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
