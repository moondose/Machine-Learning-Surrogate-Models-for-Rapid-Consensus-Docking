#!/usr/bin/env python3
"""Dock prepared MOL2 ligands into a binding pocket using classic LeDock.
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


def count_molecules(path):
    """Classic batch execution expects one molecule in each MOL2 file."""
    with path.open(errors="replace") as handle:
        return sum(line.strip().upper() == "@<TRIPOS>MOLECULE" for line in handle)


def dock(ligand, relative, args, executable, output):
    # Keep the original extension in the directory name to prevent stem collisions.
    work = output / "work" / relative
    pose = work / "ligand.dok"
    log = work / "ledock.log"
    command = [executable, "dock.in"]
    started = time.monotonic()
    code, poses, error = None, 0, ""
    try:
        work.mkdir(parents=True, exist_ok=False)
        if count_molecules(ligand) != 1:
            raise ValueError("Expected exactly one @<TRIPOS>MOLECULE record per MOL2 file")
        shutil.copyfile(ligand, work / "ligand.mol2")
        # Short local filenames avoid limitations in LeDock's input path parsing.
        (work / "pro.pdb").symlink_to(output / "receptor.pdb")
        (work / "ligands.list").write_text("ligand.mol2\n")
        bounds = "\n".join(f"{center-size/2:.6f} {center+size/2:.6f}"
                           for center, size in zip(args.center, args.size))
        (work / "dock.in").write_text(
            f"Receptor\npro.pdb\n\nRMSD\n{args.rmsd}\n\n"
            f"Binding pocket\n{bounds}\n\nNumber of binding poses\n{args.num_modes}\n\n"
            "Ligands list\nligands.list\n\nEND\n")
        env = os.environ.copy()
        env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        with log.open("w") as handle:
            result = subprocess.run(command, cwd=work, stdout=handle,
                                    stderr=subprocess.STDOUT, env=env, timeout=args.timeout)
        code = result.returncode
        if pose.is_file():
            with pose.open(errors="replace") as handle:
                poses = sum(line.startswith("REMARK Cluster") for line in handle)
        if code != 0 or poses == 0:
            error = "LeDock failed or produced no pose clusters; inspect the log"
    except subprocess.TimeoutExpired:
        error = "LeDock exceeded the per-ligand timeout"
    except (OSError, ValueError) as exc:
        error = str(exc)
    success = code == 0 and poses > 0 and not error
    return dict(ligand=str(ligand), output=str(pose), log=str(log),
                status="success" if success else "failed", exit_code=code,
                poses_saved=poses, wall_seconds=round(time.monotonic() - started, 3),
                error=error, command=command, cwd=str(work))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receptor", required=True, type=Path, help="Prepared PDB receptor, e.g. LePro's pro.pdb")
    parser.add_argument("--ligands", required=True, type=Path, help="Directory of single-molecule SYBYL MOL2 files")
    parser.add_argument("--output", required=True, type=Path, help="New output directory (must not exist)")
    parser.add_argument("--center", required=True, nargs=3, type=float, metavar=("X", "Y", "Z"),
                        help="Binding pocket center in Angstroms")
    parser.add_argument("--size", required=True, nargs=3, type=float, metavar=("X", "Y", "Z"),
                        help="Full box dimensions in Angstroms")
    parser.add_argument("--recursive", action="store_true", help="Include ligand subdirectories")
    parser.add_argument("--jobs", type=positive_int, default=1, help="Concurrent single-CPU LeDock processes (default: 1)")
    parser.add_argument("--ledock", default="ledock", help="LeDock executable path or command name")
    parser.add_argument("--num-modes", type=positive_int, default=20, help="Requested binding poses (default: 20)")
    parser.add_argument("--rmsd", type=float, default=1.0, help="Pose clustering RMSD in Angstroms (default: 1.0)")
    parser.add_argument("--timeout", type=positive_int, help="Maximum LeDock seconds per ligand")
    args = parser.parse_args()
    args.receptor = args.receptor.resolve()
    args.ligands = args.ligands.resolve()
    output = args.output.resolve()
    if not args.receptor.is_file() or args.receptor.suffix.lower() != ".pdb":
        parser.error("--receptor must be an existing prepared PDB file")
    if not args.ligands.is_dir():
        parser.error("--ligands must be an existing directory")
    if output.exists():
        parser.error("--output already exists; choose a new directory")
    if not all(math.isfinite(x) for x in args.center + args.size) or any(x <= 0 for x in args.size):
        parser.error("box values must be finite and all dimensions must be positive")
    if not math.isfinite(args.rmsd) or args.rmsd <= 0:
        parser.error("--rmsd must be finite and positive")
    executable = shutil.which(args.ledock)
    if not executable:
        parser.error("LeDock executable not found or not executable")
    executable = str(Path(executable).resolve())
    allocated = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    if args.jobs > allocated:
        parser.error(f"requested {args.jobs} processes but only {allocated} CPUs are available")
    iterator = args.ligands.rglob("*") if args.recursive else args.ligands.iterdir()
    ligands = sorted(p for p in iterator if p.is_file() and p.suffix.lower() == ".mol2")
    if not ligands:
        parser.error("no MOL2 files found; prepare SYBYL MOL2 ligands before running LeDock")
    if len({path.stem for path in ligands}) != len(ligands):
        parser.error("Ligand filename stems must be unique across the batch")
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(args.receptor, output / "receptor.pdb")
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update(executable=executable, ligand_files=[str(p) for p in ligands])
    (output / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")
    items = iter(ligands)
    failures, completed = 0, 0
    started = time.monotonic()
    fields = ["ligand", "output", "log", "status", "exit_code", "poses_saved", "wall_seconds", "error"]
    with (output / "results.csv").open("w", newline="") as csv_handle, \
            (output / "commands.jsonl").open("w") as commands, \
            ThreadPoolExecutor(max_workers=args.jobs) as pool:
        writer = csv.DictWriter(csv_handle, fieldnames=fields)
        writer.writeheader()
        # Bound the submission queue so large libraries do not create one future per ligand.
        pending = set()
        for _ in range(min(args.jobs, len(ligands))):
            ligand = next(items)
            pending.add(pool.submit(dock, ligand, ligand.relative_to(args.ligands), args, executable, output))
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                row = future.result()
                commands.write(json.dumps(dict(ligand=row["ligand"], command=row.pop("command"), cwd=row.pop("cwd"))) + "\n")
                commands.flush()
                writer.writerow(row)
                csv_handle.flush()
                completed += 1
                failures += row["status"] != "success"
                print(f"[{completed}/{len(ligands)}] {row['status']}: {row['ligand']}", flush=True)
                ligand = next(items, None)
                if ligand is not None:
                    pending.add(pool.submit(dock, ligand, ligand.relative_to(args.ligands), args, executable, output))
    summary = dict(files=len(ligands), successes=len(ligands) - failures,
                   failures=failures, elapsed_seconds=round(time.monotonic() - started, 3))
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
