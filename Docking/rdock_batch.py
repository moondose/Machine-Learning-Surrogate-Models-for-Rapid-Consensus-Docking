#!/usr/bin/env python3
"""Run parallel rDock docking with a prepared receptor and cavity.
"""

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import csv
import json
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
    """Count complete SD records in an input or docking result file."""
    with path.open(errors="replace") as handle:
        return sum(line.strip() == "$$$$" for line in handle)


def dock(ligand, relative, args, executable, output):
    # Isolate every ligand while preserving its relative source path.
    work = output / "work" / relative
    pose = work / "poses.sd"
    log = work / "rdock.log"
    command = [executable, "-i", "ligand.sdf", "-o", "poses",
               "-r", args.receptor_prm.name, "-p", str(args.protocol),
               "-n", str(args.runs), "-s", str(args.seed)]
    started = time.monotonic()
    code, poses, error = None, 0, ""
    try:
        work.mkdir(parents=True, exist_ok=False)
        if count_molecules(ligand) != 1:
            raise ValueError("Expected exactly one complete molecule per SD/SDF file")
        shutil.copyfile(ligand, work / "ligand.sdf")
        # Relative receptor paths in the PRM are resolved from the working directory.
        # Link the prepared assets; rbdock reads them without modifying them.
        for asset in args.receptor_prm.parent.iterdir():
            if asset.name not in {".git", "__pycache__"} and asset.resolve() != output and not (work / asset.name).exists():
                (work / asset.name).symlink_to(asset.resolve())
        env = os.environ.copy()
        env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                   RBT_ROOT=str(args.rdock_root))
        env["LD_LIBRARY_PATH"] = str(args.rdock_root / "lib") + os.pathsep + env.get("LD_LIBRARY_PATH", "")
        with log.open("w") as handle:
            result = subprocess.run(command, cwd=work, stdout=handle,
                                    stderr=subprocess.STDOUT, env=env, timeout=args.timeout)
        code = result.returncode
        if pose.is_file():
            poses = count_molecules(pose)
        if code != 0 or poses == 0:
            error = "rDock failed or produced no complete SD poses; inspect the log"
    except subprocess.TimeoutExpired:
        error = "rDock exceeded the per-ligand timeout"
    except (OSError, ValueError) as exc:
        error = str(exc)
    success = code == 0 and poses > 0 and not error
    return dict(ligand=str(ligand), output=str(pose), log=str(log),
                status="success" if success else "failed", exit_code=code,
                poses_saved=poses, wall_seconds=round(time.monotonic() - started, 3),
                error=error, command=command, cwd=str(work))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receptor-prm", required=True, type=Path,
                        help="Prepared system PRM; matching .as and receptor assets alongside it")
    parser.add_argument("--rdock-root", required=True, type=Path,
                        help="rDock installation containing bin/, lib/ and data/")
    parser.add_argument("--protocol", type=Path, help="Docking protocol (default: RBT_ROOT/data/scripts/dock.prm)")
    parser.add_argument("--ligands", required=True, type=Path, help="Directory of single-molecule SD/SDF files")
    parser.add_argument("--output", required=True, type=Path, help="New output directory")
    parser.add_argument("--recursive", action="store_true", help="Include ligand subdirectories")
    parser.add_argument("--jobs", type=positive_int, default=1, help="Concurrent single-CPU rbdock processes")
    parser.add_argument("--runs", type=positive_int, default=10, help="Docking runs per ligand (default: 10)")
    parser.add_argument("--seed", type=int, default=123, help="rDock random seed (default: 123)")
    parser.add_argument("--timeout", type=positive_int, help="Maximum seconds per ligand")
    args = parser.parse_args()
    args.receptor_prm = args.receptor_prm.resolve()
    args.rdock_root = args.rdock_root.resolve()
    args.protocol = (args.protocol or args.rdock_root / "data/scripts/dock.prm").resolve()
    args.ligands = args.ligands.resolve()
    output = args.output.resolve()
    if not args.receptor_prm.is_file():
        parser.error("--receptor-prm must be an existing system definition file")
    cavity = args.receptor_prm.with_suffix(".as")
    if not cavity.is_file() or cavity.stat().st_size == 0:
        parser.error("Prepare the matching cavity first: rbcavity -r receptor.prm -was")
    if not args.protocol.is_file():
        parser.error("Docking protocol file not found")
    if not args.ligands.is_dir():
        parser.error("--ligands must be an existing directory")
    if output.exists():
        parser.error("--output already exists; choose a new directory")
    if not 0 <= args.seed <= 2147483647:
        parser.error("--seed must be in [0, 2147483647]")
    executable = args.rdock_root / "bin/rbdock"
    if not executable.is_file() or not os.access(executable, os.X_OK):
        parser.error("rbdock executable not found under --rdock-root/bin")
    executable = str(executable)
    allocated = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    if args.jobs > allocated:
        parser.error(f"requested {args.jobs} processes but only {allocated} CPUs are available")
    iterator = args.ligands.rglob("*") if args.recursive else args.ligands.iterdir()
    ligands = sorted(p for p in iterator if p.is_file() and p.suffix.lower() in {".sdf", ".sd"})
    if not ligands:
        parser.error("no SD/SDF ligand files found")
    if len({path.stem for path in ligands}) != len(ligands):
        parser.error("Ligand filename stems must be unique across the batch")
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(args.receptor_prm, output / "receptor.prm.snapshot")
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
