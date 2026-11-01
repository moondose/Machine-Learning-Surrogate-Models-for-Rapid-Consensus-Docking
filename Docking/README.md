# Docking engines

Run the commands below from this directory. Ligand filename stems must be unique across a batch.

## GNINA batch docking

For classic LeDock, see [LeDock](#ledock-batch-docking).
For DiffDockHPC, see [DiffDock](#diffdock-batch-docking).

`gnina_batch.py` requires Python 3.8+ (standard library only) and a working GNINA
installation. It runs independently of MetaScreener. All inputs must already be
prepared for docking; the script does not clean receptors or generate ligand 3D
coordinates. PDB and PDBQT receptors are accepted and passed directly to GNINA.

## Native installation

```bash
python3 gnina_batch.py \
  --gnina /path/to/gnina \
  --receptor /path/to/receptor.pdbqt \
  --ligands /path/to/sdf_library \
  --recursive \
  --center 23.176 5.742 13.697 \
  --size 20 20 20 \
  --jobs 4 --cpus-per-job 2 \
  --output /path/to/new_gnina_results
```

Replace the example pocket coordinates with your binding pocket. `--size` specifies
the full box dimensions in Angstroms. Use `--receptor receptor.pdb` for PDB input.
By default, only the immediate ligand directory is scanned. Add `--recursive` to
scan subdirectories. Input extensions are matched without case sensitivity.

`--jobs` controls independent GNINA processes; `--cpus-per-job` controls GNINA CPU
threads per process. Allocate at least their product in CPU cores (8 in the
example). Under Slurm the script checks `SLURM_CPUS_PER_TASK`; otherwise it checks
the host logical CPU count. Memory must also support all concurrent processes.

## Apptainer or Singularity

```bash
python3 gnina_batch.py \
  --container /path/to/metascreener_22.04.simg \
  --gnina /path/to/MetaScreener/external_sw/gnina/gnina \
  --receptor /path/to/receptor.pdbqt \
  --ligands /path/to/sdf_library --recursive \
  --center 23.176 5.742 13.697 --size 20 20 20 \
  --jobs 4 --cpus-per-job 2 \
  --output /path/to/new_gnina_results
```

The container provides runtime dependencies; the path-based executable must exist
on the host. Alternatively use `--gnina gnina` if GNINA is on PATH inside the image.
Use `--container-runtime singularity` for Singularity. Receptor, ligand, output,
and host executable directories are bound automatically at the same absolute paths.
Container bind paths cannot contain commas or colons.

## Settings and output

Defaults: CPU mode (`--no_gpu`), exhaustiveness 8, at most 9 modes, seed 2026,
and CNN rescoring. Override with `--exhaustiveness`, `--num-modes`, `--seed`, or
`--cnn-scoring`. The same explicit seed is passed to every file; identical seeds
do not guarantee identical results across software versions or hardware.
For a single GPU use `--device gpu --gpu-id 0 --jobs 1`; container execution adds
`--nv`. CPU mode disables GPU use, not CNN scoring. GPU availability and runtime
errors are reported by GNINA in the per-file log. Add `--timeout SECONDS` to limit
each file's GNINA execution time.

The output directory must be new, preventing accidental reuse of stale results.
It contains:

- `poses/`: output SDF files, preserving ligand subdirectories and filenames.
- `logs/`: one combined standard output/error log per input file.
- `results.csv`: input/output paths, status, exit code, elapsed time, and errors.
- `commands.jsonl`: exact GNINA command arguments for each completed attempt.
- `run_config.json`: settings and ordered input file list.
- `summary.json`: file counts and total campaign elapsed time.

Each SDF file is one scheduling unit. Prefer one prepared molecule per file for
per-molecule status. Multi-record SDFs are passed intact to GNINA: success means
exit code zero and at least one complete output record, not confirmation that
every input molecule succeeded. Poses retain GNINA's SDF score properties and
native pose ordering; the script does not reinterpret scores or rank compounds.
Failed files do not stop other files. The script exits with status 1 if any file
fails. Run failures are diagnosed using `results.csv` and the corresponding log.

Upstream documentation: https://github.com/gnina/gnina

----------------------------------------------------------------------------------------------------------

## DiffDock batch docking

`diffdock_batch.py` wraps the DiffDock-L implementation in **Jnelen/DiffDockHPC**.
It targets the locally inspected interface at commit
`58d97d04f24a73b5117f6b83f2e074a8faa2dd50`. It is not a generic adapter for every
DiffDock fork. No upstream files are modified. The launcher requires Python 3.8+
and the standard library; inference requires the complete DiffDockHPC environment,
including PyTorch, PyYAML, RDKit, PyG, ESM, and its other dependencies.

## Required inputs

- A prepared PDB receptor. PDBQT is not supported by this wrapper.
- One complete molecule per SDF file, including the `$$$$` terminator. Multi-record
  SDF files are rejected to prevent silently using only their first molecule.
- A matching DiffDockHPC ESM embedding `.pt` file for the receptor's chain sequences.
- Local score and confidence model directories, each containing its checkpoint
  and `model_parameters.yml`. Model/checkpoint names come from the inference YAML.
- The DiffDockHPC installation and its inference YAML configuration.

Prepare the receptor and embedding together. Regenerate the embedding when chain
selection, sequence, or residue ordering changes. A matching file extension or
tensor dimension does not prove that an embedding belongs to the receptor.

This DiffDockHPC interface performs whole-protein inference and has no GNINA-like
binding-box constraint. Accordingly, this wrapper does not expose `--center` or
`--size`. Cropping a receptor or filtering poses by a pocket would be a separate,
explicit workflow change; the script does neither.

## Native inference environment

Activate the DiffDockHPC environment, then run:

```bash
python3 diffdock_batch.py \
  --diffdock-dir /path/to/DiffDockHPC \
  --receptor /path/to/receptor.pdb \
  --embedding /path/to/receptor_embedding.pt \
  --ligands /path/to/sdf_library --recursive \
  --device gpu --jobs 1 --cpus-per-job 2 \
  --samples 3 --batch-size 3 \
  --output /path/to/new_diffdock_results
```

Use `--python /path/to/environment/bin/python` if the launcher runs outside the
inference environment. Model directories default to the supplied inference YAML,
resolved relative to the DiffDockHPC installation. Override with `--model-dir`
and `--confidence-model-dir` if needed. Use `--config PATH` to select a different
YAML compatible with the installed code and weights. Required model files are
checked inside each worker before inference; automatic model downloading is not
used by this wrapper.

## DiffDockHPC container

```bash
python3 diffdock_batch.py \
  --diffdock-dir /path/to/DiffDockHPC \
  --container /path/to/DiffDockHPC/singularity/DiffDockHPC.sif \
  --receptor /path/to/receptor.pdb \
  --embedding /path/to/receptor_embedding.pt \
  --ligands /path/to/sdf_library --recursive \
  --device gpu --jobs 1 --cpus-per-job 2 \
  --samples 3 --batch-size 3 \
  --output /path/to/new_diffdock_results
```

The launcher uses `apptainer run` because the official DiffDockHPC image's
runscript selects its conda Python. GPU mode adds `--nv`. Use
`--container-runtime singularity` if appropriate. Required host paths are bound
at the same absolute paths. Bind paths cannot contain commas or colons. Model
directories outside the installation must be supplied using the directory
override options so the launcher binds them as well. This command assumes the
DiffDockHPC image's runscript; other images may require a different entry point.

## Parallelism

One worker loads the score and confidence models once and processes its ligand
shard sequentially. `--samples` controls predictions per ligand; `--batch-size`
controls the number of samples in a diffusion minibatch, not simultaneous ligands.

For two GPUs, each with its own model-loaded worker:

```bash
python3 diffdock_batch.py \
  --diffdock-dir /path/to/DiffDockHPC \
  --receptor /path/to/receptor.pdb \
  --embedding /path/to/receptor_embedding.pt \
  --ligands /path/to/sdf_library --recursive \
  --device gpu --gpu-ids 0 1 --jobs 2 --cpus-per-job 2 \
  --output /path/to/new_diffdock_results
```

When `CUDA_VISIBLE_DEVICES` is set (for example by Slurm), GPU identifiers are
selected from that allocation. Explicit `--gpu-ids` must use its identifiers,
including GPU UUIDs if applicable, rather than renumbering them. By default the
first `jobs` allocated GPUs are selected. Each worker sees only its assigned GPU.
GPU mode refuses CPU fallback if CUDA is unavailable.

For CPU execution use `--device cpu --jobs 4 --cpus-per-job 2`, allocating at least
8 CPUs and enough RAM for four model copies. CPU mode hides GPUs explicitly.
Under Slurm, CPU requirements are checked against `SLURM_CPUS_PER_TASK`; otherwise
the host logical CPU count is used. Run this launcher in one allocated task,
not one launcher per Slurm task. Prepare upstream SO(2)/SO(3) lookup caches once
using the installation's test procedure before launching concurrent workers.

## Embedding preparation

The installed upstream helper generates an embedding using ESM2 and writes it
under `data/protein_embeddings/`. Run it from the DiffDockHPC directory using its
inference environment:

```bash
cd /path/to/DiffDockHPC
mkdir -p data/protein_embeddings
python proteinEmbedding.py /path/to/receptor.pdb
```

For `receptor.pdb`, the expected output is `data/protein_embeddings/receptor.pt`.
ESM weights must be installed/cached or downloadable. Avoid reusing an old file
for a different receptor with the same basename. For the container, invoke this
helper with the same `apptainer run --nv` mechanism and bind the receptor directory
and DiffDockHPC installation. The batch wrapper requires the finished embedding;
it does not regenerate it or download ESM weights automatically.

## Output and validation

The output directory must be new. `--prepare-only` writes manifests and commands
without running inference or importing the inference dependencies. It validates
input paths and SDF record counts, but not model compatibility or GPU availability.
Prepared commands can be inspected in each shard's `command.json`; use a new
output directory for the actual run.

- `run_config.json`: original input paths, unique ligand IDs, shard mapping,
  settings, and receptor/embedding/inference/YAML SHA256 values.
- `inference_source.py` and `inference_args.yaml`: snapshots of the entry point and
  selected YAML. Imported utility modules and model weights remain external;
  preserve the recorded installation version and weights for reproducibility.
- `shard_XXXX/inputs.csv`: the exact input manifest sent to that worker.
- `shard_XXXX/poses/`: native `VS_DD_mol_XXXXXXXX_rankN_confidenceVALUE.sdf` files
  and any upstream preprocessing artifacts. Confidence is not a binding energy.
- Each shard also contains `worker_config.json`, `command.json`, `inference.log`,
  `effective_config.json` (when worker setup completes), and `process_status.json`.
- `results.csv`: one row per original ligand, with status, pose count, output
  directory, shard log, and errors. Unique complex IDs map saved poses back to original inputs; ligand filename stems must be unique.
- `summary.json`: ligand counts, worker failures, and total campaign elapsed time.

Success requires a complete SDF for every requested pose rank. Upstream inference
can skip ligands and still exit zero, so worker exit status alone is insufficient.
Partial predictions are retained but marked failed. A fatal worker error affects
its remaining shard; other workers continue. Completed ligand predictions may
remain successful even if their worker later fails; `worker_failures` and the
launcher's nonzero exit status report that incomplete process execution.
`--timeout SECONDS` limits an entire worker shard, including model loading, not
each ligand. A timed-out shard can therefore contain completed predictions.
Results are collected after all workers terminate; follow shard logs during runs.

The wrapper uses seed `2026 + shard_index` by default, seeds Python/NumPy/PyTorch,
and applies wrapper settings after loading YAML. Change `--seed` to override the
base value. Changing worker count changes RNG consumption and can change poses.
Identical seeds do not guarantee identical results across hardware or versions.

Upstream sources and licenses:

- https://github.com/Jnelen/DiffDockHPC
- https://github.com/gcorso/DiffDock

Code, models, containers, and their dependencies are obtained separately; retain
their applicable licenses and citations.

----------------------------------------------------------------------------------------------------------

## LeDock batch docking

`ledock_batch.py` requires Python 3.8+ (standard library only), Linux, and a working
classic LeDock executable. It runs independently of MetaScreener and does not
modify the input ligand library.

## Input preparation

Use a prepared PDB receptor, such as LePro's `pro.pdb`, with explicit hydrogens.
Classic LeDock reads receptor atoms from `ATOM` records; review any retained water,
metal ions, or cofactors according to the official manual. The script does not
prepare the receptor or change record types.

Each ligand file must contain one molecule in SYBYL MOL2 format, with appropriate
atom types, hydrogens, and 3D coordinates. SDF ligands must be prepared/converted
separately. The script checks the MOL2 record count but does not validate chemistry.
It does not accept PDBQT receptors or convert SDF files automatically.

## Run

```bash
python3 ledock_batch.py \
  --ledock /path/to/ledock_linux_x86 \
  --receptor /path/to/pro.pdb \
  --ligands /path/to/mol2_library \
  --recursive \
  --center 23.176 5.742 13.697 \
  --size 20 20 20 \
  --jobs 8 \
  --num-modes 20 \
  --rmsd 1.0 \
  --output /path/to/new_ledock_results
```

Replace the example paths and binding pocket coordinates. As in `gnina_batch.py`,
`--center` specifies the box center and `--size` its full dimensions in Angstroms.
The script converts these into LeDock bounds: `minimum = center - size / 2` and
`maximum = center + size / 2`. It generates `dock.in` automatically for each ligand.

`--jobs 8` runs up to eight independent LeDock processes, with one CPU per process.
There is no `--cpus-per-job` option for this classic LeDock workflow. Allocate at
least `jobs` CPU cores and sufficient memory. Under Slurm the script checks
`SLURM_CPUS_PER_TASK`; otherwise it checks the host logical CPU count. For example,
run it within an allocation with `--ntasks=1 --cpus-per-task=8`.

Without `--recursive`, only the immediate ligand directory is scanned. File
extensions are matched without case sensitivity. Add `--timeout SECONDS` to limit
each LeDock process. Defaults are one process, 20 requested poses, and 1.0 Angstrom
clustering RMSD. This wrapper does not supply a random seed because the existing
classic LeDock interface does not expose a documented seed setting.

## Output

The output directory must be new. It contains:

- `receptor.pdb`: a snapshot of the supplied prepared receptor.
- `work/<relative-input-path>/`: an isolated directory per ligand, preserving its
  subdirectory structure and full filename (including `.mol2`).
- Each work directory contains `ligand.mol2`, a `pro.pdb` symlink to the receptor
  snapshot, `dock.in`, `ligands.list`, `ledock.log`, and any generated `ligand.dok`.
- `results.csv`: original ligand paths, output/log paths, status, exit codes,
  saved pose counts, elapsed times, and errors.
- `commands.jsonl`: command arguments and working directories for each attempt.
- `run_config.json`: run settings, executable path, and ordered ligand list.
- `summary.json`: success/failure counts and campaign elapsed time.

Success means exit code zero and at least one `REMARK Cluster` record in the DOK
output; it does not assess pose quality. Invalid MOL2 record counts, timeouts, and
docking failures are recorded per ligand and do not stop the remaining work. The
script exits with status 1 if any ligand fails. Per-ligand elapsed time includes
input checks, copying, configuration writing, docking, and output inspection.
No existing results are reused, and no cross-ligand score ranking is performed.

LeDock is a separate third-party dependency. Download it from the official source;
the binary is not included with this script.

- Download: https://www.lephar.com/download
- User guide: https://www.lephar.com/download/LeDock.pdf

## rDock

`rdock_batch.py` runs prepared SD/SDF ligands with rDock. It follows the existing rDock Slurm settings: `dock.prm`, 10 docking runs per ligand and seed 123. Python 3.9+ and a working [rDock installation](https://rdock.github.io/) are required.

Prepare a system definition `.prm`, its matching cavity `.as`, and the receptor assets referenced by the PRM. Keep the files together, preserving relative paths. For a new receptor, configure the mapper and generate the cavity with `rbcavity -r receptor.prm -was` before docking. A reference-ligand mapper requires its reference ligand in the receptor coordinate frame. See the [rDock preparation guide](https://rdock.github.io/docking-in-3-steps/).

```bash
python rdock_batch.py \
  --rdock-root /path/to/rDock \
  --receptor-prm /path/to/prepared/1L2S.prm \
  --ligands /path/to/prepared/sdf \
  --output results/rdock \
  --recursive --jobs 8 --runs 10 --seed 123
```

`--protocol` selects a custom docking protocol; the default is `data/scripts/dock.prm` inside the rDock installation. `--timeout` sets a per-ligand time limit in seconds. Each input file must contain one complete SD molecule. Use a new output directory for each batch.

For Slurm, activate the intended Python environment and submit:

```bash
sbatch --cpus-per-task=8 --mem=16G --time=24:00:00 \
  --wrap="python rdock_batch.py --rdock-root /path/to/rDock --receptor-prm /path/to/prepared/1L2S.prm --ligands /path/to/sdf --output results/rdock --recursive --jobs 8"
```

The script sets `RBT_ROOT`, adds the installation's `lib` directory to `LD_LIBRARY_PATH`, and runs one rbdock process per worker. Each ligand has an isolated directory under `work/<relative ligand path>/` containing `poses.sd` and `rdock.log`. Batch outputs are `results.csv`, `summary.json`, `run_config.json`, `commands.jsonl` and a snapshot of the receptor PRM. The SD output contains rDock scores and poses in engine output order. `--runs` specifies docking attempts per ligand.
