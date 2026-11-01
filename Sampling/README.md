# Molecular sampling

This directory provides two strategies for selecting molecules from a CSV dataset.

1) Random sampling, `random_sampling.py` , uniform sampling without replacement using reservoir sampling, outputs sampled CSV
2) Diverse sampling `submo_diverse_sampling.py`, SUBMO GNN descriptors followed by global greedy log-determinant maximization, outputs Embeddings, aligned metadata and selected CSV

## Input dataset
Use your own CSV or compressed CSV. Random sampling preserves every input column and requires a header. Diverse sampling requires a SMILES column; specify its name with `--smiles-col`. Keep a molecule identifier such as `zincid` to trace selected molecules back to the dataset.

```csv
zincid,smiles,dockscore
ZINC000000388164,Nc1cc(Br)ccc1Br,-17.45
```

The dataset can be stored anywhere. Pass its path through `--input`; neither script downloads the dataset. The examples below use an AmpC screening table named `AmpC_screen_table.csv.gz`.

## Random sampling

### Requirements

Python 3.9 or later. The script uses only the Python standard library. No SUBMO files or model checkpoint are needed.

### Run

From this directory:

```bash
python random_sampling.py \
  --input /path/to/AmpC_screen_table.csv.gz \
  --output results/ampc_random_500k.csv \
  --n-select 500000 \
  --seed 42
```

Use `--sep` for a different input delimiter and `--encoding` for a different text encoding. Supported input formats are plain text, `.gz`, `.bz2` and `.xz`. The output is an uncompressed comma-separated CSV containing the original header and sampled rows.

The script reads the dataset once and stores only the requested sample in memory. A fixed seed reproduces the selection for the same dataset in the same row order. If fewer rows exist than requested, the script reports the available count. Choose a new output filename for each run.

## SUBMO diverse sampling

### Download the upstream source

Get the original implementation from [tomotomonakanaka/SUBMO](https://github.com/tomotomonakanaka/SUBMO):

```bash
git clone https://github.com/tomotomonakanaka/SUBMO.git /path/to/SUBMO
```

The integrated script imports `GNNMultitask` from the upstream `model/gnn_models.py`. Keep the cloned repository available on `PYTHONPATH` during execution. Worker processes inherit this setting.

```bash
export PYTHONPATH="/path/to/SUBMO${PYTHONPATH:+:$PYTHONPATH}"
```

`GNNMultitask.forward(task=None)` returns the descriptors directly, so the upstream model file does not need the locally added `get_graph_embedding()` method. The upstream `utils/algorithms.py` is used by the smoke-test comparison; the integrated script implements the equivalent global logdet update itself.

### Obtain a QM9 checkpoint

Supply a full-model SUBMO `GNNMultitask` checkpoint trained on QM9. For the AttentiveFP workflow used in this project, the local checkpoint is:

```text
save_pickle/qm9_normalize_relu_1_attentivefp_model.p
```

Place your checkpoint in `Sampling/save_pickle/` or pass an absolute path through `--model`. The script reads the model architecture, ReLU and normalization settings from that file.

The upstream repository README provides source and training commands but no direct pretrained-checkpoint download link. To obtain the authors' weights, request the QM9 checkpoint from the authors through the [upstream repository](https://github.com/tomotomonakanaka/SUBMO). To generate a QM9 checkpoint yourself, use the upstream `run_experiment.py` and the settings in `experiment/run_experiment.sh`; the repository includes `data/qm9.csv`, and training writes models to `save_pickle/`.

Training is performed on QM9. Applying an existing QM9 checkpoint to AmpC requires no AmpC labels or retraining. Use the checkpoint corresponding to the experiment you want to reproduce.

The script supports upstream AttentiveFP and GCN full-model checkpoints using canonical atom/bond features. AttentiveFP uses atom and bond features with an attention readout; the GCN branch uses atom features and weighted-sum/max readout. Model type is detected automatically. Load checkpoints obtained from a trusted source, since full-model PyTorch files use Python pickle.

### Install dependencies

Create an isolated environment:

```bash
conda create -n submo_sampling python=3.9 pip -y
conda activate submo_sampling
python -m pip install \
  torch==2.5.1 dgl==1.1.3 dgllife==0.3.2 \
  rdkit==2025.3.5 numpy==2.0.2 pandas threadpoolctl
```

These core package versions were used for the CPU AttentiveFP smoke test. The upstream README lists the versions used in the original paper separately. For CUDA inference, install a mutually compatible CUDA build of PyTorch and DGL, then select the GPU with `--device cuda`.

### Run

```bash
conda activate submo_sampling
export PYTHONPATH="/path/to/SUBMO${PYTHONPATH:+:$PYTHONPATH}"

python submo_diverse_sampling.py \
  --input /path/to/AmpC_screen_table.csv.gz \
  --smiles-col smiles \
  --model save_pickle/qm9_normalize_relu_1_attentivefp_model.p \
  --output-dir results/ampc_submo_500k \
  --n-select 500000 \
  --workers 8 \
  --chunk-size 10000 \
  --batch-size 256 \
  --device cpu
```

Use `--fraction 0.01` instead of `--n-select` to select 1% of valid molecules. Set `--sep` for a different input delimiter. Use a new output directory for each run.

`--workers` controls parallel graph construction and candidate score updates. `--chunk-size` controls CSV ingestion, `--batch-size` controls inference batches, and `--score-block-size` controls blocks used for candidate score updates. Graph construction and scoring run in parallel; each greedy choice follows the preceding choice.

The objective is `log det(I + X_S X_Sᵀ)`. Every valid molecule remains eligible at each selection step. The implementation uses a mathematically equivalent feature-space inverse update to avoid storing the upstream N-by-k coefficient matrix. Runtime grows with the number of molecules, embedding dimensions and selected molecules; use a small run to estimate the cost before screening the full dataset.

### Outputs

| File | Contents |
| --- | --- |
| `embeddings.npy` | Embeddings for all valid molecules |
| `metadata.csv` | Input columns plus `source_row` and `embedding_row` |
| `selected.csv` | Selected molecules in greedy order, with `selection_rank` |
| `selected_embeddings.npy` | Embeddings in the same order as `selected.csv` |
| `rejected.csv` | Failed source rows and graph-construction errors |
| `run.json` | Arguments, package versions, model SHA-256 and row counts |

Row indices and selection ranks start at zero. Row `i` of `embeddings.npy` corresponds to row `i` of `metadata.csv`. Invalid molecules are recorded in `rejected.csv` and excluded from selection.

Load the embeddings with NumPy:

```python
import numpy as np
import pandas as pd

embeddings = np.load('results/ampc_submo_500k/embeddings.npy', mmap_mode='r')
selected = pd.read_csv('results/ampc_submo_500k/selected.csv')
selected_embeddings = np.load('results/ampc_submo_500k/selected_embeddings.npy')
```

## Source and citation

SUBMO source: [tomotomonakanaka/SUBMO](https://github.com/tomotomonakanaka/SUBMO).

Paper: [Selecting molecules with diverse structures and properties by maximizing submodular functions of descriptors learned with graph neural networks](https://www.nature.com/articles/s41598-022-04967-9).

SUBMO is distributed under the MIT license. Preserve the upstream copyright notice and `LICENSE` when redistributing its source files. This directory uses the upstream code as an external dependency through `PYTHONPATH`.
