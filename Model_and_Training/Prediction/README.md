# Ensemble prediction

`predict.py` applies a saved MBE or percentile-consensus ensemble to Morgan + RDKit 2D/3D descriptors. Model type and fold count are detected from the saved artifacts.

## Requirements

Python 3.9+, PyTorch 2.3 or later, NumPy, pandas, SciPy and scikit-learn. Use the training environment. The script imports the model definitions from `../Training/model_utils.py`.

## Generate descriptors

Use the same feature schema as the model. From this directory:

```bash
python ../Descriptor_Generation/generate_descriptors.py \
  --input /path/to/prepared/sdf \
  --feature-names ../Models/mbe/feature_names.csv \
  --output-dir /path/to/descriptors \
  --workers 8
```

The descriptor directory must contain `X_features.npy`, `feature_names.csv` and `zinc_ids_aligned.csv`. The predictor checks the exact feature-name order and ID row count. Docking scores and training target labels are not required for inference.

## Predict MBE

```bash
python predict.py \
  --descriptor-dir /path/to/descriptors \
  --model-dir ../Models/mbe \
  --output /path/to/predictions/mbe.csv \
  --batch-size 4096
```

The mean predicts `(GNINA minimizedAffinity + LeDock score) / 2` in the original target units.

## Predict percentile consensus

```bash
python predict.py \
  --descriptor-dir /path/to/descriptors \
  --model-dir ../Models/percentile_consensus \
  --output /path/to/predictions/percentile.csv \
  --batch-size 4096
```

The packaged percentile ensemble was trained on the mean mid-rank percentile of GNINA, LeDock and DOCK3.7. A newly trained percentile ensemble uses the target supplied to its trainer.

## Processing and output

Each fold uses its saved feature scaler, model weights and target scaler. Predictions are converted back to original target units before averaging. The saved scalers are used directly.

The output CSV preserves the source ID columns and descriptor row order, adding:

| Column | Meaning |
| --- | --- |
| `predicted_mbe_mean` or `predicted_percentile_consensus_mean` | Mean across all fold predictions |
| Corresponding `_std` column | Population standard deviation across folds |
| `ensemble_folds` | Number of contributing models |
| `prediction_rank` | Ascending mean prediction rank; tied values use average rank |
| `row_index` | Original descriptor matrix row |

Lower predictions rank better for both targets. Add `--sort` to sort the CSV by predicted mean. Add `--save-fold-predictions` to save an N-by-folds NumPy array; its rows always follow the original descriptor order, including when the CSV is sorted.

The script also saves `<output>.run.json` with configuration, target definition, hashes and package versions, and `<output>.fold_runtime.csv` with fold timings. Choose a new output filename for every run.

`--device auto` uses CUDA when available and CPU otherwise. Use `--device cpu` or `--device cuda:0` to choose explicitly. Inference uses FP32. `--batch-size` controls working memory; `--input-cache ram` optionally caches the cleaned feature matrix in RAM. The default reads memory-mapped input in batches. `--max-rows 10` runs a small prediction subset.
