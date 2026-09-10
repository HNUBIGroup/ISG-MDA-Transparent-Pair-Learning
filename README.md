# ISG-MDA

This repository provides the implementation, training, evaluation, and inference workflow for ISG-MDA, a multimodal model for miRNA–drug association prediction.

ISG-MDA combines two views of each entity. A convolutional sequence encoder and a frozen RNA-FM backbone encode miRNAs, while a graph attention network and a frozen ChemBERTa backbone encode drugs. Bidirectional cross-attention fuses the topology and semantic views for each entity. The prediction head represents each miRNA–drug pair as the concatenation of the two entity vectors, their absolute difference, and their signed elementwise product, followed by a multilayer classifier.

No research data, trained checkpoints, pretrained-model weights, predictions, or experiment results are distributed in this repository.

## Installation

Python 3.9–3.11 is supported. Create an isolated environment, install a PyTorch build appropriate for your CPU or CUDA platform, and then install this package:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Install the appropriate PyTorch and PyTorch Geometric builds first.
python -m pip install -e .
```

The dependency list is also available in `requirements.txt`. The semantic encoders require RNA-FM and the ChemBERTa model named in `configs/default.yaml`. They may be obtained by their libraries on first use or supplied from an existing local cache. No pretrained weights are bundled here.

## Data preparation

Prepare drug, miRNA, and pair tables using the schema in [`data/README.md`](data/README.md). The repository does not provide these tables. To create a deterministic split from user-provided positive and control pairs:

```bash
export PROJECT_ROOT="$PWD"
export DATA_ROOT="${PROJECT_ROOT}/data"
python scripts/preprocessing/prepare_data.py \
  --positive-pairs "${DATA_ROOT}/positive_pairs.tsv" \
  --negative-pairs "${DATA_ROOT}/negative_pairs.tsv" \
  --output-dir "${DATA_ROOT}/splits" \
  --split-mode random \
  --seed 0
```

Available split modes are `random`, `mirna_cold`, `drug_cold`, and `strict_cold`. The preprocessing command audits pair overlap and the required cold-entity separation before writing its outputs.

## Training

Set `DATA_ROOT` or pass all input paths explicitly. The default configuration uses Adam with a learning rate and weight decay of `0.0005`, batch size `32`, and `80` fixed epochs.

```bash
python scripts/training/train.py \
  --config configs/default.yaml \
  --train-pairs "${DATA_ROOT}/splits/train.tsv" \
  --drug-table "${DATA_ROOT}/drugs.tsv" \
  --mirna-table "${DATA_ROOT}/mirnas.tsv" \
  --output-dir outputs/training \
  --device cuda
```

Use `--device cpu` when appropriate. Use `--local-files-only` to require the configured ChemBERTa asset to be available locally. The example Slurm script at `scripts/slurm/train_example.sbatch` uses only generic environment variables.

## Evaluation

```bash
python scripts/evaluation/evaluate.py \
  --config configs/default.yaml \
  --checkpoint outputs/training/model_final_epoch.pt \
  --pairs "${DATA_ROOT}/splits/test.tsv" \
  --drug-table "${DATA_ROOT}/drugs.tsv" \
  --mirna-table "${DATA_ROOT}/mirnas.tsv" \
  --output-dir outputs/evaluation \
  --device cpu
```

The command writes per-pair probabilities and binary classification metrics under the selected output directory.

## Inference

The inference pair table contains `mirna_index` and `drug_index`; labels are not required.

```bash
python scripts/inference/predict.py \
  --config configs/default.yaml \
  --checkpoint outputs/training/model_final_epoch.pt \
  --pairs "${DATA_ROOT}/pairs_to_score.tsv" \
  --drug-table "${DATA_ROOT}/drugs.tsv" \
  --mirna-table "${DATA_ROOT}/mirnas.tsv" \
  --output outputs/inference/predictions.tsv \
  --device cpu
```

Evaluation metric JSON files can be visualized with:

```bash
python scripts/visualization/plot_metrics.py \
  outputs/evaluation/metrics.json \
  --output outputs/figures/metrics.png
```

## Configuration

`configs/default.yaml` contains five sections:

- `model`: encoder names, input and hidden dimensions, fusion settings, pair representation, and prediction head;
- `training`: loss, optimizer, learning rate, weight decay, batch size, epoch policy, seed, workers, and default device;
- `data`: table paths, identifier/feature columns, and pair-label schema;
- `output`: default generated-output root;
- `metrics`: classification threshold, calibration bins, and probability clipping.

Environment variables such as `${DATA_ROOT}` are expanded when the configuration is loaded. Generated outputs belong under `outputs/`, `checkpoints/`, `predictions/`, `results/`, `logs/`, or `cache/`; these locations are excluded by `.gitignore`.

## Tests

Run the unit tests and the synthetic CPU smoke test from the repository root:

```bash
python -m unittest discover -s tests -v
python scripts/run_smoke_test.py --config configs/default.yaml
```

The smoke test uses small synthetic strings and molecular graphs. It checks tensor dimensions, one forward/backward pass, finite loss and gradients, and metric plumbing. It does not produce a scientific result and does not download semantic-backbone weights.

## Citation

Citation metadata are provided in [`CITATION.cff`](CITATION.cff). The source code is released under the [MIT License](LICENSE).
