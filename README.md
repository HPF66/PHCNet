# PHCNet: peptide half-life regression

This repository contains the locked PHCNet final workflow for peptide half-life
regression. It trains ESM2-8M, ESM2-650M, and ProtBERT under the same fixed
10-fold protocol, calibrates each seed on inner validation data, averages three
seeds per backbone, and combines the three backbones with fixed equal weights.

## Protocol

- Dataset: 1,556 exact labels and 655 mean+SD auxiliary labels (2,211 records).
- Target: `log10(half-life in minutes)`.
- Evaluation: locked record-level random 10-fold outer cross-validation.
- Outer test sets contain only the 1,556 exact labels.
- Mean+SD rows are auxiliary training data only. They never enter inner
  validation or outer test sets.
- Target normalization, early stopping, affine calibration, pairwise loss, and
  calibration loss use exact labels only.
- All ten folds use the same locked hyperparameters.
- Seeds: `3407`, `2026`, and `777`.
- Official ensemble: equally weighted calibrated log10 predictions
  (`1/3` ESM2-8M, `1/3` ESM2-650M, `1/3` ProtBERT). No outer-test labels
  are used to fit these official weights.

The configuration retains exploratory OOF-optimized weights
(`0.15/0.30/0.55`) for comparison only. They must not be reported as an
independent evaluation result.

## Repository layout

- `configs/final_model_config.json`: locked protocol, architecture, optimizer,
  and official ensemble weights.
- `data/peplife2_exact_plus_mean_sd.csv`: training records.
- `splits_10fold_exact_plus_mean_sd_train/`: locked train/test splits.
- `runtime/`: model implementations, trainers, calibration, seed ensembling,
  and error analysis.
- `run_final_model.py`: Python entry point.
- `run_final_model.sh`: shell entry point for Linux GPU servers.

No Hugging Face model weights or token caches are included. The 650M ESM2 and
ProtBERT caches are built automatically under `HF_HOME/phcnet_token_cache`
when absent.

## Environment

The workflow was tested on Linux with:

```text
Python 3.10
PyTorch 2.5.1+cu121
transformers 4.46.3
peft 0.13.2
```

Install a PyTorch build compatible with the target CUDA driver first, then
install the remaining dependencies:

```bash
python -m pip install -r requirements.txt
```

Do not replace an existing CUDA-compatible PyTorch installation with the
generic requirements command.

## Quick validation

Clone this repository, enter its root directory, activate the prepared Python
environment, and run:

```bash
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false

python -u run_final_model.py --validate-only
```

By default, Hugging Face uses its official endpoint. Users with a locally
approved mirror can set `HF_ENDPOINT` themselves before running.

## Smoke run

This runs one split, one seed, and two epochs per backbone:

```bash
mkdir -p result/final_three_backbone_smoke

RESULT_DIR=result/final_three_backbone_smoke \
nohup bash run_final_model.sh --smoke \
  > result/final_three_backbone_smoke/nohup.log 2>&1 &

tail -f result/final_three_backbone_smoke/nohup.log
```

## Full run

This launches 90 independent training jobs:

```text
3 backbones x 10 folds x 3 seeds = 90 runs
```

Example using four physical GPUs:

```bash
mkdir -p result/final_three_backbone

RESULT_DIR=result/final_three_backbone \
nohup bash run_final_model.sh --gpu-ids 0 1 2 3 \
  > result/final_three_backbone/nohup.log 2>&1 &

tail -f result/final_three_backbone/nohup.log
```

The multi-GPU mode uses task-level parallelism. Each trainer process sees one
GPU, and the per-model batch size, learning rate, and architecture stay fixed.
Completed split/seed runs are detected and skipped on restart.

## Outputs

```text
result/final_three_backbone/training_logs/<backbone>/
result/final_three_backbone/ensemble/pooled_metrics.csv
result/final_three_backbone/ensemble/fold_metrics.csv
result/final_three_backbone/ensemble/all_test_predictions.csv
result/final_three_backbone/ensemble/error_analysis/
```

`pooled_metrics.csv` contains RMSE, MAE, R2, Pearson, Spearman, accuracy,
precision, recall, F1, and AUC for each backbone and the official ensemble.

## Data and reporting

The records are derived from the PEPlife2 database:

> Alam U, Chaudhary K, Kumar N, Tomer R, Patiyal S, Raghava GPS.
> PEPlife2: An Updated Repository of the Half-Life of Peptides and Proteins.
> *Immuno*. 2026;6(2):26.
> [https://doi.org/10.3390/immuno6020026](https://doi.org/10.3390/immuno6020026)

- Database: [https://webs.iiitd.edu.in/raghava/peplife2/](https://webs.iiitd.edu.in/raghava/peplife2/)
- Upstream API records retrieved: 4,500 natural/modified entries.
- Redistribution: this repository maintainer has confirmed that processed
  PEPlife2 records and the derived split files may be redistributed. The
  PEPlife2 citation and source attribution must be retained.

### Processing pipeline

1. Sequence recovery and normalization:
   - Started from 4,500 PEPlife2 natural/modified API records.
   - Filled 230 missing modified sequences from consistent duplicate-PDB
     mappings and 2,414 from the reported original sequence.
   - Records still lacking a usable modified sequence were excluded.
   - No sequence-length filter was used during dataset construction.
2. Condition filtering:
   - Species: human, monkey, mouse, or rat.
   - Matrix: serum, plasma, or blood.
   - Experimental setting: in vivo or in vitro.
   - Experimental, terminal-modification, chemical-modification, cyclicity,
     chirality, assay, concentration, protease, and source metadata were
     retained.
3. Exact labels:
   - 2,929 source exact-label records were screened.
   - 1,556 records passed the sequence and condition criteria.
4. Mean+SD labels:
   - 905 uncertain records passed the initial condition screen.
   - 693 contained a parseable positive mean and SD.
   - Duplicate records were merged using PMID, modified sequence, species,
     matrix, setting, sample, assay, mean, and SD.
   - Exclusions after parsing/deduplication: 27 condition-level overlaps with
     exact labels, 2 records with SD >= mean, and 1 invalid/nonpositive record.
   - 655 mean+SD records were retained as auxiliary training data.
5. Uncertainty conversion:
   - The target is `log10(half-life in minutes)`.
   - Label uncertainty is approximated by
     `sigma_log10 = SD_minutes / (mean_minutes * ln(10))`.
   - The mean+SD supervised weight is
     `0.1 * 0.1^2 / (0.1^2 + sigma_log10^2)`.
   - Mean+SD records never enter inner validation, target-normalization
     statistics, pairwise/calibration losses, or outer test sets.
6. Locked cross-validation:
   - The 1,556 exact records use the locked target-stratified random 10-fold
     assignment generated with seed 3407.
   - Each exact record appears in one outer test fold.
   - All 655 mean+SD records are appended only to the corresponding outer
     training set.
   - The same hyperparameters and seeds are used across all ten folds.

Current results are record-level random-CV out-of-fold estimates. They do not
establish unseen-sequence, unseen-study, or external-cohort generalization.

See [DATA_NOTICE.md](DATA_NOTICE.md) and
`splits_10fold_exact_plus_mean_sd_train/summary.json` for the packaged data
provenance and split statistics.
