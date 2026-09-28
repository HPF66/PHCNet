# PHCNet

PHCNet predicts peptide and protein half-life in log10(minutes) from sequence,
experimental context, chemical modification information, and deterministic
sequence physicochemical features.

The locked final ensemble is:

    ESM2-8M + ESM2-650M + ProtBERT

The three protein language models are frozen feature extractors. No LoRA or
other backbone fine-tuning is used. Each PLM is processed by a separate PHCNet
configuration with the same adapter structure and full PHCNet branch topology.
Adapter and PHCNet parameters are trained independently and are not shared
between PLMs. Residual settings are locked separately for each backbone/fold.

## Layout

    data/                 Dataset and locked 10-fold splits
    feature/              Cache manifest and local feature-cache directory
    model/                Unified PHCNet and residual branches
    my_loader/            CSV normalization, vocabularies, and PyTorch loaders
    scratch/get_feature/  Reproducible PLM feature-cache construction
    train/train.py        Locked training, calibration, seed ensemble, and evaluation
    utils/                Locked final configuration

## Data And Evaluation Protocol

The dataset has 2,211 records: 1,556 exact labels and 655 mean +/- SD labels.
Targets are log10(half_life_minutes). Mean +/- SD records are auxiliary
training rows with:

    lambda_aux = mean_sd_loss_weight = 0.2
    tau = mean_sd_noise_floor_log10 = 0.1
    mean_sd_weight_mode = sd_adaptive
    reliability = lambda_aux * tau^2 / (tau^2 + sigma_log10^2)

Mean +/- SD rows never enter inner validation, target normalization, pairwise
ranking loss, calibration loss, or the outer test set. The released model uses
30 locked residual configurations: three backbones x ten outer folds, with
three seeds per configuration. These settings were selected using the mean
inner-validation RMSE across seeds, then locked before outer-test prediction.
`utils/final_config.json` records every retained configuration and its source
selection-lock SHA256. Reproduction reads these settings directly; no search
or ablation is performed. Backbone and seed ensemble weights remain equal.

## Model

All three PLMs use exactly the same long-sequence protocol. Every sequence is
split into 480-residue chunks with 64-residue overlap. Embeddings for residues
that occur in two adjacent chunks are merged by position-aligned averaging;
the stitched full-sequence tensor is then passed to the trainable adapter and
pooling layer. No sequence is truncated, including sequences longer than 512
residues. The internal value 512 is only the tokenizer capacity for one chunk
plus model-specific boundary tokens.

All three PLMs use the same adapter architecture, with independently learned
parameters:

    ESM2-8M: 320 -> LayerNorm -> Linear -> GELU -> 128
    ESM2-650M: 1280 -> LayerNorm -> Linear -> GELU -> 128
    ProtBERT: 1024 -> LayerNorm -> Linear -> GELU -> 128
    128 residue features -> masked mean plus masked max -> Linear -> 128

The pooled 128-dimensional vector is projected to a 256-dimensional sequence
representation and fused with categorical experimental and modification
embeddings. PHCNet then adds:

1. A bounded context-conditioned residual.
2. A bounded modification-aware text-CNN residual.
3. A bounded global physicochemical residual.

The physicochemical branch uses 432 deterministic features and maps
432 -> H_physchem -> 256, where H_physchem is 64 or 128 in the retained
configurations. The context residual width, residual bounds, dropout, and
physicochemical gate temperature also follow the locked backbone/fold
settings. The modification branch uses character embeddings of width 48,
32 channels per CNN kernel (5, 9, and 13), and a 128-dimensional output.

Each backbone is trained over 10 folds x 3 seeds. Seed predictions are affine
calibrated and averaged. The backbone ensemble uses fixed equal weights in
calibrated log10 space:

    ESM2-8M   = 1/3
    ESM2-650M = 1/3
    ProtBERT  = 1/3

The final three-backbone prediction is the equal-weight mean of calibrated
log10 predictions.

The published reference uses 1,556 pooled outer-test records. Its calibrated
three-seed, three-backbone ensemble has RMSE 0.532364, MAE 0.359182,
R2 0.718636, and Pearson r 0.847783. The mean and sample standard deviation
of the ten fold-specific RMSEs are 0.530921 +/- 0.040433. The uncalibrated
ensemble RMSE (0.531690) is a different output of the same trained models.
These are exploratory internal results on a previously examined cohort.
Auxiliary lambda/tau sensitivity analyses also used that cohort.

The source model and training hashes, selection-lock hash, and reference
metrics are in `utils/final_config.json`. Retraining can vary with hardware,
PyTorch/CUDA versions, and regenerated frozen embeddings. Reference metrics
are provenance, not a replacement for evaluating the outputs of a new run.

## Feature Caches

Precomputed `.pt` feature caches are intentionally not included in this
repository. They depend on the curated PHCNet sequences and are generated
locally from these public Hugging Face checkpoints:

    ESM2-8M:   facebook/esm2_t6_8M_UR50D
    ESM2-650M: facebook/esm2_t33_650M_UR50D
    ProtBERT:  Rostlab/prot_bert

The feature builder downloads the checkpoints automatically on first use and
stores them in the Hugging Face cache. Set `HF_HOME` if the default cache
location is unsuitable:

    export HF_HOME="$HOME/huggingface_cache"

Then generate all three PHCNet feature caches. The 480-residue chunk size and
64-residue overlap are locked defaults in the builder:

    python -m scratch.get_feature.build_features \
      --split-root data/splits_10fold_exact_plus_mean_sd_train \
      --models esm8m esm650m protbert \
      --output-dir feature \
      --batch-size 8 \
      --device cuda

To download the source checkpoints before feature generation, for example on
a login node with internet access, run:

    python -c "from huggingface_hub import snapshot_download; [snapshot_download(model) for model in ('facebook/esm2_t6_8M_UR50D', 'facebook/esm2_t33_650M_UR50D', 'Rostlab/prot_bert')]"

The command produces the three `.pt` files listed in
`feature/cache_manifest.json`. A batch size of 8 was used for the released
protocol; reduce `--batch-size` if GPU memory is limited. Do not commit the
generated `.pt` files: they are excluded by `.gitignore`.

## Setup And Reproduction

Use Python 3.10 or later. Install a CUDA-compatible PyTorch build first, then
install the remaining dependencies:

    python -m pip install -r requirements.txt

On a machine with internet access, the normal training command automatically
downloads any missing source checkpoint and builds any missing feature cache.
For an offline machine, run the checkpoint pre-download and feature-builder
commands above before enabling offline mode.

Validate the data package, fixed splits, and configuration:

    python train/train.py --validate-only

After cache generation, validate the three local feature files against their
size and SHA256 values in `feature/cache_manifest.json`:

    python train/train.py --validate-only --verify-feature-caches

## Run

Validate data, locked splits, model names, and final protocol:

    python train/train.py --validate-only

Train all three backbones, building any missing cache when needed:

    python train/train.py \
      --result-dir result/phcnet_locked_20260923

Run independent jobs across four physical GPUs:

    python train/train.py \
      --gpu-ids 0 1 2 3 \
      --result-dir result/phcnet_locked_20260923

Each completed run saves its effective configuration, trainable weights,
inner-validation predictions, and raw outer-test predictions. Calibration
uses that run's primary inner-validation predictions only. Final outputs
under `ensemble/` include `pooled_metrics.csv`, `fold_metrics.csv`,
`fold_summary.csv` (fold mean and sample SD), and `all_test_predictions.csv`.
Runs with only a subset of folds or seeds are partial checks, not the full
published ensemble. Existing results are reused only if their configuration
signature matches; use a new result directory for results from older releases.

## Publishing To GitHub

The generated feature caches, checkpoints, results, and logs are not part of
the source release. The `.pt` files remain local and are ignored by Git. To
publish the reproducible source package:

    git add README.md LICENSE CITATION.cff DATA_NOTICE.md requirements.txt
    git add data feature model my_loader scratch train utils
    git status
    git commit -m "Release PHCNet"

Before committing, confirm that `git status` does not list any `.pt` files.
The tracked feature manifest documents the expected cache names and protocol;
the builder updates it after local generation. The `result/` directory,
checkpoints, logs, and Python bytecode are also intentionally ignored.

## Data Attribution

The data originate from PEPlife2. Retain the PEPlife2 citation and the data
processing and redistribution notice when using or redistributing the CSV and
locked splits. See DATA_NOTICE.md and CITATION.cff.
