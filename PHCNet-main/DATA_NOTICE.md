# Data source, redistribution, and transformations

## Source

The packaged CSV and fixed split files are derived from PEPlife2:

- Website: https://webs.iiitd.edu.in/raghava/peplife2/
- Citation: Alam U, Chaudhary K, Kumar N, Tomer R, Patiyal S, Raghava GPS.
  PEPlife2: An Updated Repository of the Half-Life of Peptides and Proteins.
  Immuno. 2026;6(2):26.
- DOI: https://doi.org/10.3390/immuno6020026

The repository maintainer has confirmed that the processed PEPlife2 records
and derived cross-validation split files may be redistributed. Any copy or
derivative dataset should retain this source notice and cite PEPlife2.

This notice documents data provenance and attribution. It does not change the
license of the PHCNet source code or the licenses of ESM2 and ProtBERT.

## Packaged derivative dataset

`data/peplife2_exact_plus_mean_sd.csv` contains:

- 1,556 exact half-life records.
- 655 mean+SD auxiliary records.
- 2,211 records in total.

All half-lives are converted to minutes, and the regression target is
`log10(half-life in minutes)`.

## Selection criteria

Records require a usable modified sequence and the following normalized
conditions:

- Species: human, monkey, mouse, or rat.
- Matrix: serum, plasma, or blood.
- Setting: in vivo or in vitro.

No sequence-length filter was applied during data construction. Experimental
conditions and modification metadata were retained rather than reducing the
dataset to sequence-only records.

## Exact-label processing

Of 2,929 source exact-label records, 1,556 passed the sequence and condition
criteria. These are the only records eligible for inner validation and outer
testing.

## Mean+SD processing

Of 905 condition-filtered uncertain records, 693 had parseable positive mean
and SD values. Duplicate measurements were merged. Records were then excluded
for condition-level overlap with exact labels (27), SD greater than or equal to
the mean (2), or invalid/nonpositive values (1), leaving 655 auxiliary rows.

For each retained row, the auxiliary training weight is computed at runtime as:

```text
sigma_log10 = SD_minutes / (mean_minutes * ln(10))
weight_mean_sd = 0.2 * 0.1^2 / (0.1^2 + sigma_log10^2)
```

For the packaged auxiliary cohort, the resulting weights range from
approximately 0.0132 to 0.2000. Mean+SD rows are used only in supervised
inner-training losses.

## Split protocol

The 1,556 exact records follow a locked target-stratified random 10-fold
assignment generated with seed 3407. Every exact row appears in an outer test
set exactly once. The 655 mean+SD rows are included only in the training
portion of every fold.

The evaluation unit is a condition-specific record. These splits therefore
measure record-level random-CV performance, not strict unseen-sequence or
unseen-study generalization.
