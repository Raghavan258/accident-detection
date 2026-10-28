# Phase 3 diagnosis

Raw files and the official test shards were not modified. Rows removed: 0.
Selected model: `None`.

## Why the full logistic model floods the second test shard

The full model median-imputes, standardizes with the training mean and scale, then fits balanced logistic regression at probability 0.5.
A GPS length of 17 is 23.38 training standard deviations above the training mean, adding 7.48 to the logit.
`gyro_is_hd` has coefficient 0.0 because every training row is 0. It does not move the score.
Mean contribution on rows the model calls crashes in the second test shard:

| Feature | Coefficient | Train scale | Mean contribution |
| --- | ---: | ---: | ---: |
| gps_samples | 0.3200 | 0.042814 | 7.4193 |
| gps_min_kmh | 0.9996 | 16.275762 | 1.5140 |
| gps_end_kmh | -1.1969 | 20.084770 | -1.4884 |
| std_horizontal_g | 0.6624 | 0.048918 | -0.5027 |
| gps_drop_kmh | -0.3338 | 23.352877 | 0.3123 |
| size_medium | 0.3856 | 0.372011 | 0.2766 |
| min_ax_g | -0.1569 | 0.409046 | -0.0729 |
| peak_magnitude_g | 0.5059 | 0.493757 | 0.0675 |

## Feature distributions

| File | Rows | Crashes | HD gyro | GPS length 17 | Peak horizontal p50 | GPS end p50 | Gyro peak p50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train-00000-of-00011.parquet | 12542 | 245 | 0 | 6 | 0.7301 | 8.0000 | 10.0000 |
| test-00000-of-00002.parquet | 12087 | 4221 | 4127 | 4337 | 0.9602 | 6.0000 | 19.0000 |
| test-00001-of-00002.parquet | 12087 | 0 | 8708 | 10948 | 0.6380 | 31.0000 | 8.4870 |

## Resolution features

High-definition and standard gyroscopes use different stored resolutions. This training shard contains no HD rows, so the fit never observes that encoding.
The HD indicator is constant, and standardizing it does not create a usable resolution correction.
Gyroscope magnitude is not clipped. Its coefficient stays in the full model, but an acceleration-only model leaves the column out rather than rescaling it.
Full-model features: peak_horizontal_g, peak_magnitude_g, min_ax_g, max_ax_g, std_horizontal_g, peak_abs_gyro_deg_s, std_gyro_magnitude_deg_s, imu_samples, gyro_samples, gps_start_kmh, gps_end_kmh, gps_min_kmh, gps_drop_kmh, gps_samples, gyro_is_hd, size_light, size_medium, size_heavy.
Acceleration-only features: peak_horizontal_g, peak_magnitude_g, min_ax_g, max_ax_g, std_horizontal_g.

## Acceleration-only baseline

The acceleration-only logistic model uses the five acceleration summaries already defined for the full model.
Its imputer and scaler are fit on those training columns only. The probability threshold stays 0.5.
This set was specified because the paper's physical baseline uses acceleration and because length and resolution columns are not acceleration. It was not chosen by searching the test set.
No validation shard is available, so this run does not promote it over the peak baseline.

## Exploratory test results

These numbers describe the frozen training fits. They were not used to pick features or thresholds.

### pooled_test

Role: `exploratory_not_for_selection`. Rows: 24174.

| Model | Precision | Recall | F1 | Balanced accuracy | Average precision |
| --- | ---: | ---: | ---: | ---: | ---: |
| peak_baseline | 0.8158 | 0.8593 | 0.8370 | 0.9091 | 0.8962 |
- peak_baseline confusion `[[19134, 819], [594, 3627]]`.
| full_logistic | 0.2733 | 0.9964 | 0.4289 | 0.7179 | 0.9243 |
- full_logistic confusion `[[8768, 11185], [15, 4206]]`.
| accel_logistic | 0.7222 | 0.9668 | 0.8268 | 0.9441 | 0.9045 |
- accel_logistic confusion `[[18383, 1570], [140, 4081]]`.

### test-00000-of-00002.parquet

Role: `exploratory_not_for_selection`. Rows: 12087.

| Model | Precision | Recall | F1 | Balanced accuracy | Average precision |
| --- | ---: | ---: | ---: | ---: | ---: |
| peak_baseline | 0.9396 | 0.8593 | 0.8977 | 0.9148 | 0.9660 |
- peak_baseline confusion `[[7633, 233], [594, 3627]]`.
| full_logistic | 0.8766 | 0.9964 | 0.9327 | 0.9606 | 0.9900 |
- full_logistic confusion `[[7274, 592], [15, 4206]]`.
| accel_logistic | 0.8979 | 0.9668 | 0.9311 | 0.9539 | 0.9660 |
- accel_logistic confusion `[[7402, 464], [140, 4081]]`.

### test-00001-of-00002.parquet

Role: `exploratory_not_for_selection`. Rows: 12087.

| Model | Precision | Recall | F1 | Balanced accuracy | Average precision |
| --- | ---: | ---: | ---: | ---: | ---: |
| peak_baseline | 0.0000 | 0.0000 | 0.0000 | undefined | undefined |
- peak_baseline confusion `[[11501, 586], [0, 0]]`.
| full_logistic | 0.0000 | 0.0000 | 0.0000 | undefined | undefined |
- full_logistic confusion `[[1494, 10593], [0, 0]]`.
| accel_logistic | 0.0000 | 0.0000 | 0.0000 | undefined | undefined |
- accel_logistic confusion `[[10981, 1106], [0, 0]]`.

Figures: `reports\phase3\figures`.

## Why one shard is not a generalization claim

- The training matrix has 245 crashes and 0 high-definition gyroscope rows.
- One shard is 1 of 11 official training shards, and this schema has no vehicle id.
- No validation shard is present, so a model or threshold cannot be frozen without using test labels.
- The second official test shard has no crashes, so its precision is a stress check rather than a tuning target.
- Train-only standardization of a near-constant length or resolution column does not transfer to a shard with a different encoding.

## Download plan

Missing training shards: 10. Missing validation shards: 3.
Approximate download size: 6.1 GiB.

```
hf download vzc-research-chapter/VZCrash data/train-00001-of-00011.parquet data/train-00002-of-00011.parquet data/train-00003-of-00011.parquet data/train-00004-of-00011.parquet data/train-00005-of-00011.parquet data/train-00006-of-00011.parquet data/train-00007-of-00011.parquet data/train-00008-of-00011.parquet data/train-00009-of-00011.parquet data/train-00010-of-00011.parquet data/validation-00000-of-00003.parquet data/validation-00001-of-00003.parquet data/validation-00002-of-00003.parquet --repo-type dataset --local-dir data/VZCrash
```

- Leave every file name unchanged so train, validation, and test stay in their official shards.
- Do not copy a training row into validation or test.
- Do not download the test shards again.
- Fit on the training shards, select the model and threshold on validation, then score test once.

## Decision

Status: `blocked`.
validation split is absent or has no defined average precision
When the validation shards are present, rerun the same command. Selection uses validation average precision only, and the official test split is scored after that choice is frozen.
Command: `python -m accident_detection.phase3 --data-dir data/VZCrash/data --output-dir reports/phase3`.
