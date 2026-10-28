# Phase 4 validation selection

Status: `frozen`. Selected model: `defensible_logistic`.
Rows removed: 0.

No missed-crash versus false-alarm cost ratio was specified. Average precision ranks crash scores without an operating point. Crash F1 is the documented balanced threshold. Precision, recall, false positives, and false negatives are reported beside that choice.

## Coverage

Missing training shards: 0.
Missing validation shards: 0.
Missing test shards: 0.

| File | Split | Bytes | Hard links |
| --- | --- | ---: | ---: |
| test-00000-of-00002.parquet | test | 494437326 | 1 |
| test-00001-of-00002.parquet | test | 494255434 | 1 |
| train-00000-of-00011.parquet | train | 512916857 | 1 |
| train-00001-of-00011.parquet | train | 512949454 | 2 |
| train-00002-of-00011.parquet | train | 512949841 | 2 |
| train-00003-of-00011.parquet | train | 512838950 | 2 |
| train-00004-of-00011.parquet | train | 512835365 | 2 |
| train-00005-of-00011.parquet | train | 513139543 | 2 |
| train-00006-of-00011.parquet | train | 512796422 | 2 |
| train-00007-of-00011.parquet | train | 512746735 | 2 |
| train-00008-of-00011.parquet | train | 512753396 | 2 |
| train-00009-of-00011.parquet | train | 512819118 | 2 |
| train-00010-of-00011.parquet | train | 512796165 | 2 |
| validation-00000-of-00003.parquet | validation | 370491448 | 2 |
| validation-00001-of-00003.parquet | validation | 370432581 | 2 |
| validation-00002-of-00003.parquet | validation | 370368262 | 2 |

## Shard audit

Counts below include the official test shards as a consistency check. They are not model-selection inputs.

| File | Rows | Crash | Near miss | Normal | HD gyro | GPS 17 | Nulls | Non-finite | IMU mismatches |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train-00000-of-00011.parquet | 12542 | 245 | 1496 | 10801 | 0 | 6 | 0 | 0 | 0 |
| train-00001-of-00011.parquet | 12542 | 63 | 2789 | 9690 | 0 | 0 | 0 | 0 | 0 |
| train-00002-of-00011.parquet | 12542 | 365 | 3088 | 9089 | 0 | 1 | 0 | 0 | 0 |
| train-00003-of-00011.parquet | 12541 | 1475 | 2187 | 8879 | 4 | 687 | 0 | 0 | 0 |
| train-00004-of-00011.parquet | 12541 | 3268 | 607 | 8666 | 1955 | 2486 | 0 | 0 | 0 |
| train-00005-of-00011.parquet | 12541 | 12541 | 0 | 0 | 12480 | 12486 | 0 | 0 | 1 |
| train-00006-of-00011.parquet | 12541 | 4646 | 839 | 7056 | 8449 | 11211 | 0 | 0 | 0 |
| train-00007-of-00011.parquet | 12541 | 0 | 9796 | 2745 | 7672 | 11022 | 0 | 0 | 0 |
| train-00008-of-00011.parquet | 12541 | 0 | 0 | 12541 | 9852 | 11594 | 0 | 0 | 2 |
| train-00009-of-00011.parquet | 12541 | 0 | 0 | 12541 | 9818 | 11538 | 0 | 0 | 1 |
| train-00010-of-00011.parquet | 12541 | 0 | 0 | 12541 | 9799 | 11536 | 0 | 0 | 1 |
| validation-00000-of-00003.parquet | 9059 | 54 | 1892 | 7113 | 0 | 0 | 0 | 0 | 0 |
| validation-00001-of-00003.parquet | 9058 | 4201 | 1515 | 3342 | 5907 | 6994 | 0 | 0 | 0 |
| validation-00002-of-00003.parquet | 9058 | 0 | 839 | 8219 | 6984 | 8341 | 0 | 0 | 0 |
| test-00000-of-00002.parquet | 12087 | 4221 | 1366 | 6500 | 4127 | 4337 | 0 | 0 | 0 |
| test-00001-of-00002.parquet | 12087 | 0 | 2267 | 9820 | 8708 | 10948 | 0 | 0 | 1 |

Schema problems: none.
Extra columns: none.

## GPS length 17

103187 rows store 17 GPS samples. 53144 repeat the final sample and 34181 repeat the first sample. Those counts can overlap. 6294 rows have no adjacent duplicate, so length 17 is not one parse error. Rows stay at their stored length.

Length 17 totals: rows 103187, no adjacent duplicate 6294, last sample repeated 53144, first sample repeated 34181.

## Excluded features

These columns stay out of every candidate. The exclusion is part of the experiment definition, not a test-set search.

- `gps_samples`: GPS list length is a storage count, not a speed. Standardizing it turns a one-sample format gap into a large score when one length dominates training.
- `imu_samples`: Accelerometer list length is a storage count, not an acceleration.
- `gyro_samples`: Gyroscope list length is a storage count, not a rotation.
- `gyro_is_hd`: This is a sensor-resolution flag. Standardization does not put high-definition and standard gyroscope samples on one scale.
- `peak_abs_gyro_deg_s`: Standard gyroscope rows can saturate near the int16 full scale, while high-definition rows use a finer resolution. One standardized column does not correct that encoding.
- `std_gyro_magnitude_deg_s`: Same gyroscope encoding mismatch as the peak gyroscope feature.

The acceleration-only model uses the five acceleration summaries. The defensible logistic model adds GPS start, end, minimum, and drop, plus Light, Medium, and Heavy vehicle-size indicators. Unknown size is the reference.

## Identifiers

The schema has event_id, label, vehiclesize, gyro_is_hd, gsensor, gyro, and gps_speed. There is no vehicle id, so the paper's vehicle-disjoint split cannot be rebuilt. Official shard names are preserved instead.
Within-file duplicate event IDs: none.
Cross-file duplicate event IDs recorded: 0.

## Validation comparison

Model rule: highest validation average precision; model name breaks ties.
Threshold rule: maximum crash F1 on the selected model's validation scores; the lowest score wins ties.
Frozen threshold: `0.6152401422448208`.

Operating points below maximize crash F1 on validation for display. The frozen model is the one with the highest validation average precision.

| Model | Average precision | Precision | Recall | F1 | Balanced accuracy | FP | FN | Threshold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| peak_baseline | 0.8848 | 0.7815 | 0.8912 | 0.8328 | 0.9225 | 1060 | 463 | 1.9937 |
| accel_logistic | 0.9079 | 0.7964 | 0.9065 | 0.8479 | 0.9317 | 986 | 398 | 0.5811 |
| accel_logistic at 0.5 | 0.9079 | 0.7804 | 0.9246 | 0.8464 | 0.9381 | 1107 | 321 | 0.5000 |
| defensible_logistic | 0.9227 | 0.8169 | 0.9156 | 0.8635 | 0.9388 | 873 | 359 | 0.6152 |
| defensible_logistic at 0.5 | 0.9227 | 0.7902 | 0.9384 | 0.8580 | 0.9461 | 1060 | 262 | 0.5000 |

Validation curves: `reports/phase4/figures/pr_curves_validation.png`.

## Frozen test evaluation

Model `defensible_logistic` was scored once at threshold `0.6152401422448208` after the config was written.
These numbers were not used to change features or the threshold.

### Pooled official test

Precision 0.8244, recall 0.9173, F1 0.8684, balanced accuracy 0.9380, average precision 0.9273. False positives 825, false negatives 349. Confusion `[[19128, 825], [349, 3872]]`.

### test-00000-of-00002.parquet

Precision 0.9465, recall 0.9173, F1 0.9317, balanced accuracy 0.9447, average precision 0.9744. False positives 219, false negatives 349. Confusion `[[7647, 219], [349, 3872]]`.

### test-00001-of-00002.parquet

Precision undefined, recall undefined, F1 undefined, balanced accuracy undefined, average precision undefined. False positives 606, false negatives 0. Confusion `[[11481, 606], [0, 0]]`.
This slice has zero crash labels, so precision, recall, F1, and average precision are undefined.
False positives on this zero-crash shard: 606.

## Limitations

- Official shard names are the split. There is no vehicle id, so this run does not reconstruct the paper's vehicle-disjoint assignment.
- Near-miss events stay in the negative class.
- No missed-crash versus false-alarm cost was supplied, so the threshold is validation crash F1 rather than a cost-weighted point.
- A change motivated by the test result needs a new validation selection and a fresh evaluation. This test pass is not reused for tuning.
- Phase 2 and phase 3 reports are separate outputs and are not rewritten by this command.

Command: `python -m accident_detection.phase4 --data-dir data/VZCrash/data --output-dir reports/phase4`.
