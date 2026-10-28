# Phase 2 preparation

Raw Parquet files were not modified. Rows removed: 0.
No hyperparameter was chosen from the test shards.

## Split limits

- The paper uses vehicle-disjoint shares of 72% train, 14% validation, and 14% test, stratified by crash and vehicle size.
- Official parquet shard counts are train 11, validation 3, test 2.
- This run found train 1, validation 0, test 2.
- The released columns do not include a vehicle id, so the vehicle split cannot be rebuilt inside one shard.
- Validation files are left unused. They are not fit and they are not added to the test set.
- The paper reports 31,090 verified crashes in the full release. A single training shard is not one eleventh of that crash count.
- The training files in this run contain 245 crash events.
- Metrics below are a first pass, not the ITSC benchmark and not a tuned final model.

## GPS length

A 16-second, 1 Hz speed trace is 16 samples. Length 17 on every row would match an inclusive clock. It does not.
The training shard is almost entirely length 16. Length 17 is not a duplicated last sample often enough to treat it as padding.
Test shards have a different length mix. That mix was not used to pick a target length.

| File | Length | Rows | HD gyro | Crashes | Last sample equals previous |
| --- | ---: | ---: | ---: | ---: | ---: |
| train-00000-of-00011.parquet | 15 | 17 | 0 | 0 | 4 |
| train-00000-of-00011.parquet | 16 | 12519 | 0 | 239 | 4398 |
| train-00000-of-00011.parquet | 17 | 6 | 0 | 6 | 3 |
| test-00000-of-00002.parquet | 11 | 1 | 1 | 1 | 0 |
| test-00000-of-00002.parquet | 12 | 1 | 1 | 1 | 0 |
| test-00000-of-00002.parquet | 13 | 1 | 1 | 1 | 0 |
| test-00000-of-00002.parquet | 14 | 3 | 3 | 3 | 0 |
| test-00000-of-00002.parquet | 15 | 9 | 2 | 2 | 4 |
| test-00000-of-00002.parquet | 16 | 7733 | 0 | 110 | 2628 |
| test-00000-of-00002.parquet | 17 | 4337 | 4117 | 4101 | 2486 |
| test-00000-of-00002.parquet | 42 | 2 | 2 | 2 | 2 |
| test-00001-of-00002.parquet | 11 | 2 | 0 | 0 | 0 |
| test-00001-of-00002.parquet | 12 | 3 | 0 | 0 | 0 |
| test-00001-of-00002.parquet | 13 | 7 | 2 | 0 | 2 |
| test-00001-of-00002.parquet | 14 | 2 | 1 | 0 | 1 |
| test-00001-of-00002.parquet | 15 | 5 | 1 | 0 | 3 |
| test-00001-of-00002.parquet | 16 | 1120 | 2 | 0 | 330 |
| test-00001-of-00002.parquet | 17 | 10948 | 8702 | 0 | 5281 |

## IMU length

Windows whose length is not 1600 usually still match across accelerometer and gyroscope. That is a shorter or longer recording, not a broken axis.
Mismatched channels stay in the matrix. Each sensor is reduced on its own samples. Nothing is resampled to 1600.

| File | Rows | Not 1600 | Mismatched | Max accelerometer | Max gyroscope |
| --- | ---: | ---: | ---: | ---: | ---: |
| train-00000-of-00011.parquet | 12542 | 25 | 0 | 1600 | 1600 |
| test-00000-of-00002.parquet | 12087 | 19 | 0 | 4100 | 4100 |
| test-00001-of-00002.parquet | 12087 | 31 | 1 | 1600 | 1600 |
- Mismatch in `test-00001-of-00002.parquet`: accelerometer 1280, gyroscope 1270, label 2.

## Gyroscope extremes

Non-HD gyroscopes are stored at 1 deg/s. HD gyroscopes use 0.001 deg/s. 32767 deg/s is the int16 full scale.
The 30000 deg/s row in the table is a reporting level, not a clip. No gyroscope sample is replaced.
A training shard with no HD rows cannot teach the resolution difference. That is a limit of this download, not a reason to drop HD test rows.

| File | HD | Rows | Median peak | Max peak | Rows > 1000 | Rows > 30000 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| train-00000-of-00011.parquet | False | 12542 | 10.0000 | 1885.0000 | 2 | 0 |
| train-00000-of-00011.parquet | True | 0 | undefined | undefined | 0 | 0 |
| test-00000-of-00002.parquet | False | 7960 | 11.0000 | 32762.0000 | 7 | 2 |
| test-00000-of-00002.parquet | True | 4127 | 50.1650 | 1360.3040 | 7 | 0 |
- `test-00000-of-00002.parquet` row 7115: peak 32762.0 deg/s, HD False, label 2.
- `test-00000-of-00002.parquet` row 7212: peak 32654.0 deg/s, HD False, label 2.
| test-00001-of-00002.parquet | False | 3379 | 12.0000 | 880.0000 | 0 | 0 |
| test-00001-of-00002.parquet | True | 8708 | 7.5325 | 1253.4709 | 2 | 0 |

## Decisions

- Crash is the positive class. Near-miss and normal driving stay negative. No class is resampled.
- GPS lists stay at their stored length. They are not truncated to 16 or padded to 17.
- IMU windows stay at their stored length. They are not resampled to 1600 samples.
- Accelerometer and gyroscope features are reduced separately when their lengths differ.
- Gyroscope samples are not clipped, including values near the int16 full scale.
- Median imputation and standardization are fit on training rows only.
- The peak threshold is chosen on training rows only. Logistic regression uses probability 0.5.
- Test rows are scored once and are not used to change features, the scaler, or the threshold.

## Baselines

Peak baseline score: max over time of hypot(accel_x, accel_y).
Its threshold is 2.201621 g, chosen by maximum crash F1 on the training rows; lowest threshold wins ties.
Logistic regression uses C=1.0, class_weight=balanced, and crash probability >= 0.5.
The imputer and scaler means come from the training rows only.
Unknown vehicle size is the reference category. Light, Medium, and Heavy are indicators.

## Command

`python -m accident_detection.experiment --data-dir data/VZCrash/data --output-dir reports/phase2`

## This run

Training metrics are fit diagnostics. Test metrics are one pass on the local test shards.
They are not a claim of final performance.

### Training diagnostics

Role: `fit_diagnostic`. Rows: 12542.

| Model | Precision | Recall | F1 | Balanced accuracy | Average precision |
| --- | ---: | ---: | ---: | ---: | ---: |
| peak_baseline | 0.7725 | 0.8041 | 0.7880 | 0.8997 | 0.8201 |
| logistic_regression | 0.4314 | 0.9633 | 0.5960 | 0.9690 | 0.8445 |
- peak_baseline confusion [[tn, fp], [fn, tp]] = `[[12239, 58], [48, 197]]`.
- logistic_regression confusion [[tn, fp], [fn, tp]] = `[[11986, 311], [9, 236]]`.

### test-00000-of-00002.parquet

Role: `single_preliminary_evaluation`. Rows: 12087.

| Model | Precision | Recall | F1 | Balanced accuracy | Average precision |
| --- | ---: | ---: | ---: | ---: | ---: |
| peak_baseline | 0.9396 | 0.8593 | 0.8977 | 0.9148 | 0.9660 |
| logistic_regression | 0.8766 | 0.9964 | 0.9327 | 0.9606 | 0.9900 |
- peak_baseline confusion [[tn, fp], [fn, tp]] = `[[7633, 233], [594, 3627]]`.
- logistic_regression confusion [[tn, fp], [fn, tp]] = `[[7274, 592], [15, 4206]]`.

### test-00001-of-00002.parquet

Role: `single_preliminary_evaluation`. Rows: 12087.

| Model | Precision | Recall | F1 | Balanced accuracy | Average precision |
| --- | ---: | ---: | ---: | ---: | ---: |
| peak_baseline | 0.0000 | 0.0000 | 0.0000 | undefined | undefined |
| logistic_regression | 0.0000 | 0.0000 | 0.0000 | undefined | undefined |
- peak_baseline confusion [[tn, fp], [fn, tp]] = `[[11501, 586], [0, 0]]`.
- logistic_regression confusion [[tn, fp], [fn, tp]] = `[[1494, 10593], [0, 0]]`.

### Pooled test shards

Role: `single_preliminary_evaluation`. Rows: 24174.

| Model | Precision | Recall | F1 | Balanced accuracy | Average precision |
| --- | ---: | ---: | ---: | ---: | ---: |
| peak_baseline | 0.8158 | 0.8593 | 0.8370 | 0.9091 | 0.8962 |
| logistic_regression | 0.2733 | 0.9964 | 0.4289 | 0.7179 | 0.9243 |
- peak_baseline confusion [[tn, fp], [fn, tp]] = `[[19134, 819], [594, 3627]]`.
- logistic_regression confusion [[tn, fp], [fn, tp]] = `[[8768, 11185], [15, 4206]]`.

