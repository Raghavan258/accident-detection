# VZCrash data audit

Generated at 2026-10-28T11:30:00.630872+00:00.
Command: `python -m accident_detection.cli --data-dir data/VZCrash/data --report-dir reports/data_audit --figure-dir reports/figures`.

Raw Parquet files were read and not modified. No rows were removed.
No model was trained, and no cleaning threshold was fit on the test split.

## File inventory

| File | Split | Rows | Row groups | Bytes | Schema problems |
| --- | --- | ---: | ---: | ---: | --- |
| test-00000-of-00002.parquet | test | 12087 | 5 | 494437326 | none |
| test-00001-of-00002.parquet | test | 12087 | 5 | 494255434 | none |
| train-00000-of-00011.parquet | train | 12542 | 5 | 512916857 | none |

## Class distribution

Counts below are per file. A shard is not assumed to match the full split.

| File | Split | Label | Count | Percent of file |
| --- | --- | --- | ---: | ---: |
| test-00000-of-00002.parquet | test | 0 crash | 4221 | 34.92 |
| test-00000-of-00002.parquet | test | 1 near_miss | 1366 | 11.30 |
| test-00000-of-00002.parquet | test | 2 normal_driving | 6500 | 53.78 |
| test-00001-of-00002.parquet | test | 0 crash | 0 | 0.00 |
| test-00001-of-00002.parquet | test | 1 near_miss | 2267 | 18.76 |
| test-00001-of-00002.parquet | test | 2 normal_driving | 9820 | 81.24 |
| train-00000-of-00011.parquet | train | 0 crash | 245 | 1.95 |
| train-00000-of-00011.parquet | train | 1 near_miss | 1496 | 11.93 |
| train-00000-of-00011.parquet | train | 2 normal_driving | 10801 | 86.12 |

## Split totals

These sums hide the per-file differences shown above.

| Split | Label | Count | Percent of split |
| --- | --- | ---: | ---: |
| test | 0 crash | 4221 | 17.46 |
| test | 1 near_miss | 3633 | 15.03 |
| test | 2 normal_driving | 16320 | 67.51 |
| train | 0 crash | 245 | 1.95 |
| train | 1 near_miss | 1496 | 11.93 |
| train | 2 normal_driving | 10801 | 86.12 |

## Missingness

Outer nulls are null column values. Null elements are missing samples inside a list.

- `test-00000-of-00002.parquet` outer nulls: event_id=0, label=0, vehiclesize=0, gyro_is_hd=0, gsensor=0, gyro=0, gps_speed=0.
  Null elements: gsensor=0, gyro=0, gps_speed=0. Non-finite scalars: gsensor=0, gyro=0, gps_speed=0.
- `test-00001-of-00002.parquet` outer nulls: event_id=0, label=0, vehiclesize=0, gyro_is_hd=0, gsensor=0, gyro=0, gps_speed=0.
  Null elements: gsensor=0, gyro=0, gps_speed=0. Non-finite scalars: gsensor=0, gyro=0, gps_speed=0.
- `train-00000-of-00011.parquet` outer nulls: event_id=0, label=0, vehiclesize=0, gyro_is_hd=0, gsensor=0, gyro=0, gps_speed=0.
  Null elements: gsensor=0, gyro=0, gps_speed=0. Non-finite scalars: gsensor=0, gyro=0, gps_speed=0.

## Duplicates

No duplicate `event_id` values were found within or across the audited files.

## Signal integrity

### test-00000-of-00002.parquet

- `gsensor` lengths: 1020:1, 1130:1, 1250:1, 1300:1, 1350:2, 1400:1, 1420:1, 1430:2, 1440:1, 1460:1, 1470:1, 1500:2, 1550:1, 1580:1, 1600:12068, 4100:2.
  Finite range: min=-32.7239990234375 max=32.60200119018555 count=58021740.
- `gyro` lengths: 1020:1, 1130:1, 1250:1, 1300:1, 1350:2, 1400:1, 1420:1, 1430:2, 1440:1, 1460:1, 1470:1, 1500:2, 1550:1, 1580:1, 1600:12068, 4100:2.
  Finite range: min=-32762.0 max=32730.0 count=58021740.
- `gps_speed` lengths: 11:1, 12:1, 13:1, 14:3, 15:9, 16:7733, 17:4337, 42:2.
  Finite range: min=0.0 max=163.0 count=197754.

### test-00001-of-00002.parquet

- `gsensor` lengths: 1080:1, 1100:1, 1140:1, 1150:1, 1160:1, 1170:1, 1200:1, 1250:1, 1280:1, 1300:4, 1310:1, 1400:1, 1410:2, 1440:1, 1450:1, 1480:1, 1490:1, 1510:2, 1530:3, 1540:2, 1550:1, 1570:1, 1590:1, 1600:12056.
  Finite range: min=-5.065000057220459 max=5.8379998207092285 count=57996360.
- `gyro` lengths: 1080:1, 1100:1, 1140:1, 1150:1, 1160:1, 1170:1, 1200:1, 1250:1, 1270:1, 1300:4, 1310:1, 1400:1, 1410:2, 1440:1, 1450:1, 1480:1, 1490:1, 1510:2, 1530:3, 1540:2, 1550:1, 1570:1, 1590:1, 1600:12056.
  Finite range: min=-1006.7249755859375 max=1253.470947265625 count=57996330.
- `gps_speed` lengths: 11:2, 12:3, 13:7, 14:2, 15:5, 16:1120, 17:10948.
  Finite range: min=0.0 max=216.0 count=204288.

### train-00000-of-00011.parquet

- `gsensor` lengths: 1410:1, 1420:4, 1430:1, 1450:2, 1460:1, 1480:1, 1500:7, 1510:1, 1520:2, 1540:2, 1550:1, 1560:1, 1590:1, 1600:12517.
  Finite range: min=-5.248000144958496 max=5.247000217437744 count=60193170.
- `gyro` lengths: 1410:1, 1420:4, 1430:1, 1450:2, 1460:1, 1480:1, 1500:7, 1510:1, 1520:2, 1540:2, 1550:1, 1560:1, 1590:1, 1600:12517.
  Finite range: min=-1885.0 max=1674.0 count=60193170.
- `gps_speed` lengths: 15:17, 16:12519, 17:6.
  Finite range: min=0.0 max=183.0 count=200661.


## Train-only descriptive statistics

Computed only on scanned training files. Not a scaler and not a threshold.

| Class | Peaks | Mean g | p50 g | p95 g |
| --- | ---: | ---: | ---: | ---: |
| all | 12542 | 1.4996 | 1.3820 | 2.0641 |
| crash | 245 | 3.9392 | 3.8681 | 6.2884 |
| near_miss | 1496 | 1.4881 | 1.4168 | 1.9987 |
| normal_driving | 10801 | 1.4459 | 1.3711 | 1.9195 |

## Anomaly policy

- Every audited row is kept. `rows_removed` is 0.
- IMU lists that are not 1600 samples are flagged `length_not_nominal`.
- GPS lists that are not 16 samples are flagged `length_not_nominal` and kept at their original length.
- Accelerometer and gyroscope length disagreements are flagged `imu_length_mismatch`.
- Nulls, empty lists, non-finite values, unknown labels, and duplicate event IDs are flagged.
- Flags are listed in `anomalies.csv`. They are not a filter.

## Representative examples

Plots use the first nominal-length training row of each class. Test rows are not plotted.

- crash: `train-00000-of-00011.parquet` row 1, event 9366, nominal.
- near_miss: `train-00000-of-00011.parquet` row 132, event 19241, nominal.
- normal_driving: `train-00000-of-00011.parquet` row 0, event 9322, nominal.

## Limitations

- train: found 1 of 11 official parquet shards in this directory.
- validation: found 0 of 3 official parquet shards in this directory.
- This phase does not train a model and does not standardize signals.
- Nominal IMU length is 100 Hz times 16 seconds. Nominal GPS length is 1 Hz times 16 seconds.
- Those nominal lengths were not replaced by the most common length in a test shard.
- Class-conditional peak summaries and example plots use training rows only.
- Test and validation files are still counted, checked for integrity, and kept separate.
- Large gyroscope values are reported in the finite range and are not clipped.
- The release license is CC BY-NC 4.0. This audit does not add video or location fields.
