# Accident detection on VZCrash

Crash detection from 16-second ego-vehicle IMU and GPS windows. The released model was chosen on the official validation split and scored **once** on the official test split.

**Frozen test result:** precision **0.8244**, recall **0.9173**, F1 **0.8684**, average precision **0.9273**.

That number is the end of a four-phase study, not a tuned leaderboard entry. An earlier logistic model looked strong on ranking quality and then predicted crashes for 10,593 of 12,087 rows in a test shard that contains **zero** crashes. The fix was to remove a storage-count feature, not to search thresholds on the test set.

Dataset: [VZCrash](https://huggingface.co/datasets/vzc-research-chapter/VZCrash) (Bianconcini et al., IEEE ITSC 2026, CC BY-NC 4.0). Raw Parquet is not in this repository.

## Result a reviewer can check

| | Precision | Recall | F1 | Balanced accuracy | Average precision | False positives | False negatives |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Official test, pooled | 0.8244 | 0.9173 | 0.8684 | 0.9380 | 0.9273 | 825 | 349 |
| `test-00000` | 0.9465 | 0.9173 | 0.9317 | 0.9447 | 0.9744 | 219 | 349 |
| `test-00001` (0 crashes) | undefined | undefined | undefined | undefined | undefined | 606 | 0 |

Confusion on the pooled test, rows `[non-crash, crash]`: `[[19128, 825], [349, 3872]]`.

`test-00001` has no positive labels, so precision, recall, F1, and average precision are undefined there. The 606 false positives are the stress check.

Selection rule, fixed before that score:

- Model: highest **validation** average precision. Ties break by model name.
- Threshold: maximum **validation** crash F1. Ties keep the lower score.
- No missed-crash versus false-alarm cost was supplied, so F1 is the documented balanced point, not a hidden optimum.

| Candidate | Validation AP | Precision | Recall | F1 | FP | FN | Threshold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Acceleration peak | 0.8848 | 0.7815 | 0.8912 | 0.8328 | 1,060 | 463 | 1.9937 g |
| Acceleration-only logistic | 0.9079 | 0.7964 | 0.9065 | 0.8479 | 986 | 398 | 0.5811 |
| **Defensible logistic (selected)** | **0.9227** | **0.8169** | **0.9156** | **0.8635** | **873** | **359** | **0.6152** |

At probability 0.5 the selected model has validation recall 0.9384 and 1,060 false positives. The frozen operating point is the F1 threshold above. Config: `reports/phase4/experiment_config.json`. Curves: `reports/phase4/figures/`.

## What the model uses

Positive class: crash (`label` 0). Near-miss and normal driving stay negative. No class is resampled.

The selected model is class-weighted logistic regression (`C=1`, `lbfgs`, seed 0) on features fit with a training-only median imputer and standard scaler:

- horizontal and vector acceleration peaks, longitudinal min and max, horizontal standard deviation
- GPS start, end, minimum, and drop
- Light, Medium, and Heavy vehicle-size indicators (Unknown is the reference)

Left out on purpose, before any test score:

| Feature | Why it is excluded |
| --- | --- |
| `gps_samples`, `imu_samples`, `gyro_samples` | List length is a storage count. On one training shard, GPS length 17 was 23.38 training standard deviations above the mean and added about **+7.42** to the logit. |
| `gyro_is_hd` | Resolution flag. Standardization does not put 1 deg/s and 0.001 deg/s encodings on one scale. |
| Gyroscope peak and spread | Standard rows reach 32,766 deg/s and, in one validation shard, 2,147,483.75 deg/s. High-definition peaks top out near 1,394 deg/s. |

The acceleration-peak baseline is `max hypot(accel_x, accel_y)`. On the one-shard test split it reproduced the paper's reported physical-baseline average precision of **0.8962**.

## Why the first logistic model failed

Phase 2 fit one training shard (12,542 rows, 245 crashes) and scored both official test shards (24,174 rows).

| Model | Test precision | Recall | F1 | Average precision |
| --- | ---: | ---: | ---: | ---: |
| Acceleration peak, threshold 2.2016 g | 0.8158 | 0.8593 | 0.8370 | 0.8962 |
| Full logistic, probability 0.5 | 0.2733 | 0.9964 | 0.4289 | 0.9243 |

The full model’s average precision was high and its operating point was not. On `test-00001` it predicted **10,593** crashes. Training GPS length was almost always 16, so the training scale of `gps_samples` was 0.0428. The second test shard stores length 17 on 10,948 of 12,087 rows. `gyro_is_hd` had coefficient 0 because that shard had no high-definition rows.

Those test numbers were diagnostic. They were not used to pick the Phase 4 model.

## Data, kept intact

| Split | Shards | Rows | Crash labels |
| --- | ---: | ---: | ---: |
| Train | 11 | 137,954 | 22,603 |
| Validation | 3 | 27,175 | 4,255 |
| Test | 2 | 24,174 | 4,221 |
| **All** | **16** | **189,303** | **31,079** |

Audit of every shard: one schema, no extra columns, 0 nulls, 0 non-finite values, 0 duplicate `event_id`s, 0 rows removed. Six accelerometer/gyroscope length mismatches were kept and summarized separately. Nothing was clipped, padded, truncated, or resampled to 1,600 IMU samples.

GPS length 17 appears in 103,187 rows. 53,144 repeat the last sample and 34,181 repeat the first (those counts overlap). 6,294 have no adjacent duplicate, so length 17 is not a single parse error.

The shards are not interchangeable draws. `train-00005` is 12,541 crashes. `train-00008` through `train-00010` contain none. There is no vehicle id, so this repository uses the official shard names and does not claim to rebuild the paper’s vehicle-disjoint split.

## Layout

```text
src/accident_detection/
  audit.py  schema.py  columns.py  signals.py     # read-only audit, no row drops
  features.py  quality.py  baselines.py  metrics.py
  experiment.py          # phase 2, pinned to the original training shard
  phase3.py  diagnosis.py # generalization diagnosis, still one training shard
  phase4.py  selection.py # train fit, validation choice, one test score
  coverage.py  protocol.py
tests/                   # 47 tests: leakage, zero-crash slices, no silent edits
reports/phase2|phase3|phase4
```

Preprocessing, coefficients, and thresholds are fit on training rows. `validation_choice` takes validation scores only. Phase 2 and Phase 3 stay pinned to `train-00000-of-00011.parquet`, so a later full download cannot silently rewrite those results.

## Run

Python 3.11. The VZCrash files are gated. Download them into `data/VZCrash/` and keep the official file names.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m accident_detection.phase4 --data-dir data/VZCrash/data --output-dir reports/phase4
```

Phase 1 audit: `python -m accident_detection.cli --data-dir data/VZCrash/data --report-dir reports/data_audit --figure-dir reports/figures`.

## Limits

- Official shard assignment is the split. It is not a reconstructed vehicle-level split.
- The threshold maximizes validation F1. A different cost ratio needs a new validation selection and a new test evaluation.
- The test metrics above were not used to change features or the threshold.
- Near-miss events are negatives, matching the dataset’s crash-detection task.
- Counts here are from this checkout (31,079 crash labels). The paper reports 31,090 verified crashes.

Cite Bianconcini et al., “VZCrash: A Large-Scale IMU Dataset of Ego-Vehicle Crashes,” IEEE ITSC 2026, when the data are used.
