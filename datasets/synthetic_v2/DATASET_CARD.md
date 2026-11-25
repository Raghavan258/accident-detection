# iveras-synthetic-v2

Synthetic event table for a map demo. It is inspired by the VZCrash crash-detection task. It is not VZCrash, and it is not a geocoding of those events.

## Motivation

The published windows have accelerometer, gyroscope, and GPS speed. They have no latitude, longitude, locality, or clock time. This table adds those fields so a review can show where a synthetic event sits. It must not be used to replace the frozen VZCrash test score.

## Composition

1,000 events from seed `20261125`:

| Label | Id | Rows | Role in the crash task |
| --- | ---: | ---: | --- |
| crash | 0 | 200 | positive |
| near_miss | 1 | 150 | negative |
| normal_driving | 2 | 650 | negative |

Each row is one event summary, not a raw 16-second waveform. Columns are the twelve defensible features used by the existing logistic model, plus `latitude`, `longitude`, `locality`, `observed_at`, `vehiclesize`, `split`, and `event_id`.

## Collection

No sensor was recorded and no crash report was read. A NumPy generator draws each row.

- Coordinates are WGS84 (`EPSG:4326`), uniform inside one of eight rectangles in Bengaluru. Those rectangles are sampling frames, not official boundaries and not known crash sites.
- Locality is chosen independently of the label, so neighborhood is not a code for crash.
- `observed_at` is uniform over 2024 and is also independent of the label. The timezone is UTC.
- Motion and speed are class-conditional and overlap on purpose. Crash peaks are higher on average. A single draw can still sit near the other class.
- Peak horizontal acceleration is clipped to 0.05–12 g.
- `gps_drop_kmh` is `gps_start_kmh - gps_end_kmh`. The minimum speed is the smaller of the start and the end.
- Vehicle size is one of Light, Medium, or Heavy, and exactly one size indicator is 1.
- The split is stratified by label, 70% train, 15% validation, 15% test, after the rows exist. An event id is in one split only.

## Uses

Use it to show a pin, a locality, and a timestamp next to a synthetic event. Do not quote its accuracy as the VZCrash test result. Do not train a replacement for the production model on this file and present that model as the study model.

## Distribution

`events.parquet` is the table. `manifest.json` records the seed, the counts, the rectangles, and the file hash. Regenerate with:

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m accident_detection.synthetic_v2
```

## Maintenance

Change the seed or the counts only by editing `generate.py` and writing the table again. Do not hand-edit coordinates in the parquet file.
