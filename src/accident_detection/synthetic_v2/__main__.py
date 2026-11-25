"""Write the synthetic v2 table and a generation manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from accident_detection.synthetic_v2.generate import COUNTS, SEED, generate_events, locality_boxes


def main() -> None:
    """Write events.parquet and manifest.json."""

    args = _parser().parse_args()
    frame = generate_events(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.output, index=False)
    manifest = _manifest(frame, args.output, args.seed)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"rows": int(len(frame)), "output": str(args.output)}, indent=2))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Write the synthetic v2 crash table.")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output", type=Path, default=Path("datasets/synthetic_v2/events.parquet"))
    parser.add_argument("--manifest", type=Path, default=Path("datasets/synthetic_v2/manifest.json"))
    return parser


def _manifest(frame, output: Path, seed: int) -> dict[str, object]:
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    counts = frame["label_name"].value_counts().to_dict()
    return {
        "name": "iveras-synthetic-v2",
        "seed": seed,
        "rows": int(len(frame)),
        "requested_counts": COUNTS,
        "actual_counts": {name: int(counts[name]) for name in counts},
        "crs": "EPSG:4326",
        "sha256": digest,
        "localities": list(locality_boxes()),
        "not_vzcrash": True,
    }


if __name__ == "__main__":
    main()
