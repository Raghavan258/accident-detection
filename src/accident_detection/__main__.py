"""Allow `python -m accident_detection` to run the audit."""

from accident_detection.cli import main

raise SystemExit(main())
