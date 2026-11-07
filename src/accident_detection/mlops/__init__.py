"""IVERAS MLOps lifecycle around the existing crash detector."""

import os

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

MODEL_NAME = "iveras-crash-detector"
EXPERIMENT_NAME = "iveras-crash-detection"
MIN_VALIDATION_AP = 0.85
MIN_VALIDATION_RECALL = 0.85
DRIFT_SHARE_THRESHOLD = 0.3
