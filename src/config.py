"""Shared configuration for Stack Overflow question quality classification."""

from pathlib import Path

# Supported labels in canonical order
LABELS = ["HQ", "LQ_EDIT", "LQ_CLOSE"]
LABEL_TO_ID = {label: idx for idx, label in enumerate(LABELS)}
ID_TO_LABEL = {idx: label for label, idx in LABEL_TO_ID.items()}

# Default artifact locations
ARTIFACT_DIR = Path("artifacts")
CHECKPOINT_DIR = ARTIFACT_DIR / "checkpoints"
METRICS_DIR = ARTIFACT_DIR / "metrics"
PLOTS_DIR = ARTIFACT_DIR / "plots"
BASELINE_DIR = ARTIFACT_DIR / "baseline"

# Ensure directories exist when module is imported
for path in [ARTIFACT_DIR, CHECKPOINT_DIR, METRICS_DIR, PLOTS_DIR, BASELINE_DIR]:
    path.mkdir(parents=True, exist_ok=True)

