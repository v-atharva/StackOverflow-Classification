"""Utility script to compute metrics from a predictions CSV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from . import config


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate predictions CSV.")
    parser.add_argument("--predictions_csv", type=str, required=True)
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Optional path to save metrics JSON.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.predictions_csv)
    true = df["true_label"].map(config.LABEL_TO_ID).values
    pred = df["pred_label"].map(config.LABEL_TO_ID).values
    accuracy = accuracy_score(true, pred)
    macro_f1 = f1_score(true, pred, average="macro")
    report = classification_report(
        true, pred, target_names=config.LABELS, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(true, pred).tolist()
    metrics = {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "report": report,
        "confusion_matrix": cm,
    }
    print(json.dumps(metrics, indent=2))
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()

