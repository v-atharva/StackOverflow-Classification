"""Utility script to compute metrics, error analysis, and calibration from a predictions CSV.

Improvements
------------
* **Error analysis** — dumps the most-confident wrong predictions.
* **Calibration** — computes Expected Calibration Error (ECE) and plots
  a reliability diagram.
* Prints a clear per-class report.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from . import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Calibration helpers
# ---------------------------------------------------------------------------

def expected_calibration_error(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute ECE across *n_bins* confidence bins."""
    confidences = y_probs.max(axis=1)
    predictions = y_probs.argmax(axis=1)
    accuracies = (predictions == y_true).astype(float)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bin_boundaries[:-1], bin_boundaries[1:]):
        in_bin = (confidences > lo) & (confidences <= hi)
        prop = in_bin.mean()
        if prop > 0:
            acc_in_bin = accuracies[in_bin].mean()
            avg_conf = confidences[in_bin].mean()
            ece += np.abs(avg_conf - acc_in_bin) * prop
    return float(ece)


def plot_reliability_diagram(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    output_path: Path,
    n_bins: int = 10,
) -> None:
    """Plot a reliability (calibration) diagram."""
    confidences = y_probs.max(axis=1)
    predictions = y_probs.argmax(axis=1)
    accuracies = (predictions == y_true).astype(float)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)

    bin_accs: List[float] = []
    bin_confs: List[float] = []
    bin_counts: List[int] = []

    for lo, hi in zip(bin_boundaries[:-1], bin_boundaries[1:]):
        in_bin = (confidences > lo) & (confidences <= hi)
        cnt = int(in_bin.sum())
        bin_counts.append(cnt)
        if cnt > 0:
            bin_accs.append(float(accuracies[in_bin].mean()))
            bin_confs.append(float(confidences[in_bin].mean()))
        else:
            bin_accs.append(0.0)
            bin_confs.append((lo + hi) / 2)

    fig, ax = plt.subplots(figsize=(6, 5))
    bar_width = 1.0 / n_bins
    positions = [(lo + hi) / 2 for lo, hi in zip(bin_boundaries[:-1], bin_boundaries[1:])]
    ax.bar(positions, bin_accs, width=bar_width * 0.8, alpha=0.7, label="Accuracy")
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.set_xlabel("Mean predicted confidence")
    ax.set_ylabel("Fraction of positives (accuracy)")
    ax.set_title("Reliability Diagram")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    logger.info("Reliability diagram saved to %s", output_path)


# ---------------------------------------------------------------------------
# Error analysis
# ---------------------------------------------------------------------------

def error_analysis(
    df: pd.DataFrame,
    top_n: int = 20,
) -> pd.DataFrame:
    """Return the *top_n* highest-confidence wrong predictions."""
    prob_cols = [c for c in df.columns if c.startswith("prob_")]
    if not prob_cols:
        logger.warning("No probability columns found — skipping error analysis.")
        return pd.DataFrame()

    wrong = df[df["true_label"] != df["pred_label"]].copy()
    wrong["max_prob"] = wrong[prob_cols].max(axis=1)
    worst = wrong.nlargest(top_n, "max_prob")
    return worst[["Id", "true_label", "pred_label", "max_prob"] + prob_cols]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate predictions CSV.")
    parser.add_argument("--predictions_csv", type=str, required=True)
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Optional path to save metrics JSON.",
    )
    parser.add_argument(
        "--error_csv",
        type=str,
        default=None,
        help="Optional path to write error-analysis CSV.",
    )
    parser.add_argument(
        "--reliability_plot",
        type=str,
        default=None,
        help="Optional path to save reliability diagram.",
    )
    return parser.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )
    args = parse_args()
    df = pd.read_csv(args.predictions_csv)
    true = df["true_label"].map(config.LABEL_TO_ID).values
    pred = df["pred_label"].map(config.LABEL_TO_ID).values

    accuracy = accuracy_score(true, pred)
    macro_f1 = f1_score(true, pred, average="macro")
    report = classification_report(
        true,
        pred,
        target_names=config.LABELS,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(true, pred).tolist()

    # --- Calibration ---
    prob_cols = [f"prob_{label}" for label in config.LABELS]
    ece = None
    if all(c in df.columns for c in prob_cols):
        y_probs = df[prob_cols].values
        ece = expected_calibration_error(true, y_probs)
        logger.info("Expected Calibration Error (ECE): %.4f", ece)

        if args.reliability_plot:
            out = Path(args.reliability_plot)
        else:
            out = config.PLOTS_DIR / "reliability_diagram.png"
        plot_reliability_diagram(true, y_probs, out)

    metrics = {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "report": report,
        "confusion_matrix": cm,
    }
    if ece is not None:
        metrics["ece"] = ece

    print(json.dumps(metrics, indent=2))
    print("\n" + classification_report(true, pred, target_names=config.LABELS, zero_division=0))

    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(metrics, f, indent=2)

    # --- Error analysis ---
    errors = error_analysis(df)
    if not errors.empty:
        print("\n=== Most-confident wrong predictions ===")
        print(errors.to_string(index=False))
        error_path = args.error_csv or str(
            config.METRICS_DIR / "error_analysis.csv"
        )
        errors.to_csv(error_path, index=False)
        logger.info("Error analysis saved to %s", error_path)


if __name__ == "__main__":
    main()
