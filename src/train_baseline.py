"""Train a traditional feature-based baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config, features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Baseline classifier for Stack Overflow quality.")
    parser.add_argument("--train_csv", type=str, default="train.csv")
    parser.add_argument("--valid_csv", type=str, default="valid.csv")
    parser.add_argument("--max_iter", type=int, default=500)
    parser.add_argument("--model_name", type=str, default="logreg_baseline")
    parser.add_argument("--use_class_weights", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    train_df = pd.read_csv(args.train_csv)
    valid_df = pd.read_csv(args.valid_csv)
    X_train = features.build_feature_frame(train_df)
    X_valid = features.build_feature_frame(valid_df)
    y_train = train_df["Y"].map(config.LABEL_TO_ID).values
    y_valid = valid_df["Y"].map(config.LABEL_TO_ID).values

    if args.use_class_weights:
        counts = np.bincount(y_train)
        total = len(y_train)
        weights = {idx: total / (len(counts) * c) for idx, c in enumerate(counts)}
    else:
        weights = None

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=args.max_iter,
                    multi_class="multinomial",
                    class_weight=weights,
                ),
            ),
        ]
    )
    pipeline.fit(X_train, y_train)
    probs = pipeline.predict_proba(X_valid)
    preds = probs.argmax(axis=1)
    accuracy = accuracy_score(y_valid, preds)
    macro_f1 = f1_score(y_valid, preds, average="macro")
    report = classification_report(
        y_valid, preds, target_names=config.LABELS, output_dict=True, zero_division=0
    )
    metrics = {"accuracy": accuracy, "macro_f1": macro_f1, "report": report}
    metrics_path = config.METRICS_DIR / f"{args.model_name}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    model_path = config.BASELINE_DIR / f"{args.model_name}.joblib"
    joblib.dump(pipeline, model_path)

    pred_df = pd.DataFrame(
        {
            "Id": valid_df["Id"],
            "true_label": valid_df["Y"],
            "pred_label": [config.ID_TO_LABEL[idx] for idx in preds],
        }
    )
    for idx, label in enumerate(config.LABELS):
        pred_df[f"prob_{label}"] = probs[:, idx]
    pred_path = config.BASELINE_DIR / f"{args.model_name}_predictions.csv"
    pred_df.to_csv(pred_path, index=False)
    print(f"Baseline accuracy={accuracy:.3f} macro_f1={macro_f1:.3f}")


if __name__ == "__main__":
    main()

