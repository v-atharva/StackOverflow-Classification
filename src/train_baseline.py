"""Train a traditional feature-based baseline.

Improvements
------------
* TF-IDF unigram+bigram features combined with handcrafted features.
* LightGBM / Gradient Boosted option alongside Logistic Regression.
* Data-leakage-safe feature imputation (train stats applied to val).
* Reports per-class metrics clearly.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config, features

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Baseline classifier for Stack Overflow quality."
    )
    parser.add_argument("--train_csv", type=str, default="train.csv")
    parser.add_argument("--valid_csv", type=str, default="valid.csv")
    parser.add_argument("--max_iter", type=int, default=500)
    parser.add_argument("--model_name", type=str, default="logreg_baseline")
    parser.add_argument("--use_class_weights", action="store_true")
    parser.add_argument(
        "--use_tfidf",
        action="store_true",
        help="Concatenate TF-IDF features with handcrafted features.",
    )
    parser.add_argument(
        "--tfidf_max_features",
        type=int,
        default=5000,
        help="Max number of TF-IDF features.",
    )
    parser.add_argument(
        "--model_type",
        choices=["logreg", "lgbm"],
        default="logreg",
        help="Which classifier to use.",
    )
    return parser.parse_args()


def _get_text_col(df: pd.DataFrame) -> pd.Series:
    """Combine title + body into a single text column for TF-IDF."""
    return (df["Title"].fillna("") + " " + df["Body"].fillna("")).str.strip()


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s"
    )
    args = parse_args()
    train_df = pd.read_csv(args.train_csv)
    valid_df = pd.read_csv(args.valid_csv)

    # --- Handcrafted features (leakage-safe) ---
    X_train_hc, fill_values = features.build_feature_frame(train_df)
    X_valid_hc, _ = features.build_feature_frame(valid_df, fill_values=fill_values)

    y_train = train_df["Y"].map(config.LABEL_TO_ID).values
    y_valid = valid_df["Y"].map(config.LABEL_TO_ID).values

    # --- Optional TF-IDF ---
    if args.use_tfidf:
        from scipy.sparse import hstack
        from sklearn.feature_extraction.text import TfidfVectorizer

        tfidf = TfidfVectorizer(
            max_features=args.tfidf_max_features,
            ngram_range=(1, 2),
            sublinear_tf=True,
            strip_accents="unicode",
        )
        train_text = _get_text_col(train_df)
        valid_text = _get_text_col(valid_df)
        X_train_tfidf = tfidf.fit_transform(train_text)
        X_valid_tfidf = tfidf.transform(valid_text)

        from scipy.sparse import csr_matrix

        X_train = hstack([csr_matrix(X_train_hc.values), X_train_tfidf])
        X_valid = hstack([csr_matrix(X_valid_hc.values), X_valid_tfidf])
        logger.info(
            "TF-IDF features: %d  |  total feature dim: %d",
            X_train_tfidf.shape[1],
            X_train.shape[1],
        )
    else:
        X_train = X_train_hc
        X_valid = X_valid_hc

    # --- Class weights ---
    if args.use_class_weights:
        counts = np.bincount(y_train)
        total = len(y_train)
        weights = {idx: total / (len(counts) * c) for idx, c in enumerate(counts)}
    else:
        weights = None

    # --- Model selection ---
    if args.model_type == "lgbm":
        try:
            from lightgbm import LGBMClassifier
        except ImportError:
            raise ImportError("Install lightgbm: pip install lightgbm")

        from scipy.sparse import issparse

        X_tr = X_train.toarray() if issparse(X_train) else np.asarray(X_train)
        X_va = X_valid.toarray() if issparse(X_valid) else np.asarray(X_valid)
        clf = LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=63,
            max_depth=-1,
            class_weight="balanced" if args.use_class_weights else None,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        clf.fit(X_tr, y_train)
        probs = clf.predict_proba(X_va)
        preds = probs.argmax(axis=1)
        model_artifact = clf
    else:
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler(with_mean=not args.use_tfidf)),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=args.max_iter,
                        multi_class="multinomial",
                        class_weight=weights,
                        solver="saga",
                        C=1.0,
                    ),
                ),
            ]
        )
        pipeline.fit(X_train, y_train)
        probs = pipeline.predict_proba(X_valid)
        preds = probs.argmax(axis=1)
        model_artifact = pipeline

    # --- Metrics ---
    accuracy = accuracy_score(y_valid, preds)
    macro_f1 = f1_score(y_valid, preds, average="macro")
    report = classification_report(
        y_valid,
        preds,
        target_names=config.LABELS,
        output_dict=True,
        zero_division=0,
    )
    metrics = {"accuracy": accuracy, "macro_f1": macro_f1, "report": report}
    metrics_path = config.METRICS_DIR / f"{args.model_name}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    model_path = config.BASELINE_DIR / f"{args.model_name}.joblib"
    joblib.dump(model_artifact, model_path)

    # --- Save predictions ---
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
    print(classification_report(y_valid, preds, target_names=config.LABELS, zero_division=0))

    # --- Save TF-IDF vectoriser for reuse ---
    if args.use_tfidf:
        tfidf_path = config.BASELINE_DIR / f"{args.model_name}_tfidf.joblib"
        joblib.dump(tfidf, tfidf_path)
        logger.info("TF-IDF vectoriser saved to %s", tfidf_path)

    # --- Save fill values for reproducible imputation ---
    fill_path = config.BASELINE_DIR / f"{args.model_name}_fill_values.joblib"
    joblib.dump(fill_values, fill_path)


if __name__ == "__main__":
    main()
