"""
classifier.py — TeleGuard Phase 3: anomaly classifier + A/B experiment.

This is the experiment that proves the thesis. We train the SAME XGBoost model
two ways and compare them on the SAME real held-out test set:

    Model A (baseline)  : trained on the ORIGINAL real data only.
    Model B (augmented) : trained on real data + CTGAN synthetic anomalies.

Thesis: Model B should detect rare anomalies better -> higher recall & F1 on
the anomaly class.

FAIR-COMPARISON RULES (critical)
--------------------------------
* Both models use the EXACT same hyperparameters and the same random seed, so
  the ONLY difference is the training data. (We deliberately do NOT add
  XGBoost class weighting, because that would confound the augmentation effect
  we're trying to measure.)
* The held-out test set is REAL data only. Synthetic rows NEVER enter the test.

WHY KDDTest+ IS THE TEST SET (not a 20% slice of train)
-------------------------------------------------------
The CTGAN in Phase 2 was trained on ALL anomalies in train_processed.csv. If we
carved a 20% test slice out of that same file, Model B's synthetic data would
have been generated from rows sitting in the test set -> data leakage that
unfairly inflates Model B. KDDTest+ (test_processed.csv) is a separate real
dataset the GAN never saw, so it is leakage-free AND it is the standard NSL-KDD
benchmark holdout. That gives the fair comparison the thesis requires.

Run:
    python src/classifier.py
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # headless: save figures without a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from xgboost import XGBClassifier

# --- Make `from src...` work whether run as script or module -----------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import (  # noqa: E402
    MODELS_DIR,
    PROCESSED_DIR,
    TARGET_COL,
    get_feature_columns,
)

RANDOM_SEED = 42
POS_LABEL = 1  # the anomaly class

MODEL_B_FILENAME = "classifier_model_b.pkl"   # the better model -> dashboard
METRICS_FILENAME = "ab_metrics_summary.csv"
CM_FIG = "confusion_matrices_AB.png"
ROC_FIG = "roc_curves_AB.png"
BAR_FIG = "metrics_comparison_AB.png"

# Same hyperparameters for BOTH models — only the training DATA differs.
XGB_PARAMS = dict(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.9,
    colsample_bytree=0.9,
    eval_metric="logloss",
    tree_method="hist",
    n_jobs=-1,
    random_state=RANDOM_SEED,
)


def set_seeds(seed: int = RANDOM_SEED) -> None:
    """Pin all RNGs we touch, for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_datasets(processed_dir: Path):
    """Load real train, synthetic anomalies, and the real held-out test set."""
    processed_dir = Path(processed_dir)
    train_path = processed_dir / "train_processed.csv"
    synth_path = processed_dir / "synthetic_anomalies.csv"
    test_path = processed_dir / "test_processed.csv"

    if not train_path.exists():
        raise FileNotFoundError(f"{train_path} missing. Run: python -m src.preprocessing")
    if not test_path.exists():
        raise FileNotFoundError(f"{test_path} missing. Run: python -m src.preprocessing")
    if not synth_path.exists():
        raise FileNotFoundError(
            f"{synth_path} missing. Run Phase 2 first: python src/gan_train.py"
        )

    train_df = pd.read_csv(train_path)
    synth_df = pd.read_csv(synth_path)
    test_df = pd.read_csv(test_path)
    return train_df, synth_df, test_df


def split_xy(df: pd.DataFrame, feature_cols: list[str]):
    """Return (X, y) using the canonical feature order from preprocessing."""
    X = df[feature_cols]
    y = df[TARGET_COL].astype(int)
    return X, y


# --------------------------------------------------------------------------- #
# Train + evaluate
# --------------------------------------------------------------------------- #
def train_xgb(X, y) -> XGBClassifier:
    """Train an XGBoost classifier with the shared, fixed hyperparameters."""
    model = XGBClassifier(**XGB_PARAMS)
    model.fit(X, y)
    return model


def evaluate(model, X_test, y_test, name: str) -> dict:
    """Compute the metrics we report for the anomaly (positive) class."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]  # P(anomaly)

    metrics = {
        "model": name,
        "f1_anomaly": f1_score(y_test, y_pred, pos_label=POS_LABEL),
        "precision_anomaly": precision_score(y_test, y_pred, pos_label=POS_LABEL),
        "recall_anomaly": recall_score(y_test, y_pred, pos_label=POS_LABEL),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    fpr, tpr, _ = roc_curve(y_test, y_proba, pos_label=POS_LABEL)
    return {**metrics, "confusion_matrix": cm, "fpr": fpr, "tpr": tpr,
            "y_pred": y_pred, "y_proba": y_proba}


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def plot_confusion_matrices(res_a, res_b, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    labels = ["normal (0)", "anomaly (1)"]
    for ax, res in zip(axes, (res_a, res_b)):
        sns.heatmap(res["confusion_matrix"], annot=True, fmt="d", cmap="Blues",
                    cbar=False, xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_title(f"{res['model']}  (F1={res['f1_anomaly']:.3f})")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    fig.suptitle("Confusion matrices — same real test set")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_roc_curves(res_a, res_b, out_path: Path):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(res_a["fpr"], res_a["tpr"],
            label=f"{res_a['model']} (AUC={res_a['roc_auc']:.3f})")
    ax.plot(res_b["fpr"], res_b["tpr"],
            label=f"{res_b['model']} (AUC={res_b['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC curves — Model A vs Model B")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_metrics_bar(res_a, res_b, out_path: Path):
    keys = ["f1_anomaly", "precision_anomaly", "recall_anomaly", "roc_auc"]
    labels = ["F1\n(anomaly)", "Precision\n(anomaly)", "Recall\n(anomaly)", "ROC-AUC"]
    a_vals = [res_a[k] for k in keys]
    b_vals = [res_b[k] for k in keys]

    x = np.arange(len(keys))
    width = 0.38
    fig, ax = plt.subplots(figsize=(8, 5))
    bars_a = ax.bar(x - width / 2, a_vals, width, label=res_a["model"], color="#8d99ae")
    bars_b = ax.bar(x + width / 2, b_vals, width, label=res_b["model"], color="#2a9d8f")
    ax.bar_label(bars_a, fmt="%.3f", fontsize=8, padding=2)
    ax.bar_label(bars_b, fmt="%.3f", fontsize=8, padding=2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model A (baseline) vs Model B (GAN-augmented)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #
def summary_table(res_a, res_b) -> pd.DataFrame:
    keys = ["f1_anomaly", "precision_anomaly", "recall_anomaly", "roc_auc"]
    rows = []
    for k in keys:
        a, b = res_a[k], res_b[k]
        rows.append({
            "metric": k,
            "Model_A_baseline": a,
            "Model_B_augmented": b,
            "delta_(B-A)": b - a,
            "pct_change": (100.0 * (b - a) / a) if a else float("nan"),
        })
    return pd.DataFrame(rows)


def print_summary(table: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("A/B RESULTS - Model A (baseline) vs Model B (GAN-augmented)")
    print("=" * 70)
    with pd.option_context("display.float_format", "{:.4f}".format):
        print(table.to_string(index=False))
    print("=" * 70)

    f1_row = table.loc[table["metric"] == "f1_anomaly"].iloc[0]
    rec_row = table.loc[table["metric"] == "recall_anomaly"].iloc[0]
    verdict = (
        "Model B IMPROVED anomaly detection"
        if (f1_row["delta_(B-A)"] > 0 and rec_row["delta_(B-A)"] > 0)
        else "Model B did NOT clearly beat the baseline"
    )
    print(f"VERDICT: {verdict}.")
    print(f"  F1 (anomaly):     {f1_row['Model_A_baseline']:.4f} -> "
          f"{f1_row['Model_B_augmented']:.4f}  "
          f"({f1_row['delta_(B-A)']:+.4f})")
    print(f"  Recall (anomaly): {rec_row['Model_A_baseline']:.4f} -> "
          f"{rec_row['Model_B_augmented']:.4f}  "
          f"({rec_row['delta_(B-A)']:+.4f})")
    print("=" * 70)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run(
    processed_dir: Path = PROCESSED_DIR,
    models_dir: Path = MODELS_DIR,
    figures_dir: Path = PROJECT_ROOT / "reports" / "figures",
) -> pd.DataFrame:
    set_seeds()
    processed_dir, models_dir, figures_dir = map(Path, (processed_dir, models_dir, figures_dir))
    models_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    feature_cols, _ = get_feature_columns()
    train_df, synth_df, test_df = load_datasets(processed_dir)

    # ----- Build the two training sets -----
    # Model A: original real data only.
    X_a, y_a = split_xy(train_df, feature_cols)
    # Model B: real data + synthetic anomalies (shuffled, seeded).
    augmented = pd.concat([train_df, synth_df], ignore_index=True)
    augmented = augmented.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
    X_b, y_b = split_xy(augmented, feature_cols)

    # ----- Same REAL held-out test for BOTH (zero synthetic) -----
    X_test, y_test = split_xy(test_df, feature_cols)

    print(f"Model A train: {X_a.shape[0]:,} rows "
          f"(normal={int((y_a==0).sum()):,}, anomaly={int((y_a==1).sum()):,})")
    print(f"Model B train: {X_b.shape[0]:,} rows "
          f"(normal={int((y_b==0).sum()):,}, anomaly={int((y_b==1).sum()):,})  "
          f"[+{len(synth_df):,} synthetic]")
    print(f"Test (real):   {X_test.shape[0]:,} rows "
          f"(normal={int((y_test==0).sum()):,}, anomaly={int((y_test==1).sum()):,})")

    # ----- Train -----
    print("\nTraining Model A (baseline)...")
    model_a = train_xgb(X_a, y_a)
    print("Training Model B (augmented)...")
    model_b = train_xgb(X_b, y_b)

    # ----- Evaluate on the same real test set -----
    res_a = evaluate(model_a, X_test, y_test, "Model A (baseline)")
    res_b = evaluate(model_b, X_test, y_test, "Model B (augmented)")

    # ----- Plots -----
    plot_confusion_matrices(res_a, res_b, figures_dir / CM_FIG)
    plot_roc_curves(res_a, res_b, figures_dir / ROC_FIG)
    plot_metrics_bar(res_a, res_b, figures_dir / BAR_FIG)
    print(f"\nSaved figures -> {figures_dir / CM_FIG}")
    print(f"              -> {figures_dir / ROC_FIG}")
    print(f"              -> {figures_dir / BAR_FIG}")

    # ----- Summary -----
    table = summary_table(res_a, res_b)
    print_summary(table)
    table.to_csv(figures_dir / METRICS_FILENAME, index=False)
    print(f"Saved metrics summary -> {figures_dir / METRICS_FILENAME}")

    # ----- Save the better model (Model B) for the dashboard -----
    model_path = models_dir / MODEL_B_FILENAME
    joblib.dump(model_b, model_path)
    print(f"Saved Model B -> {model_path}")
    print("\nPhase 3 complete. [OK]")
    return table


def main(argv=None) -> None:
    argparse.ArgumentParser(
        description="Phase 3 A/B experiment: baseline vs GAN-augmented anomaly detector."
    ).parse_args(argv)
    run()


if __name__ == "__main__":
    main()
