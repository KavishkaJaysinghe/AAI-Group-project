"""
gan_train.py — TeleGuard Phase 2: CTGAN synthetic anomaly generator.

This is the project's core contribution. We train a CTGAN (Conditional Tabular
GAN, via the SDV library) on the ANOMALY records only, then generate synthetic
anomalies to rebalance the training set. Phase 3 will show that a detector
trained on this balanced data catches rare anomalies better.

Pipeline
--------
1. Load processed training data (data/processed/train_processed.csv).
2. Keep only the minority class (target == 1, i.e. anomalies).
3. Train CTGAN on those anomalies.
4. Sample N synthetic anomalies (default N rebalances the dataset).
5. Save the trained model (models/) and synthetic data (data/processed/).
6. Validate quality: overlaid real-vs-synthetic distribution plots
   (reports/figures/) + a mean/std similarity summary + SDV quality score.

CLI
---
    python src/gan_train.py --n-samples 50000
    python src/gan_train.py --epochs 50            # quick test run
    python src/gan_train.py                        # auto-balance, 300 epochs

------------------------------------------------------------------------------
WHY CTGAN, MODE COLLAPSE, AND CONDITIONAL VECTORS  (read me!)
------------------------------------------------------------------------------
A GAN has a Generator (makes fake rows) and a Discriminator (real vs fake).

* MODE COLLAPSE: the failure mode where the Generator finds ONE (or a few)
  outputs that reliably fool the Discriminator and keeps producing those,
  ignoring the rest of the data's diversity. For us that would mean synthetic
  "anomalies" that are all nearly identical — useless for training a detector,
  because they don't cover the many *kinds* of attacks (neptune, smurf, ...).

* HOW CTGAN HELPS: CTGAN uses "conditional vectors" + "training-by-sampling".
  For each discrete column it picks a specific category, one-hot encodes it
  into a conditional vector, and forces the Generator to produce a row for THAT
  category — while sampling rare categories more often than their natural
  frequency. This pushes the Generator to cover ALL categories (not just the
  common ones), which directly fights mode collapse on imbalanced/categorical
  data like network-intrusion records. (CTGAN also uses mode-specific
  normalization for multi-modal continuous columns.)

------------------------------------------------------------------------------
TRAINING TIME WARNING
------------------------------------------------------------------------------
CTGAN is a deep model trained for many epochs.
* CPU:  SLOW. On the full NSL-KDD anomaly set (~58k rows) expect roughly
        ~30s-2min PER EPOCH depending on your CPU, so 300 epochs can take
        HOURS. For a first pass use --epochs 50 (and/or --max-train-rows).
* GPU:  Much faster (often 10-50x). Pass --enable-gpu if you have a CUDA GPU
        with PyTorch CUDA installed; otherwise it silently stays on CPU.
Tip: validate the pipeline with a small --epochs first, then do a long run.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: save figures without a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# --- Make `from src...` work whether run as `python src/gan_train.py` or
#     `python -m src.gan_train` ------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import (  # noqa: E402  (after sys.path tweak)
    CATEGORICAL_COLS,
    MODELS_DIR,
    PROCESSED_DIR,
    TARGET_COL,
    get_feature_columns,
)

TABLE_NAME = "anomalies"
SYNTH_FILENAME = "synthetic_anomalies.csv"
MODEL_FILENAME = "ctgan_anomaly.pkl"
DIST_FIG_NAME = "real_vs_synth_distributions.png"

# Numeric columns to showcase in the real-vs-synthetic plots (kept if present).
KEY_PLOT_COLUMNS = [
    "src_bytes", "dst_bytes", "duration", "count", "srv_count",
    "dst_host_count", "serror_rate", "same_srv_rate",
]


# --------------------------------------------------------------------------- #
# SDV imports + cross-version shims
# --------------------------------------------------------------------------- #
def _import_sdv():
    """Import SDV pieces, failing with an actionable message if missing."""
    try:
        from sdv.single_table import CTGANSynthesizer  # noqa: F401
    except Exception as exc:  # pragma: no cover - import-time guard
        raise ImportError(
            "SDV is not installed. Activate your venv and run:\n"
            "    pip install -r requirements.txt\n"
            f"(original error: {exc})"
        )
    return CTGANSynthesizer


def build_metadata(df: pd.DataFrame):
    """Detect metadata, then mark our known categorical columns as categorical.

    The processed data stores protocol_type/service/flag as INTEGER codes, so
    auto-detection would treat them as numerical. We override them to
    'categorical' so CTGAN's conditional vectors model them properly.
    Works across SDV versions (newer `Metadata`, older `SingleTableMetadata`).
    """
    metadata = None
    # Newer unified API
    try:
        from sdv.metadata import Metadata

        metadata = Metadata.detect_from_dataframe(data=df, table_name=TABLE_NAME)
    except Exception:
        metadata = None
    # Older single-table API
    if metadata is None:
        from sdv.metadata import SingleTableMetadata

        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(df)

    # Force the categorical sdtype on the columns we know are categorical.
    for col in CATEGORICAL_COLS:
        if col not in df.columns:
            continue
        try:
            metadata.update_column(column_name=col, sdtype="categorical")
        except TypeError:
            # Some versions need the table name for multi-table Metadata.
            metadata.update_column(
                column_name=col, sdtype="categorical", table_name=TABLE_NAME
            )
        except Exception:
            pass  # leave as-detected if the override isn't accepted
    return metadata


def make_synthesizer(metadata, epochs, batch_size, verbose, use_gpu):
    """Construct CTGANSynthesizer, tolerating the GPU-kwarg rename."""
    CTGANSynthesizer = _import_sdv()
    base = dict(epochs=epochs, batch_size=batch_size, verbose=verbose)
    # GPU flag was `cuda=` in older SDV, `enable_gpu=` in newer. Try both.
    for gpu_kw in ("enable_gpu", "cuda"):
        try:
            return CTGANSynthesizer(metadata, **base, **{gpu_kw: use_gpu})
        except TypeError:
            continue
    return CTGANSynthesizer(metadata, **base)  # no GPU kwarg accepted


def save_synthesizer(synth, path: Path):
    synth.save(filepath=str(path))


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_anomalies(processed_dir: Path) -> tuple[pd.DataFrame, int, int]:
    """Load processed train set; return (anomaly_features, n_normal, n_anomaly).

    CTGAN trains on FEATURES only — we drop the constant target column (all 1s
    in the minority subset) and re-attach target=1 after sampling.
    """
    train_path = processed_dir / "train_processed.csv"
    if not train_path.exists():
        raise FileNotFoundError(
            f"{train_path} not found. Run Phase 1 first:\n"
            "    python -m src.preprocessing"
        )
    df = pd.read_csv(train_path)
    if TARGET_COL not in df.columns:
        raise ValueError(f"'{TARGET_COL}' column missing from {train_path}")

    n_normal = int((df[TARGET_COL] == 0).sum())
    n_anomaly = int((df[TARGET_COL] == 1).sum())
    anomalies = df[df[TARGET_COL] == 1].drop(columns=[TARGET_COL]).reset_index(drop=True)
    return anomalies, n_normal, n_anomaly


# --------------------------------------------------------------------------- #
# Quality validation
# --------------------------------------------------------------------------- #
def plot_real_vs_synth(real: pd.DataFrame, synth: pd.DataFrame, out_path: Path):
    """Overlaid KDE/hist of real vs synthetic for several key numeric columns."""
    cols = [c for c in KEY_PLOT_COLUMNS if c in real.columns]
    if not cols:  # fall back to first few numeric columns
        cols = real.select_dtypes("number").columns.tolist()[:6]

    n = len(cols)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, col in zip(axes, cols):
        # KDE where possible; histogram is the robust fallback for spiky data.
        try:
            sns.kdeplot(real[col], ax=ax, label="real", fill=True, alpha=0.4)
            sns.kdeplot(synth[col], ax=ax, label="synthetic", fill=True, alpha=0.4)
        except Exception:
            ax.hist(real[col], bins=30, alpha=0.5, label="real", density=True)
            ax.hist(synth[col], bins=30, alpha=0.5, label="synthetic", density=True)
        ax.set_title(col)
        ax.set_xlabel("")
        ax.legend(fontsize=8)

    for ax in axes[len(cols):]:  # hide unused subplots
        ax.set_visible(False)

    fig.suptitle("Real vs Synthetic anomaly feature distributions", y=1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path, cols


def similarity_summary(real: pd.DataFrame, synth: pd.DataFrame, columns: list[str]):
    """Print a mean/std comparison table for the given columns."""
    rows = []
    for col in columns:
        rows.append({
            "column": col,
            "real_mean": real[col].mean(),
            "synth_mean": synth[col].mean(),
            "mean_diff": abs(real[col].mean() - synth[col].mean()),
            "real_std": real[col].std(),
            "synth_std": synth[col].std(),
        })
    summary = pd.DataFrame(rows)
    print("\n--- Real vs Synthetic: mean/std comparison (key columns) ---")
    with pd.option_context("display.float_format", "{:.4f}".format):
        print(summary.to_string(index=False))
    print(f"\nAverage |mean difference| across columns: "
          f"{summary['mean_diff'].mean():.4f}  (lower = more similar)")
    return summary


def sdv_quality_score(real, synth, metadata):
    """Optional: SDV's overall quality score in [0,1] (1 = identical stats)."""
    try:
        from sdv.evaluation.single_table import evaluate_quality

        report = evaluate_quality(real, synth, metadata)
        score = report.get_score()
        print(f"\nSDV quality score: {score:.4f}  (1.0 = perfect statistical match)")
        return score
    except Exception as exc:
        print(f"\n(SDV quality report unavailable — skipping. {exc})")
        return None


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run(
    n_samples: int | None,
    epochs: int,
    batch_size: int,
    use_gpu: bool,
    max_train_rows: int | None,
    processed_dir: Path = PROCESSED_DIR,
    models_dir: Path = MODELS_DIR,
    figures_dir: Path = PROJECT_ROOT / "reports" / "figures",
) -> None:
    processed_dir = Path(processed_dir)
    models_dir = Path(models_dir)
    figures_dir = Path(figures_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    # 1-2. Load processed data, keep anomalies only.
    anomalies, n_normal, n_anomaly = load_anomalies(processed_dir)
    print(f"Loaded anomalies for CTGAN: {anomalies.shape[0]:,} rows, "
          f"{anomalies.shape[1]} features")
    print(f"(original split: {n_normal:,} normal vs {n_anomaly:,} anomaly)")

    # Decide how many synthetic rows to make.
    if n_samples is None:
        # Default: rebalance so anomalies match the normal count.
        n_samples = max(n_normal - n_anomaly, n_anomaly)
        print(f"--n-samples not given -> auto-balancing to {n_samples:,} "
              f"synthetic anomalies")

    # Optional subsample to keep CPU training tractable.
    train_df = anomalies
    if max_train_rows and len(anomalies) > max_train_rows:
        train_df = anomalies.sample(max_train_rows, random_state=42).reset_index(drop=True)
        print(f"Subsampled training anomalies to {len(train_df):,} rows "
              f"(--max-train-rows)")

    # 3. Train CTGAN.
    metadata = build_metadata(train_df)
    synth = make_synthesizer(metadata, epochs, batch_size, verbose=True, use_gpu=use_gpu)
    print(f"\nTraining CTGAN: epochs={epochs}, batch_size={batch_size}, "
          f"gpu={'on' if use_gpu else 'off'}")
    print("(CPU training can be slow - see the time warning in this file's docstring.)")
    t0 = time.time()
    synth.fit(train_df)
    mins = (time.time() - t0) / 60
    print(f"CTGAN training done in {mins:.1f} min.")

    # 4. Sample synthetic anomalies.
    print(f"Sampling {n_samples:,} synthetic anomalies...")
    synthetic = synth.sample(num_rows=n_samples)

    # Clip numeric features back into the scaled [0,1] range and re-attach label.
    _, numeric_cols = get_feature_columns()
    num_in_synth = [c for c in numeric_cols if c in synthetic.columns]
    synthetic[num_in_synth] = synthetic[num_in_synth].clip(0.0, 1.0)
    synthetic[TARGET_COL] = 1  # every synthetic row is an anomaly

    # 5. Save model + synthetic data.
    model_path = models_dir / MODEL_FILENAME
    save_synthesizer(synth, model_path)
    synth_path = processed_dir / SYNTH_FILENAME
    synthetic.to_csv(synth_path, index=False)
    print(f"\nSaved CTGAN model    -> {model_path}")
    print(f"Saved synthetic data -> {synth_path}  shape={synthetic.shape}")

    # 6. Validate quality (plots + stats + SDV score).
    real_features = train_df
    synth_features = synthetic.drop(columns=[TARGET_COL])
    fig_path, plotted_cols = plot_real_vs_synth(
        real_features, synth_features, figures_dir / DIST_FIG_NAME
    )
    print(f"Saved distribution plot -> {fig_path}")
    similarity_summary(real_features, synth_features, plotted_cols)
    sdv_quality_score(real_features, synth_features, metadata)

    # Show the resulting balance if we concatenated synthetic into the train set.
    new_anomaly = n_anomaly + len(synthetic)
    print(f"\nProjected balance after augmentation: "
          f"{n_normal:,} normal vs {new_anomaly:,} anomaly "
          f"(was {n_anomaly:,}).")
    print("\nPhase 2 complete. [OK]")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train CTGAN on NSL-KDD anomalies and generate synthetic ones."
    )
    p.add_argument("--n-samples", type=int, default=None,
                   help="Number of synthetic anomalies to generate "
                        "(default: auto-balance vs the normal class).")
    p.add_argument("--epochs", type=int, default=300,
                   help="CTGAN training epochs (default 300; try 50 for a quick test).")
    p.add_argument("--batch-size", type=int, default=500,
                   help="CTGAN batch size (default 500; must be divisible by 10).")
    p.add_argument("--enable-gpu", action="store_true",
                   help="Use CUDA GPU if available (PyTorch CUDA required).")
    p.add_argument("--max-train-rows", type=int, default=None,
                   help="Cap the number of real anomalies used for training "
                        "(speeds up CPU runs).")
    return p


def main(argv=None) -> None:
    args = build_arg_parser().parse_args(argv)
    run(
        n_samples=args.n_samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        use_gpu=args.enable_gpu,
        max_train_rows=args.max_train_rows,
    )


if __name__ == "__main__":
    main()
