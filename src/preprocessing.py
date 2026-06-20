"""
preprocessing.py — TeleGuard Phase 1: NSL-KDD data loading & preprocessing.

What this module does
---------------------
1. Loads the raw NSL-KDD .txt files (no header) with the standard column names.
2. Builds a BINARY target:  'normal' -> 0,  any attack -> 1 (anomaly).
3. Label-encodes the 3 categorical columns (protocol_type, service, flag) and
   SAVES the encoders so the exact same mapping is reused at inference time.
4. Min-max normalizes the numeric columns to [0, 1] (scaler is also saved).
5. Prints a clear class-imbalance report (counts + percentages).
6. Saves the processed data to data/processed/ as CSV and the fitted
   encoders + scaler to models/preprocessor.joblib.
7. Exposes preprocess_new_data(df) for the dashboard, which reuses the SAVED
   encoders/scaler and handles unseen categories gracefully.

Run it directly (after putting KDDTrain+.txt / KDDTest+.txt in data/raw/):
    python -m src.preprocessing
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# --------------------------------------------------------------------------- #
# Paths (resolved relative to this file, so it works from any working dir)
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
PREPROCESSOR_FILENAME = "preprocessor.joblib"

# Default raw file names (the standard NSL-KDD distribution)
DEFAULT_TRAIN_FILE = "KDDTrain+.txt"
DEFAULT_TEST_FILE = "KDDTest+.txt"

# --------------------------------------------------------------------------- #
# NSL-KDD schema: 41 feature columns + 'label' + 'difficulty' = 43 columns
# (This is the well-known, fixed NSL-KDD / KDD'99 column order.)
# --------------------------------------------------------------------------- #
COLUMN_NAMES = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
    "label",        # e.g. 'normal', 'neptune', 'smurf', ...
    "difficulty",   # NSL-KDD difficulty score (we drop this — not a feature)
]

# The 3 string/categorical features that need encoding.
CATEGORICAL_COLS = ["protocol_type", "service", "flag"]

# Column names we create / drop.
LABEL_COL = "label"          # original multi-class attack name (raw)
TARGET_COL = "target"        # our binary 0/1 target
DROP_COLS = ["difficulty"]   # not a predictive feature

# Code assigned to a category that was never seen during training.
UNKNOWN_CATEGORY_CODE = -1


# --------------------------------------------------------------------------- #
# Step 1 — Load raw data
# --------------------------------------------------------------------------- #
def load_raw(path: str | Path) -> pd.DataFrame:
    """Load a raw NSL-KDD .txt file (comma-separated, no header) with names."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Raw file not found: {path}\n"
            f"Download NSL-KDD and place KDDTrain+.txt / KDDTest+.txt in {RAW_DIR}"
        )
    # NSL-KDD is comma-separated with no header row.
    df = pd.read_csv(path, header=None, names=COLUMN_NAMES)
    return df


# --------------------------------------------------------------------------- #
# Step 2 — Binary target
# --------------------------------------------------------------------------- #
def make_binary_target(df: pd.DataFrame) -> pd.DataFrame:
    """Add TARGET_COL: 'normal' -> 0, everything else -> 1 (anomaly)."""
    df = df.copy()
    # .str.strip() guards against stray whitespace in the raw text.
    is_attack = df[LABEL_COL].astype(str).str.strip() != "normal"
    df[TARGET_COL] = is_attack.astype(int)
    return df


# --------------------------------------------------------------------------- #
# Helper — work out which columns are numeric features
# --------------------------------------------------------------------------- #
def get_feature_columns() -> tuple[list[str], list[str]]:
    """Return (all_feature_cols, numeric_feature_cols).

    Features = the 41 NSL-KDD inputs (we exclude label, difficulty, target).
    Numeric features = features that are NOT in CATEGORICAL_COLS.
    """
    non_features = set([LABEL_COL, TARGET_COL] + DROP_COLS)
    feature_cols = [c for c in COLUMN_NAMES if c not in non_features]
    numeric_cols = [c for c in feature_cols if c not in CATEGORICAL_COLS]
    return feature_cols, numeric_cols


# --------------------------------------------------------------------------- #
# Step 3 — Categorical encoding (with save/reuse + unseen-category safety)
# --------------------------------------------------------------------------- #
def fit_label_encoders(df: pd.DataFrame) -> dict[str, LabelEncoder]:
    """Fit one LabelEncoder per categorical column (on training data)."""
    encoders: dict[str, LabelEncoder] = {}
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        le.fit(df[col].astype(str))
        encoders[col] = le
    return encoders


def _safe_encode(le: LabelEncoder, values: pd.Series) -> pd.Series:
    """Transform with a fitted LabelEncoder, mapping UNSEEN values to -1.

    Plain ``le.transform`` raises on categories not seen during fit, which
    happens on the test set / new dashboard data. We map those to a reserved
    'unknown' code instead so inference never crashes.
    """
    # Map known class -> its integer code; unknown -> UNKNOWN_CATEGORY_CODE.
    class_to_code = {cls: i for i, cls in enumerate(le.classes_)}
    return (
        values.astype(str)
        .map(lambda v: class_to_code.get(v, UNKNOWN_CATEGORY_CODE))
        .astype(int)
    )


def apply_label_encoders(
    df: pd.DataFrame, encoders: dict[str, LabelEncoder]
) -> pd.DataFrame:
    """Replace categorical columns with their integer codes (unseen -> -1)."""
    df = df.copy()
    for col, le in encoders.items():
        df[col] = _safe_encode(le, df[col])
    return df


# --------------------------------------------------------------------------- #
# Step 4 — Numeric scaling
# --------------------------------------------------------------------------- #
def fit_scaler(df: pd.DataFrame, numeric_cols: list[str]) -> MinMaxScaler:
    """Fit a MinMaxScaler on the numeric columns (training data only)."""
    scaler = MinMaxScaler()
    scaler.fit(df[numeric_cols])
    return scaler


def apply_scaler(
    df: pd.DataFrame, scaler: MinMaxScaler, numeric_cols: list[str]
) -> pd.DataFrame:
    """Scale numeric columns to [0, 1] using an already-fitted scaler."""
    df = df.copy()
    df[numeric_cols] = scaler.transform(df[numeric_cols])
    return df


# --------------------------------------------------------------------------- #
# Step 5 — Class imbalance report
# --------------------------------------------------------------------------- #
def class_imbalance_report(df: pd.DataFrame, title: str = "Dataset") -> dict:
    """Print and return counts + percentages of normal vs anomaly."""
    counts = df[TARGET_COL].value_counts().sort_index()
    n_normal = int(counts.get(0, 0))
    n_anomaly = int(counts.get(1, 0))
    total = n_normal + n_anomaly

    pct_normal = 100.0 * n_normal / total if total else 0.0
    pct_anomaly = 100.0 * n_anomaly / total if total else 0.0
    # Imbalance ratio = majority : minority (e.g. "1.2 : 1").
    if n_anomaly and n_normal:
        ratio = max(n_normal, n_anomaly) / min(n_normal, n_anomaly)
    else:
        ratio = float("inf")

    print(f"\n=== Class imbalance report - {title} ===")
    print(f"  Total samples : {total:,}")
    print(f"  Normal  (0)   : {n_normal:>8,}  ({pct_normal:5.2f}%)")
    print(f"  Anomaly (1)   : {n_anomaly:>8,}  ({pct_anomaly:5.2f}%)")
    print(f"  Imbalance     : {ratio:.2f} : 1 (majority : minority)")
    print("=" * (28 + len(title)))

    return {
        "total": total,
        "normal": n_normal,
        "anomaly": n_anomaly,
        "pct_normal": pct_normal,
        "pct_anomaly": pct_anomaly,
        "imbalance_ratio": ratio,
    }


# --------------------------------------------------------------------------- #
# Save / load the fitted preprocessor (encoders + scaler + column metadata)
# --------------------------------------------------------------------------- #
def save_preprocessor(
    encoders: dict[str, LabelEncoder],
    scaler: MinMaxScaler,
    feature_cols: list[str],
    numeric_cols: list[str],
    models_dir: str | Path = MODELS_DIR,
) -> Path:
    """Bundle everything needed for inference into one joblib file."""
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    bundle = {
        "encoders": encoders,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "numeric_cols": numeric_cols,
        "categorical_cols": CATEGORICAL_COLS,
        "unknown_category_code": UNKNOWN_CATEGORY_CODE,
    }
    out_path = models_dir / PREPROCESSOR_FILENAME
    joblib.dump(bundle, out_path)
    return out_path


def load_preprocessor(models_dir: str | Path = MODELS_DIR) -> dict:
    """Load the saved preprocessor bundle."""
    path = Path(models_dir) / PREPROCESSOR_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"Preprocessor not found: {path}\n"
            f"Run `python -m src.preprocessing` first to fit and save it."
        )
    return joblib.load(path)


# --------------------------------------------------------------------------- #
# Step 7 — Reusable inference helper for the dashboard
# --------------------------------------------------------------------------- #
def preprocess_new_data(
    df: pd.DataFrame, bundle: dict | None = None
) -> pd.DataFrame:
    """Transform NEW raw data the exact same way as training data.

    Uses the SAVED encoders + scaler so the dashboard produces features that
    match what the model was trained on. Unseen categorical values become -1.

    Parameters
    ----------
    df : raw rows containing (at least) the 41 NSL-KDD feature columns.
         Extra columns like 'label'/'difficulty' are ignored.
    bundle : optional pre-loaded preprocessor; loaded from disk if None.

    Returns
    -------
    A DataFrame with exactly ``feature_cols`` (encoded + scaled), ready for the
    model's ``predict``.
    """
    if bundle is None:
        bundle = load_preprocessor()

    encoders = bundle["encoders"]
    scaler = bundle["scaler"]
    feature_cols = bundle["feature_cols"]
    numeric_cols = bundle["numeric_cols"]

    # Fail early with a helpful message if required columns are missing.
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Input is missing {len(missing)} required feature column(s): "
            f"{missing}"
        )

    out = df.copy()
    out = apply_label_encoders(out, encoders)   # unseen categories -> -1
    out = apply_scaler(out, scaler, numeric_cols)
    # Return columns in the exact training order (drop anything extra).
    return out[feature_cols]


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def _transform_split(
    df: pd.DataFrame,
    encoders: dict[str, LabelEncoder],
    scaler: MinMaxScaler,
    numeric_cols: list[str],
    feature_cols: list[str],
) -> pd.DataFrame:
    """Apply target + encoders + scaler to a split; return features + target."""
    df = make_binary_target(df)
    df = apply_label_encoders(df, encoders)
    df = apply_scaler(df, scaler, numeric_cols)
    return df[feature_cols + [TARGET_COL]]


def main(
    train_path: str | Path | None = None,
    test_path: str | Path | None = None,
    processed_dir: str | Path = PROCESSED_DIR,
    models_dir: str | Path = MODELS_DIR,
) -> None:
    """Full Phase-1 pipeline: fit on TRAIN, transform TEST, save everything."""
    train_path = Path(train_path) if train_path else RAW_DIR / DEFAULT_TRAIN_FILE
    test_path = Path(test_path) if test_path else RAW_DIR / DEFAULT_TEST_FILE
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    feature_cols, numeric_cols = get_feature_columns()

    # ----- TRAIN: load -> target -> FIT encoders & scaler -> transform -----
    print(f"Loading training data: {train_path}")
    train_raw = load_raw(train_path)
    train_raw = make_binary_target(train_raw)
    class_imbalance_report(train_raw, title="KDDTrain+ (original)")

    encoders = fit_label_encoders(train_raw)            # fit on TRAIN only
    scaler = fit_scaler(
        apply_label_encoders(train_raw, encoders), numeric_cols
    )                                                   # fit on TRAIN only

    train_proc = _transform_split(
        train_raw, encoders, scaler, numeric_cols, feature_cols
    )
    train_out = processed_dir / "train_processed.csv"
    train_proc.to_csv(train_out, index=False)
    print(f"Saved processed train -> {train_out}  shape={train_proc.shape}")

    # ----- Persist encoders + scaler for reuse -----
    pp_path = save_preprocessor(
        encoders, scaler, feature_cols, numeric_cols, models_dir
    )
    print(f"Saved preprocessor   -> {pp_path}")

    # ----- TEST (optional): reuse the SAME fitted encoders & scaler -----
    if test_path.exists():
        print(f"\nLoading test data: {test_path}")
        test_raw = load_raw(test_path)
        test_raw = make_binary_target(test_raw)
        class_imbalance_report(test_raw, title="KDDTest+ (original)")
        test_proc = _transform_split(
            test_raw, encoders, scaler, numeric_cols, feature_cols
        )
        test_out = processed_dir / "test_processed.csv"
        test_proc.to_csv(test_out, index=False)
        print(f"Saved processed test  -> {test_out}  shape={test_proc.shape}")
    else:
        print(f"\n(No test file at {test_path} — skipping. That's fine.)")

    print("\nPhase 1 preprocessing complete. [OK]")


if __name__ == "__main__":
    main()
