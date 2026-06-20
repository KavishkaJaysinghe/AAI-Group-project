"""
streamlit_app.py — TeleGuard Phase 5: the dashboard.

Ties the whole pipeline together for a non-technical user:
  1. Upload a CSV of raw network logs (NSL-KDD format).
  2. Preprocess it with the SAVED encoders/scaler (preprocess_new_data).
  3. Load the saved GAN-augmented detector (Model B) and predict anomalies.
  4. Show a results table with flagged anomalies highlighted in red.
  5. Get a plain-English AI explanation for each flagged anomaly.
  6. Plotly charts: normal-vs-anomaly pie + model feature-importance bar.

Run it (from the teleguard/ root, venv active):
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# --- Make `from src...` importable when run via `streamlit run app/...` ------ #
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import (  # noqa: E402
    COLUMN_NAMES,
    MODELS_DIR,
    get_feature_columns,
    load_preprocessor,
    preprocess_new_data,
)
from src.llm_explain import explain_anomaly  # noqa: E402

MODEL_B_PATH = MODELS_DIR / "classifier_model_b.pkl"

# Human-readable columns we send to the LLM for each flagged anomaly
# (kept short so token usage stays modest).
KEY_EXPLAIN_FEATURES = [
    "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "logged_in", "num_failed_logins", "count", "srv_count",
    "serror_rate", "same_srv_rate", "dst_host_count",
]

MAX_STYLED_ROWS = 500  # cap rows in the colored table (Styler is slow on huge frames)


# --------------------------------------------------------------------------- #
# Cached loaders (load models once per session)
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def load_artifacts():
    """Load the preprocessor bundle and Model B once, then reuse."""
    bundle = load_preprocessor()
    model = joblib.load(MODEL_B_PATH)
    return bundle, model


@st.cache_data(show_spinner=False)
def cached_explain(features_items: tuple) -> str:
    """Cache LLM explanations so reruns don't re-call the API for the same row."""
    return explain_anomaly(dict(features_items))


# --------------------------------------------------------------------------- #
# Pure helpers (no Streamlit calls -> unit-testable)
# --------------------------------------------------------------------------- #
def load_uploaded_csv(file) -> pd.DataFrame:
    """Read an uploaded CSV, handling both headered and headerless NSL-KDD."""
    feature_cols, _ = get_feature_columns()

    df = pd.read_csv(file)
    if set(feature_cols).issubset(df.columns):
        return df  # already has proper headers

    # Otherwise assume a headerless NSL-KDD export; assign standard names.
    file.seek(0)
    raw = pd.read_csv(file, header=None)
    n = raw.shape[1]
    if n >= len(COLUMN_NAMES):                       # 43+: features + label + difficulty
        raw = raw.iloc[:, : len(COLUMN_NAMES)]
        raw.columns = COLUMN_NAMES
    elif n == len(COLUMN_NAMES) - 1:                 # 42: features + label
        raw.columns = COLUMN_NAMES[:-1]
    elif n == len(feature_cols):                     # 41: features only
        raw.columns = feature_cols
    else:
        raise ValueError(
            f"Unrecognized column count ({n}). Expected {len(feature_cols)} "
            f"feature columns (optionally + label/difficulty), or a header row "
            f"with the NSL-KDD feature names."
        )
    return raw


def run_predictions(raw_df: pd.DataFrame, bundle: dict, model) -> pd.DataFrame:
    """Preprocess raw logs, predict, and append Result + probability columns."""
    X = preprocess_new_data(raw_df, bundle=bundle)        # encoded + scaled
    preds = model.predict(X)
    proba = model.predict_proba(X)[:, 1]                  # P(anomaly)

    out = raw_df.copy().reset_index(drop=True)
    out["anomaly_probability"] = np.round(proba, 4)
    out["Result"] = np.where(preds == 1, "ANOMALY", "normal")
    return out


def select_key_features(row: pd.Series) -> dict:
    """Pick a small set of human-readable features to send to the LLM."""
    feats = {}
    for col in KEY_EXPLAIN_FEATURES:
        if col in row.index:
            val = row[col]
            # convert numpy scalars to plain python for clean prompts + caching
            if isinstance(val, (np.integer,)):
                val = int(val)
            elif isinstance(val, (np.floating,)):
                val = float(val)
            feats[col] = val
    return feats


def make_pie(results_df: pd.DataFrame):
    counts = results_df["Result"].value_counts()
    fig = px.pie(
        names=counts.index,
        values=counts.values,
        title="Normal vs Anomaly",
        color=counts.index,
        color_discrete_map={"normal": "#2a9d8f", "ANOMALY": "#e63946"},
        hole=0.45,
    )
    fig.update_traces(textinfo="label+percent")
    return fig


def make_importance_fig(model, top_n: int = 15):
    names = list(getattr(model, "feature_names_in_", get_feature_columns()[0]))
    importances = np.asarray(model.feature_importances_)
    fi = (
        pd.DataFrame({"feature": names, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(top_n)
        .sort_values("importance")  # ascending so the biggest bar is on top
    )
    fig = px.bar(
        fi, x="importance", y="feature", orientation="h",
        title=f"Top {top_n} features the model relies on",
        color="importance", color_continuous_scale="Teal",
    )
    fig.update_layout(coloraxis_showscale=False, yaxis_title="", xaxis_title="importance")
    return fig


def style_results(display_df: pd.DataFrame):
    """Return a pandas Styler with readable DARK text on light row colors.

    We set both the background AND the text color on every row. Setting only the
    background leaves the text at the theme default, which washes out (light text
    on a light tint) and is hard to read — so we pin a dark, high-contrast text
    color for both anomaly and normal rows.
    """
    anomaly_style = "background-color: #ffd6d6; color: #7a0010; font-weight: 600"
    normal_style = "background-color: #e6f7f1; color: #14532d"

    def _row_style(row):
        style = anomaly_style if row.get("Result") == "ANOMALY" else normal_style
        return [style] * len(row)

    return display_df.style.apply(_row_style, axis=1)


# --------------------------------------------------------------------------- #
# Streamlit UI
# --------------------------------------------------------------------------- #
def main() -> None:
    st.set_page_config(page_title="TeleGuard", page_icon="📡", layout="wide")
    st.title("📡 TeleGuard — Telecom Anomaly Detection")
    st.caption(
        "Upload network logs and the GAN-augmented detector flags suspicious "
        "connections, with plain-English AI explanations."
    )

    # --- Sidebar: instructions + status ---
    with st.sidebar:
        st.header("How to use")
        st.markdown(
            "1. Upload a **CSV of raw network logs** (NSL-KDD format — e.g. "
            "`KDDTest+.txt`).\n"
            "2. TeleGuard cleans the data, runs the detector, and flags anomalies.\n"
            "3. Expand a flagged anomaly to read the AI explanation."
        )
        provider = os.getenv("LLM_PROVIDER", "anthropic")
        st.divider()
        st.caption(f"AI explanation provider: **{provider}**")

    # --- Load models (cached) ---
    try:
        bundle, model = load_artifacts()
    except FileNotFoundError as exc:
        st.error(
            "Model files not found. Run the earlier phases first:\n\n"
            "```\npython -m src.preprocessing\npython src/gan_train.py\n"
            "python src/classifier.py\n```\n\n"
            f"Details: {exc}"
        )
        st.stop()

    # --- File upload ---
    uploaded = st.file_uploader("Upload network logs (CSV)", type=["csv", "txt"])
    if uploaded is None:
        st.info("⬆️ Upload a CSV to begin. Tip: you can upload `data/raw/KDDTest+.txt`.")
        return

    # --- Load + predict ---
    try:
        raw_df = load_uploaded_csv(uploaded)
    except Exception as exc:
        st.error(f"Couldn't read that file: {exc}")
        return

    try:
        with st.spinner("Scanning logs for anomalies..."):
            results = run_predictions(raw_df, bundle, model)
    except ValueError as exc:
        st.error(f"This file is missing required feature columns: {exc}")
        return
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        return

    n_total = len(results)
    n_anom = int((results["Result"] == "ANOMALY").sum())
    n_norm = n_total - n_anom

    # --- Banner ---
    if n_anom:
        st.error(f"⚠️ {n_anom:,} anomalies detected out of {n_total:,} connections.")
    else:
        st.success(f"✅ No anomalies detected across {n_total:,} connections.")

    # --- Metrics ---
    c1, c2, c3 = st.columns(3)
    c1.metric("Total connections", f"{n_total:,}")
    c2.metric("Normal", f"{n_norm:,}")
    c3.metric("Anomalies", f"{n_anom:,}", delta=f"{100*n_anom/n_total:.1f}%")

    # --- Charts ---
    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(make_pie(results), use_container_width=True)
    with col_b:
        st.plotly_chart(make_importance_fig(model), use_container_width=True)

    # --- Results table (anomalies highlighted) ---
    st.subheader("Detection results")
    view = results.sort_values("anomaly_probability", ascending=False)
    if len(view) > MAX_STYLED_ROWS:
        st.caption(
            f"Showing the {MAX_STYLED_ROWS} highest-risk connections "
            f"(of {len(view):,}). Download below for the full results."
        )
        view = view.head(MAX_STYLED_ROWS)
    st.dataframe(style_results(view), use_container_width=True, height=360)

    st.download_button(
        "⬇️ Download full results (CSV)",
        results.to_csv(index=False).encode("utf-8"),
        file_name="teleguard_results.csv",
        mime="text/csv",
    )

    # --- AI explanations for flagged anomalies ---
    if n_anom:
        st.subheader("🧠 AI explanations for flagged anomalies")
        flagged = results[results["Result"] == "ANOMALY"].sort_values(
            "anomaly_probability", ascending=False
        )
        max_explain = st.slider(
            "How many of the highest-risk anomalies should the AI explain?",
            min_value=0, max_value=min(10, n_anom), value=min(3, n_anom),
            help="Each explanation is one LLM call. Kept small to control cost / "
                 "free-tier rate limits.",
        )
        for i, (idx, row) in enumerate(flagged.head(max_explain).iterrows()):
            feats = select_key_features(row)
            with st.expander(
                f"Anomaly #{i + 1} — row {idx}, "
                f"risk {row['anomaly_probability']:.0%}",
                expanded=(i == 0),
            ):
                st.write("**Key features:**")
                st.json(feats, expanded=False)
                with st.spinner("Asking the AI analyst..."):
                    explanation = cached_explain(tuple(sorted(feats.items())))
                st.warning(explanation)


if __name__ == "__main__":
    main()
