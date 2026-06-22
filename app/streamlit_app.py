"""
streamlit_app.py — TeleGuard Phase 5: the dashboard.

Two tabs:
  🔍 Detect anomalies — upload logs, predict with Model B, adjustable risk
     threshold, live performance + confusion matrix, charts, AI explanations.
  📊 Experiment results — the Model A vs Model B A/B comparison (the project's
     core result), embedded from the saved metrics CSV and figures.

Run it (from the teleguard/ root, venv active):
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import io
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# --- Make `from src...` importable when run via `streamlit run app/...` ------ #
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import (  # noqa: E402
    COLUMN_NAMES,
    LABEL_COL,
    MODELS_DIR,
    get_feature_columns,
    load_preprocessor,
    preprocess_new_data,
)
from src.llm_explain import explain_anomaly  # noqa: E402

MODEL_B_PATH = MODELS_DIR / "classifier_model_b.pkl"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

KEY_EXPLAIN_FEATURES = [
    "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "logged_in", "num_failed_logins", "count", "srv_count",
    "serror_rate", "same_srv_rate", "dst_host_count",
]

MAX_STYLED_ROWS = 500  # cap rows in the colored table (Styler is slow on huge frames)

# Standard NSL-KDD attack -> family mapping (for the category breakdown).
ATTACK_FAMILIES = {
    "DoS": {"back", "land", "neptune", "pod", "smurf", "teardrop", "mailbomb",
            "apache2", "processtable", "udpstorm", "worm"},
    "Probe": {"ipsweep", "nmap", "portsweep", "satan", "mscan", "saint"},
    "R2L": {"ftp_write", "guess_passwd", "imap", "multihop", "phf", "spy",
            "warezclient", "warezmaster", "sendmail", "named", "snmpgetattack",
            "snmpguess", "xlock", "xsnoop", "httptunnel"},
    "U2R": {"buffer_overflow", "loadmodule", "perl", "rootkit", "ps",
            "sqlattack", "xterm"},
}


def attack_category(label: str) -> str:
    """Map a raw NSL-KDD label to its family (normal / DoS / Probe / R2L / U2R / other)."""
    lab = str(label).strip().lower()
    if lab == "normal":
        return "normal"
    for family, members in ATTACK_FAMILIES.items():
        if lab in members:
            return family
    return "other"


# --------------------------------------------------------------------------- #
# Cached loaders / compute (run once, reused across reruns)
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def load_artifacts():
    """Load the preprocessor bundle and Model B once, then reuse."""
    bundle = load_preprocessor()
    model = joblib.load(MODEL_B_PATH)
    return bundle, model


@st.cache_data(show_spinner=False)
def predict_file(file_bytes: bytes) -> pd.DataFrame:
    """Heavy step (read + preprocess + predict), cached on file content so that
    moving the risk-threshold slider does NOT recompute predictions."""
    bundle, model = load_artifacts()
    raw_df = load_uploaded_csv(io.BytesIO(file_bytes))
    return run_predictions(raw_df, bundle, model)


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
    """Preprocess, attach the anomaly probability, a default Result (0.5), and —
    when the file carries true labels — the ground-truth columns for evaluation."""
    X = preprocess_new_data(raw_df, bundle=bundle)        # encoded + scaled
    proba = model.predict_proba(X)[:, 1]                  # P(anomaly)

    out = raw_df.copy().reset_index(drop=True)
    out["anomaly_probability"] = np.round(proba, 4)
    out["Result"] = np.where(out["anomaly_probability"] >= 0.5, "ANOMALY", "normal")

    if LABEL_COL in raw_df.columns:                       # ground truth available
        lab = raw_df[LABEL_COL].astype(str).str.strip().reset_index(drop=True)
        out["actual"] = np.where(lab.str.lower() != "normal", "ANOMALY", "normal")
        out["attack_category"] = lab.map(attack_category).values
    return out


def apply_threshold(results: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Re-flag rows using a user-chosen probability threshold."""
    out = results.copy()
    out["Result"] = np.where(out["anomaly_probability"] >= threshold, "ANOMALY", "normal")
    return out


def eval_metrics(results: pd.DataFrame) -> dict | None:
    """Accuracy/precision/recall/F1 of the predictions vs the true labels."""
    if "actual" not in results.columns:
        return None
    y_true = (results["actual"] == "ANOMALY").astype(int)
    y_pred = (results["Result"] == "ANOMALY").astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "cm": confusion_matrix(y_true, y_pred, labels=[0, 1]),
    }


def select_key_features(row: pd.Series) -> dict:
    """Pick a small set of human-readable features to send to the LLM."""
    feats = {}
    for col in KEY_EXPLAIN_FEATURES:
        if col in row.index:
            val = row[col]
            if isinstance(val, (np.integer,)):
                val = int(val)
            elif isinstance(val, (np.floating,)):
                val = float(val)
            feats[col] = val
    return feats


def make_pie(results_df: pd.DataFrame):
    counts = results_df["Result"].value_counts()
    fig = px.pie(
        names=counts.index, values=counts.values, title="Normal vs Anomaly",
        color=counts.index,
        color_discrete_map={"normal": "#2a9d8f", "ANOMALY": "#e63946"}, hole=0.45,
    )
    fig.update_traces(textinfo="label+percent")
    return fig


def make_importance_fig(model, top_n: int = 15):
    names = list(getattr(model, "feature_names_in_", get_feature_columns()[0]))
    importances = np.asarray(model.feature_importances_)
    fi = (
        pd.DataFrame({"feature": names, "importance": importances})
        .sort_values("importance", ascending=False).head(top_n)
        .sort_values("importance")
    )
    fig = px.bar(
        fi, x="importance", y="feature", orientation="h",
        title=f"Top {top_n} features the model relies on",
        color="importance", color_continuous_scale="Teal",
    )
    fig.update_layout(coloraxis_showscale=False, yaxis_title="", xaxis_title="importance")
    return fig


def make_confusion_fig(cm):
    fig = px.imshow(
        cm, text_auto=True, color_continuous_scale="Blues", aspect="auto",
        x=["Predicted normal", "Predicted ANOMALY"],
        y=["Actual normal", "Actual ANOMALY"],
        title="Confusion matrix (this file)",
    )
    fig.update_layout(coloraxis_showscale=False)
    fig.update_xaxes(side="bottom")
    return fig


def make_category_fig(results: pd.DataFrame):
    """Among the TRUE attacks, how many did the model catch vs miss, per family."""
    if "attack_category" not in results.columns:
        return None
    atk = results[results["actual"] == "ANOMALY"].copy()
    if atk.empty:
        return None
    atk["caught"] = atk["Result"] == "ANOMALY"
    g = (atk.groupby("attack_category")
            .agg(total=("caught", "size"), caught=("caught", "sum")).reset_index())
    g["missed"] = g["total"] - g["caught"]
    melt = g.melt(id_vars="attack_category", value_vars=["caught", "missed"],
                  var_name="status", value_name="count")
    fig = px.bar(
        melt, x="attack_category", y="count", color="status", barmode="group",
        title="Attacks caught vs missed, by category",
        color_discrete_map={"caught": "#2a9d8f", "missed": "#e63946"},
    )
    fig.update_layout(xaxis_title="Attack family", yaxis_title="Number of attacks",
                      legend_title="")
    return fig


def style_results(display_df: pd.DataFrame):
    """Styler with readable DARK text on light row colors (anomaly = red)."""
    anomaly_style = "background-color: #ffd6d6; color: #7a0010; font-weight: 600"
    normal_style = "background-color: #e6f7f1; color: #14532d"

    def _row_style(row):
        style = anomaly_style if row.get("Result") == "ANOMALY" else normal_style
        return [style] * len(row)

    return display_df.style.apply(_row_style, axis=1)


# --------------------------------------------------------------------------- #
# Tab 1: Detect anomalies
# --------------------------------------------------------------------------- #
def render_detect(model) -> None:
    uploaded = st.file_uploader("Upload network logs (CSV)", type=["csv", "txt"])
    if uploaded is None:
        st.info("⬆️ Upload a CSV to begin. Tip: you can upload `data/raw/KDDTest+.txt`.")
        return

    try:
        with st.spinner("Scanning logs for anomalies..."):
            base_results = predict_file(uploaded.getvalue())
    except ValueError as exc:
        st.error(f"This file is missing required feature columns: {exc}")
        return
    except Exception as exc:
        st.error(f"Couldn't process that file: {exc}")
        return

    threshold = st.slider(
        "🎚️ Risk threshold — flag a connection when its anomaly probability is at least:",
        min_value=0.05, max_value=0.95, value=0.50, step=0.05,
        help="Lower = catch more anomalies but more false alarms (higher recall, "
             "lower precision). Higher = fewer false alarms but more misses.",
    )
    results = apply_threshold(base_results, threshold)
    has_truth = "actual" in results.columns

    n_total = len(results)
    n_anom = int((results["Result"] == "ANOMALY").sum())
    n_norm = n_total - n_anom

    if n_anom:
        st.error(f"⚠️ {n_anom:,} anomalies flagged out of {n_total:,} connections "
                 f"(at threshold {threshold:.2f}).")
    else:
        st.success(f"✅ No anomalies flagged across {n_total:,} connections "
                   f"(at threshold {threshold:.2f}).")

    c1, c2, c3 = st.columns(3)
    c1.metric("Total connections", f"{n_total:,}")
    c2.metric("Flagged normal", f"{n_norm:,}")
    c3.metric("Flagged anomalies", f"{n_anom:,}", delta=f"{100*n_anom/n_total:.1f}%")

    # Live model performance
    st.subheader("📈 Live model performance on this file")
    if has_truth:
        m = eval_metrics(results)
        st.caption("Predictions compared against the true labels in your file. "
                   "These update live as you move the risk threshold.")
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Accuracy", f"{m['accuracy']:.3f}")
        mc2.metric("Precision", f"{m['precision']:.3f}")
        mc3.metric("Recall", f"{m['recall']:.3f}")
        mc4.metric("F1-score", f"{m['f1']:.3f}")
        st.plotly_chart(make_confusion_fig(m["cm"]), use_container_width=True)
    else:
        st.info("No ground-truth `label` column found in this file, so live "
                "accuracy can't be computed. Upload a labelled NSL-KDD file "
                "(e.g. `KDDTest+.txt`) to see model performance.")

    # Charts
    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(make_pie(results), use_container_width=True)
    with col_b:
        st.plotly_chart(make_importance_fig(model), use_container_width=True)

    # Attack category breakdown
    if has_truth:
        cat_fig = make_category_fig(results)
        if cat_fig is not None:
            st.subheader("🗂️ Attack category breakdown")
            st.caption("Of the real attacks in this file, how many the model "
                       "caught vs missed in each NSL-KDD family. Rare families "
                       "such as U2R and R2L are the hardest to detect.")
            st.plotly_chart(cat_fig, use_container_width=True)

    # Results table
    st.subheader("Detection results")
    drop_cols = [c for c in ["difficulty"] if c in results.columns]
    view = results.drop(columns=drop_cols).sort_values(
        "anomaly_probability", ascending=False)
    if len(view) > MAX_STYLED_ROWS:
        st.caption(
            f"Showing the {MAX_STYLED_ROWS} highest-risk connections "
            f"(of {len(view):,}). Download below for the full results."
        )
        view = view.head(MAX_STYLED_ROWS)
    st.dataframe(style_results(view), use_container_width=True, height=360)

    dl1, dl2 = st.columns(2)
    dl1.download_button(
        "⬇️ Download full results (CSV)",
        results.to_csv(index=False).encode("utf-8"),
        file_name="teleguard_results.csv", mime="text/csv",
    )
    anomalies_only = results[results["Result"] == "ANOMALY"]
    dl2.download_button(
        "⬇️ Download flagged anomalies only (CSV)",
        anomalies_only.to_csv(index=False).encode("utf-8"),
        file_name="teleguard_anomalies.csv", mime="text/csv",
        disabled=anomalies_only.empty,
    )

    # AI explanations
    if n_anom:
        st.subheader("🧠 AI explanations for flagged anomalies")
        flagged = results[results["Result"] == "ANOMALY"].sort_values(
            "anomaly_probability", ascending=False)
        max_explain = st.slider(
            "How many of the highest-risk anomalies should the AI explain?",
            min_value=0, max_value=min(10, n_anom), value=min(3, n_anom),
            help="Each explanation is one LLM call. Kept small to control cost / "
                 "free-tier rate limits.",
        )
        for i, (idx, row) in enumerate(flagged.head(max_explain).iterrows()):
            feats = select_key_features(row)
            with st.expander(
                f"Anomaly #{i + 1} — row {idx}, risk {row['anomaly_probability']:.0%}",
                expanded=(i == 0),
            ):
                st.write("**Key features:**")
                st.json(feats, expanded=False)
                with st.spinner("Asking the AI analyst..."):
                    explanation = cached_explain(tuple(sorted(feats.items())))
                st.warning(explanation)


# --------------------------------------------------------------------------- #
# Tab 2: Experiment results (Model A vs Model B)
# --------------------------------------------------------------------------- #
def render_experiment() -> None:
    st.subheader("The A/B experiment: does GAN augmentation help?")
    st.caption(
        "Model A was trained on the ORIGINAL data; Model B on the original data "
        "PLUS CTGAN-synthetic anomalies. Both were evaluated on the SAME real "
        "KDDTest+ set (synthetic data never entered the test set)."
    )

    # --- Metrics table from the saved CSV ---
    csv_path = FIGURES_DIR / "ab_metrics_summary.csv"
    if csv_path.exists():
        raw = pd.read_csv(csv_path)
        pretty = {"f1_anomaly": "F1 (anomaly)", "precision_anomaly": "Precision",
                  "recall_anomaly": "Recall", "roc_auc": "ROC-AUC"}
        table = pd.DataFrame({
            "Metric": raw["metric"].map(lambda m: pretty.get(m, m)),
            "Model A (baseline)": raw["Model_A_baseline"].round(4),
            "Model B (augmented)": raw["Model_B_augmented"].round(4),
            "Change (B - A)": raw["delta_(B-A)"].round(4),
        })
        st.dataframe(table, use_container_width=True, hide_index=True)

        f1_delta = float(raw.loc[raw["metric"] == "f1_anomaly", "delta_(B-A)"].iloc[0])
        if f1_delta > 0:
            st.success("Verdict: Model B (GAN-augmented) improved on the baseline.")
        else:
            st.warning(
                "Verdict: Model B (GAN-augmented) did NOT beat the baseline — it was "
                "marginally worse on every metric."
            )
    else:
        st.info(f"`{csv_path.name}` not found. Run `python src/classifier.py` first "
                "to generate the A/B results.")

    # --- Figures ---
    def show(name: str, caption: str):
        path = FIGURES_DIR / name
        if path.exists():
            st.image(str(path), caption=caption, use_column_width=True)
        else:
            st.caption(f"(figure not found: {name} — run the earlier phases)")

    st.markdown("#### Metric comparison")
    show("metrics_comparison_AB.png", "Model A vs Model B across the four metrics.")

    colx, coly = st.columns(2)
    with colx:
        show("confusion_matrices_AB.png", "Confusion matrices on the identical test set.")
    with coly:
        show("roc_curves_AB.png", "ROC curves for both models.")

    st.markdown("#### GAN synthetic-data quality")
    colp, colq = st.columns(2)
    with colp:
        show("class_imbalance.png", "Original class balance (already ~balanced).")
    with colq:
        show("real_vs_synth_distributions.png", "Real vs CTGAN-synthetic distributions.")

    st.markdown("#### Conclusion")
    st.info(
        "On NSL-KDD the binary classes are already ~balanced (53.5% normal / "
        "46.5% anomaly), so GAN class-balancing had no real imbalance to fix and "
        "slightly reduced performance. GAN augmentation is expected to help where a "
        "genuine imbalance exists — for example the rare U2R and R2L attack "
        "categories (see the category breakdown in the Detect tab). That is the "
        "main direction for future work."
    )


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #
def main() -> None:
    st.set_page_config(page_title="TeleGuard", page_icon="📡", layout="wide")
    st.title("📡 TeleGuard — Telecom Anomaly Detection")
    st.caption(
        "Upload network logs and the GAN-augmented detector flags suspicious "
        "connections, with plain-English AI explanations."
    )

    with st.sidebar:
        st.header("How to use")
        st.markdown(
            "**🔍 Detect tab**\n"
            "1. Upload a **CSV of raw network logs** (e.g. `KDDTest+.txt`).\n"
            "2. Adjust the **risk threshold**.\n"
            "3. Read the live performance, charts, and AI explanations.\n\n"
            "**📊 Experiment Results tab**\n"
            "- The Model A vs Model B comparison (the project's core result)."
        )
        provider = os.getenv("LLM_PROVIDER", "anthropic")
        st.divider()
        st.caption(f"AI explanation provider: **{provider}**")

    # --- Load models (cached) ---
    try:
        _, model = load_artifacts()
    except FileNotFoundError as exc:
        st.error(
            "Model files not found. Run the earlier phases first:\n\n"
            "```\npython -m src.preprocessing\npython src/gan_train.py\n"
            "python src/classifier.py\n```\n\n"
            f"Details: {exc}"
        )
        st.stop()

    tab_detect, tab_experiment = st.tabs(
        ["🔍 Detect anomalies", "📊 Experiment Results (Model A vs Model B)"]
    )
    with tab_detect:
        render_detect(model)
    with tab_experiment:
        render_experiment()


if __name__ == "__main__":
    main()
