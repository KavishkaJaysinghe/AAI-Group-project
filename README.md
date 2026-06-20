# 📡 TeleGuard

**Synthetic telecom anomaly data generation & detection using GANs.**

TeleGuard tests a single, focused hypothesis:

> Training an anomaly detector on a **GAN-augmented, class-balanced** telecom
> dataset significantly improves detection of **rare anomalies** compared to
> training on the original **imbalanced** data.

Real telecom anomaly datasets are heavily imbalanced — anomalies are rare, so
classifiers learn to ignore them. TeleGuard uses a **CTGAN** (via the SDV
library) to synthesize realistic minority-class (anomaly) records, rebalances
the training set, and shows the measurable lift in rare-anomaly recall.

---

## Pipeline stages

| Stage | Module | What it does |
|-------|--------|--------------|
| 1. Preprocess | `src/preprocessing.py` | Load raw data → clean → encode → train/test split → save to `data/processed/`. |
| 2. GAN augment | `src/gan_train.py` | Train CTGAN on minority class → generate synthetic anomalies → build class-balanced training set. |
| 3. Detect & evaluate | `src/classifier.py` | Train detectors on **original** vs **augmented** data → compare on the **same** test set (precision/recall/F1/PR-AUC, focus on rare-class recall). |
| 4. Explain | `src/llm_explain.py` | Use an LLM (Anthropic / OpenAI) to generate plain-language explanations for flagged anomalies. |
| 5. Dashboard | `app/streamlit_app.py` | Streamlit UI for metrics, flagged anomalies, and explanations. |

The **core experiment** is the Stage 3 A/B comparison: same model, same test
set, two different training sets (imbalanced vs GAN-balanced).

---

## Project structure

```
teleguard/
├── data/
│   ├── raw/             # original datasets (git-ignored)
│   └── processed/       # cleaned / split / augmented data (git-ignored)
├── notebooks/           # experiments & EDA (.ipynb)
├── src/
│   ├── preprocessing.py # Stage 1
│   ├── gan_train.py     # Stage 2
│   ├── classifier.py    # Stage 3
│   └── llm_explain.py   # Stage 4
├── models/              # saved GAN / classifier artifacts (git-ignored)
├── app/
│   └── streamlit_app.py # Stage 5 dashboard
├── reports/figures/     # generated plots (git-ignored)
├── requirements.txt     # pinned dependencies
├── .env.example         # template for API keys → copy to .env
└── .gitignore
```

> `data/`, `models/`, and `reports/figures/` are git-ignored (large /
> regenerable). Empty `.gitkeep` files preserve the folder layout.

---

## Setup

Requires **Python 3.10+**.

### Windows (PowerShell)

```powershell
cd teleguard
python -m venv venv
venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> If activation is blocked by execution policy, run once:
> `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`

### macOS / Linux (bash)

```bash
cd teleguard
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### API keys

```bash
cp .env.example .env      # Windows: copy .env.example .env
```

Then edit `.env` and add your free `GEMINI_API_KEY` from
[Google AI Studio](https://aistudio.google.com) (no billing required).
`ANTHROPIC_API_KEY` / `OPENAI_API_KEY` are optional fallbacks.

---

## How to run (once implemented)

```bash
# 1. Put a raw telecom dataset in data/raw/
# 2. Preprocess
python -m src.preprocessing
# 3. Train GAN & build balanced set
python -m src.gan_train
# 4. Train + compare detectors
python -m src.classifier
# 5. Launch dashboard
streamlit run app/streamlit_app.py
```

> ⚠️ The `src/` modules are currently **scaffolding stubs** — ML logic is added
> phase by phase.

---

## Tech stack

Python 3.10+ · pandas · numpy · scikit-learn · xgboost · imbalanced-learn ·
SDV (CTGAN) · matplotlib · seaborn · plotly · Google Gemini (free) / Anthropic / OpenAI · Streamlit
