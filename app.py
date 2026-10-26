"""
app.py  —  Titanic Survival Prediction  |  Streamlit Web Application
=====================================================================

Sections
--------
  Home              – Project introduction
  Exploratory Analysis – EDA charts from the training dataset
  Model Performance – Metrics, comparison charts, confusion matrix
  Survival Prediction – Interactive passenger input → prediction
"""

import json
import pathlib
import sys
import warnings

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

# Make sure src/ is importable when running from the project root
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from src.preprocessing import load_dataset, compute_imputation_stats, impute_missing_values
from src.features import engineer_features, ALL_FEATURES
from predict import predict, build_input_dataframe

# ---------------------------------------------------------------------------
# Page config  (must be the very first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Titanic Survival Prediction",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = PROJECT_ROOT / "models"
METRICS_JSON = OUTPUT_DIR / "metrics.json"
METRICS_CSV = OUTPUT_DIR / "metrics_table.csv"
BEST_MODEL_TXT = OUTPUT_DIR / "best_model_name.txt"


# ---------------------------------------------------------------------------
# Custom CSS — clean light theme
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #f8fafc 0%, #e2e8f0 100%);
    border-right: 1px solid #cbd5e1;
}
[data-testid="stSidebar"] * { color: #1e293b !important; }
[data-testid="stSidebar"] .stRadio label { font-size: 0.95rem; }

/* Main area */
.stApp, .main, [data-testid="stAppViewContainer"] {
    background: #ffffff;
    color: #1e293b;
}
.stMarkdown, .stText, label, p, h1, h2, h3, h4, h5, h6 {
    color: #1e293b;
}
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
    border: 1px solid #cbd5e1;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    text-align: center;
}
.metric-card .value {
    font-size: 2rem;
    font-weight: 700;
    color: #38bdf8;
}
.metric-card .label {
    font-size: 0.8rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 0.2rem;
}

/* Hero banner */
.hero {
    background: linear-gradient(135deg, #0ea5e9 0%, #6366f1 50%, #8b5cf6 100%);
    border-radius: 16px;
    padding: 2.5rem 2rem;
    margin-bottom: 1.5rem;
    text-align: center;
}
.hero h1 { color: white; font-size: 2.2rem; font-weight: 700; margin: 0; }
.hero p  { color: rgba(255,255,255,0.85); font-size: 1.05rem; margin-top: 0.5rem; }

/* Section headers */
.section-header {
    font-size: 1.35rem;
    font-weight: 600;
    color: #1e293b;
    border-left: 4px solid #38bdf8;
    padding-left: 0.75rem;
    margin: 1.5rem 0 1rem;
}

/* Prediction result */
.result-survived {
    background: linear-gradient(135deg, #ecfdf5, #d1fae5);
    border: 2px solid #10b981;
    border-radius: 16px;
    padding: 1.5rem 2rem;
    text-align: center;
}
.result-survived h2 { color: #047857; font-size: 1.8rem; margin: 0; }

.result-not-survived {
    background: linear-gradient(135deg, #fef2f2, #fee2e2);
    border: 2px solid #ef4444;
    border-radius: 16px;
    padding: 1.5rem 2rem;
    text-align: center;
}
.result-not-survived h2 { color: #b91c1c; font-size: 1.8rem; margin: 0; }

/* Info / note box */
.note-box {
    background: #f8fafc;
    border: 1px solid #cbd5e1;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    color: #475569;
    font-size: 0.88rem;
    margin-top: 0.75rem;
}

/* DataFrames */
.stDataFrame { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

@st.cache_data
def load_train_data():
    """Load and process training data for EDA. Returns None if not found."""
    try:
        raw = load_dataset(DATA_DIR / "train.csv")
        stats = compute_imputation_stats(raw)
        clean = impute_missing_values(raw, stats)
        fe = engineer_features(clean)
        # Re-attach Survived from clean before engineer drops it
        fe["Survived"] = clean["Survived"].values
        return raw, clean, fe, stats
    except FileNotFoundError:
        return None, None, None, None


@st.cache_data
def load_metrics():
    """Load saved metrics from train.py output."""
    if not METRICS_JSON.exists():
        return None, None
    with open(METRICS_JSON) as f:
        metrics = json.load(f)
    if METRICS_CSV.exists():
        df = pd.read_csv(METRICS_CSV)
    else:
        df = None
    return metrics, df


def load_best_model_name() -> str:
    if BEST_MODEL_TXT.exists():
        return BEST_MODEL_TXT.read_text().strip()
    return "Unknown"


def make_fig(figsize=(8, 4)):
    """Create a figure with a light background matching the app theme."""
    fig, ax = plt.subplots(figsize=figsize, facecolor="#ffffff")
    ax.set_facecolor("#ffffff")
    ax.tick_params(colors="#475569")
    ax.xaxis.label.set_color("#475569")
    ax.yaxis.label.set_color("#475569")
    ax.title.set_color("#1e293b")
    for spine in ax.spines.values():
        spine.set_edgecolor("#cbd5e1")
    return fig, ax


def styled_chart(plot_fn, **kwargs):
    """Run a chart function and display via st.pyplot with tight layout."""
    fig = plot_fn(**kwargs)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.sidebar.markdown("""
<div style='text-align:center; padding: 1.2rem 0 0.5rem;'>
  <span style='font-size:3rem;'>🚢</span>
  <p style='font-weight:700; font-size:1.1rem; margin:0.3rem 0 0;'>Titanic ML</p>
  <p style='font-size:0.78rem; color:#64748b; margin:0;'>Survival Prediction</p>
</div>
<hr style='border-color:#cbd5e1; margin:1rem 0;'/>
""", unsafe_allow_html=True)

page = st.sidebar.radio(
    "Navigation",
    ["🏠 Home", "📊 Exploratory Analysis", "🏆 Model Performance", "🔮 Survival Prediction"],
    label_visibility="collapsed",
)

st.sidebar.markdown("<hr style='border-color:#cbd5e1;'/>", unsafe_allow_html=True)

raw_df, clean_df, fe_df, imp_stats = load_train_data()
metrics_dict, metrics_df = load_metrics()

# Dataset status indicator in sidebar
if raw_df is not None:
    st.sidebar.success(f"✅ Dataset loaded ({len(raw_df)} rows)")
else:
    st.sidebar.warning("⚠️ Dataset not found in data/")
    st.sidebar.caption("Place train.csv in data/ and run python train.py")

if metrics_df is not None:
    st.sidebar.success("✅ Trained model found")
else:
    st.sidebar.warning("⚠️ Run python train.py first")

# ---------------------------------------------------------------------------
#  PAGE: HOME
# ---------------------------------------------------------------------------
if page == "🏠 Home":
    st.markdown("""
    <div class="hero">
      <h1>🚢 Titanic Survival Prediction</h1>
      <p>A Machine Learning Mini Project — predicting passenger survival using supervised classification</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-card">
          <div class="value">4</div>
          <div class="label">ML Models Compared</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        nrows = len(raw_df) if raw_df is not None else "—"
        st.markdown(f"""
        <div class="metric-card">
          <div class="value">{nrows}</div>
          <div class="label">Training Passengers</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        ncols = "7" if raw_df is None else str(len(ALL_FEATURES))
        st.markdown(f"""
        <div class="metric-card">
          <div class="value">{ncols}</div>
          <div class="label">Features Used</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-header">About This Project</div>', unsafe_allow_html=True)

    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown("""
On **15 April 1912**, the RMS Titanic sank after striking an iceberg in the North Atlantic Ocean.
Of the **2,224 passengers and crew** aboard, only **710 survived**.

This project uses **supervised machine learning** to identify patterns in passenger data that
correlate with survival — such as sex, passenger class, age, and family size.

**Four classifiers** are trained and compared:
- Logistic Regression
- Decision Tree
- Random Forest *(usually the best performer)*
- Support Vector Machine

The complete pipeline — from raw CSV to interactive web prediction — is implemented in Python
using **scikit-learn, pandas, matplotlib, seaborn, and Streamlit**.
        """)
#     with c2:
#         st.markdown("""
# **Quick-Start Workflow**

# ```bash
# # 1. Create virtual environment
# python -m venv .venv

# # Windows activation
# .venv\\Scripts\\Activate.ps1

# # 2. Install dependencies
# pip install -r requirements.txt

# # 3. Place Kaggle files in data/
# #    train.csv  and  test.csv

# # 4. Train all models
# python train.py

# # 5. Launch web app
# streamlit run app.py
# ```
#         """)

    st.markdown('<div class="section-header">How to Use This App</div>', unsafe_allow_html=True)
    st.markdown("""
| Tab | What You Can Do |
|-----|----------------|
| 📊 **Exploratory Analysis** | Inspect the dataset, see survival patterns by gender/class/age |
| 🏆 **Model Performance** | Compare accuracy, precision, recall, and F1 across all models |
| 🔮 **Survival Prediction** | Enter your own passenger details and get an instant prediction |
    """)

    st.markdown('<div class="note-box">⚠️ <strong>Academic Disclaimer:</strong> Predictions are statistical estimates '
                'based on historical Titanic passenger data. They do not determine the actual fate of any individual '
                'or imply any moral judgment.</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
#  PAGE: EXPLORATORY ANALYSIS
# ---------------------------------------------------------------------------
elif page == "📊 Exploratory Analysis":
    st.markdown('<div class="hero" style="padding:1.5rem 2rem;">'
                '<h1 style="font-size:1.6rem;">📊 Exploratory Data Analysis</h1>'
                '<p>Understanding the Titanic dataset through visualisations</p></div>',
                unsafe_allow_html=True)

    if raw_df is None:
        st.error("Dataset not found. Please place `train.csv` in the `data/` folder and run `python train.py`.")
        st.stop()

    # Dataset overview
    st.markdown('<div class="section-header">Dataset Overview</div>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Passengers", len(raw_df))
    col2.metric("Survived", int(raw_df["Survived"].sum()))
    col3.metric("Features", len(raw_df.columns))
    survival_rate = raw_df["Survived"].mean() * 100
    col4.metric("Survival Rate", f"{survival_rate:.1f}%")

    with st.expander("📋 Show raw data sample (first 10 rows)"):
        st.dataframe(raw_df.head(10), use_container_width=True)

    with st.expander("📋 Dataset info & missing values"):
        missing = raw_df.isnull().sum()
        missing_pct = (raw_df.isnull().mean() * 100).round(2)
        info_df = pd.DataFrame({
            "Column": raw_df.columns,
            "dtype": raw_df.dtypes.astype(str).values,
            "Non-Null Count": raw_df.notna().sum().values,
            "Missing Count": missing.values,
            "Missing %": missing_pct.values,
        })
        st.dataframe(info_df, use_container_width=True)

    # Survival distribution
    st.markdown('<div class="section-header">Survival Distribution</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        fig, ax = make_fig(figsize=(5, 4))
        counts = raw_df["Survived"].value_counts()
        colors = ["#ef4444", "#10b981"]
        ax.pie(
            counts.values,
            labels=["Did Not Survive", "Survived"],
            colors=colors,
            autopct="%1.1f%%",
            startangle=90,
            wedgeprops={"edgecolor": "#ffffff", "linewidth": 2},
            textprops={"color": "#1e293b", "fontsize": 11},
        )
        ax.set_title("Overall Survival Rate", fontsize=13, fontweight="bold")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with col2:
        fig, ax = make_fig(figsize=(5, 4))
        survival_by_class = raw_df.groupby("Pclass")["Survived"].mean() * 100
        bars = ax.bar(
            [f"Class {c}" for c in survival_by_class.index],
            survival_by_class.values,
            color=["#38bdf8", "#818cf8", "#a78bfa"],
            edgecolor="#ffffff",
            linewidth=1.5,
        )
        ax.set_ylabel("Survival Rate (%)")
        ax.set_title("Survival Rate by Passenger Class", fontsize=13, fontweight="bold")
        ax.set_ylim(0, 100)
        for bar, val in zip(bars, survival_by_class.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{val:.1f}%", ha="center", va="bottom", color="#1e293b", fontsize=10)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # Gender analysis
    st.markdown('<div class="section-header">Survival by Gender</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        fig, ax = make_fig(figsize=(5, 4))
        gender_surv = raw_df.groupby("Sex")["Survived"].mean() * 100
        colors_g = {"female": "#f472b6", "male": "#38bdf8"}
        bars = ax.bar(
            gender_surv.index,
            gender_surv.values,
            color=[colors_g.get(g, "#818cf8") for g in gender_surv.index],
            edgecolor="#ffffff",
            linewidth=1.5,
            width=0.5,
        )
        ax.set_ylabel("Survival Rate (%)")
        ax.set_title("Survival Rate by Sex", fontsize=13, fontweight="bold")
        ax.set_ylim(0, 100)
        for bar, val in zip(bars, gender_surv.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{val:.1f}%", ha="center", va="bottom", color="#1e293b", fontsize=11)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with col2:
        fig, ax = make_fig(figsize=(5, 4))
        gender_class = raw_df.groupby(["Pclass", "Sex"])["Survived"].mean().unstack()
        x = np.arange(len(gender_class.index))
        width = 0.35
        bars_f = ax.bar(x - width / 2, gender_class.get("female", 0) * 100,
                        width, label="Female", color="#f472b6", edgecolor="#ffffff")
        bars_m = ax.bar(x + width / 2, gender_class.get("male", 0) * 100,
                        width, label="Male", color="#38bdf8", edgecolor="#ffffff")
        ax.set_xticks(x)
        ax.set_xticklabels([f"Class {c}" for c in gender_class.index])
        ax.set_ylabel("Survival Rate (%)")
        ax.set_title("Survival by Class & Gender", fontsize=13, fontweight="bold")
        ax.set_ylim(0, 100)
        ax.legend(facecolor="#ffffff", edgecolor="#cbd5e1", labelcolor="#1e293b")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # Age distribution
    st.markdown('<div class="section-header">Age Distribution</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        fig, ax = make_fig(figsize=(6, 4))
        survived_ages = raw_df[raw_df["Survived"] == 1]["Age"].dropna()
        not_survived_ages = raw_df[raw_df["Survived"] == 0]["Age"].dropna()
        ax.hist(not_survived_ages, bins=30, alpha=0.7, color="#ef4444",
                label="Did Not Survive", edgecolor="#ffffff", linewidth=0.5)
        ax.hist(survived_ages, bins=30, alpha=0.7, color="#10b981",
                label="Survived", edgecolor="#ffffff", linewidth=0.5)
        ax.set_xlabel("Age")
        ax.set_ylabel("Passenger Count")
        ax.set_title("Age Distribution by Survival", fontsize=13, fontweight="bold")
        ax.legend(facecolor="#ffffff", edgecolor="#cbd5e1", labelcolor="#1e293b")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with col2:
        fig, ax = make_fig(figsize=(6, 4))
        survived_fares = raw_df[raw_df["Survived"] == 1]["Fare"].dropna()
        not_survived_fares = raw_df[raw_df["Survived"] == 0]["Fare"].dropna()
        ax.hist(not_survived_fares.clip(upper=200), bins=30, alpha=0.7,
                color="#ef4444", label="Did Not Survive", edgecolor="#ffffff", linewidth=0.5)
        ax.hist(survived_fares.clip(upper=200), bins=30, alpha=0.7,
                color="#10b981", label="Survived", edgecolor="#ffffff", linewidth=0.5)
        ax.set_xlabel("Fare (£) — clipped at 200")
        ax.set_ylabel("Passenger Count")
        ax.set_title("Fare Distribution by Survival", fontsize=13, fontweight="bold")
        ax.legend(facecolor="#ffffff", edgecolor="#cbd5e1", labelcolor="#1e293b")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # Embarkation
    st.markdown('<div class="section-header">Embarkation Port</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    PORT_NAMES = {"S": "Southampton", "C": "Cherbourg", "Q": "Queenstown"}
    with col1:
        fig, ax = make_fig(figsize=(5, 4))
        embarked_counts = raw_df["Embarked"].value_counts()
        embarked_counts.index = [PORT_NAMES.get(p, p) for p in embarked_counts.index]
        ax.bar(embarked_counts.index, embarked_counts.values,
               color=["#38bdf8", "#a78bfa", "#34d399"], edgecolor="#ffffff")
        ax.set_ylabel("Number of Passengers")
        ax.set_title("Passengers by Embarkation Port", fontsize=13, fontweight="bold")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with col2:
        fig, ax = make_fig(figsize=(5, 4))
        emb_surv = raw_df.groupby("Embarked")["Survived"].mean() * 100
        emb_surv.index = [PORT_NAMES.get(p, p) for p in emb_surv.index]
        bars = ax.bar(emb_surv.index, emb_surv.values,
                      color=["#38bdf8", "#a78bfa", "#34d399"], edgecolor="#ffffff")
        ax.set_ylabel("Survival Rate (%)")
        ax.set_title("Survival Rate by Embarkation Port", fontsize=13, fontweight="bold")
        ax.set_ylim(0, 80)
        for bar, val in zip(bars, emb_surv.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{val:.1f}%", ha="center", va="bottom", color="#1e293b", fontsize=10)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # Key observations
    st.markdown('<div class="section-header">Key Observations</div>', unsafe_allow_html=True)
    female_rate = raw_df[raw_df["Sex"] == "female"]["Survived"].mean() * 100
    male_rate = raw_df[raw_df["Sex"] == "male"]["Survived"].mean() * 100
    class1_rate = raw_df[raw_df["Pclass"] == 1]["Survived"].mean() * 100
    class3_rate = raw_df[raw_df["Pclass"] == 3]["Survived"].mean() * 100

    st.info(
        f"📌 **Gender gap**: Female survival rate ({female_rate:.1f}%) was "
        f"{female_rate - male_rate:.1f} percentage points higher than male ({male_rate:.1f}%) — "
        f"consistent with the 'women and children first' evacuation policy.\n\n"
        f"📌 **Class disparity**: 1st-class passengers survived at {class1_rate:.1f}% vs "
        f"{class3_rate:.1f}% for 3rd-class — access to lifeboats was strongly correlated with "
        f"social class.\n\n"
        f"📌 Use the charts above to inspect the age-group relationship; the displayed rates are "
        f"calculated directly from the available rows."
    )


# ---------------------------------------------------------------------------
#  PAGE: MODEL PERFORMANCE
# ---------------------------------------------------------------------------
elif page == "🏆 Model Performance":
    st.markdown('<div class="hero" style="padding:1.5rem 2rem;">'
                '<h1 style="font-size:1.6rem;">🏆 Model Performance</h1>'
                '<p>Comparing four classifiers on the validation split</p></div>',
                unsafe_allow_html=True)

    if metrics_df is None:
        st.error("No model metrics found. Please run `python train.py` first.")
        st.stop()

    best_name = load_best_model_name()

    # Summary metrics cards for best model
    best_row = metrics_df[metrics_df["Model"] == best_name]
    if not best_row.empty:
        st.markdown(f'<div class="section-header">Best Model: {best_name}</div>',
                    unsafe_allow_html=True)
        cols = st.columns(4)
        for col, metric_name in zip(cols, ["Accuracy", "Precision", "Recall", "F1 Score"]):
            val = float(best_row[metric_name].values[0])
            col.metric(metric_name, f"{val:.4f}")

    # Comparison table
    st.markdown('<div class="section-header">All Models — Comparison Table</div>',
                unsafe_allow_html=True)

    def highlight_best(row):
        is_best = row["Model"] == best_name
        return ["background-color: #dcfce7; color: #166534" if is_best else "" for _ in row]

    styled = metrics_df.style.apply(highlight_best, axis=1).format({
        "Accuracy": "{:.4f}",
        "Precision": "{:.4f}",
        "Recall": "{:.4f}",
        "F1 Score": "{:.4f}",
    })
    st.dataframe(styled, use_container_width=True, hide_index=True)

    # Comparison chart
    st.markdown('<div class="section-header">Metric Comparison Chart</div>',
                unsafe_allow_html=True)
    comp_img = OUTPUT_DIR / "model_comparison.png"
    if comp_img.exists():
        st.image(str(comp_img), use_column_width=True)
    else:
        st.warning("Chart not found. Run `python train.py` to generate it.")

    # Confusion matrices
    st.markdown('<div class="section-header">Confusion Matrices</div>', unsafe_allow_html=True)
    cm_files = list(OUTPUT_DIR.glob("confusion_matrix_*.png"))
    if cm_files:
        cols = st.columns(min(len(cm_files), 2))
        for i, cm_file in enumerate(sorted(cm_files)):
            with cols[i % 2]:
                st.image(str(cm_file), use_column_width=True)
    else:
        st.warning("Confusion matrix images not found. Run `python train.py`.")

    # Feature importance
    fi_img = OUTPUT_DIR / "feature_importance.png"
    if fi_img.exists():
        st.markdown('<div class="section-header">Feature Importances (Best Model)</div>',
                    unsafe_allow_html=True)
        st.image(str(fi_img), use_column_width=True)

    # Model explanation
    st.markdown('<div class="section-header">Why Was This Model Selected?</div>',
                unsafe_allow_html=True)
    explanation = {
        "Random Forest": (
            "**Random Forest** is an ensemble of many decision trees. Each tree is trained on a "
            "random bootstrap sample of the data, and features are selected randomly at each "
            "split. The final prediction is a majority vote. This reduces overfitting compared "
            "to a single Decision Tree while maintaining high accuracy. It is robust to outliers "
            "and handles mixed feature types well."
        ),
        "Logistic Regression": (
            "**Logistic Regression** models the probability of survival using a sigmoid function. "
            "It is highly interpretable — you can inspect the learned weights to understand which "
            "features matter most. The main limitation is that it assumes a linear decision "
            "boundary, which may not capture complex interactions."
        ),
        "Decision Tree": (
            "**Decision Tree** splits data recursively on the feature that reduces impurity the "
            "most. It is very interpretable (can be drawn as a tree diagram) but prone to "
            "overfitting unless depth is limited."
        ),
        "SVM": (
            "**Support Vector Machine** finds the hyperplane that maximises the margin between "
            "classes. The RBF kernel allows it to model non-linear boundaries. It can be "
            "effective on smaller datasets but is harder to interpret."
        ),
    }
    st.markdown(explanation.get(best_name,
                f"The model **{best_name}** was selected based on the highest F1 score on "
                f"the validation split."))


# ---------------------------------------------------------------------------
#  PAGE: SURVIVAL PREDICTION
# ---------------------------------------------------------------------------
elif page == "🔮 Survival Prediction":
    st.markdown('<div class="hero" style="padding:1.5rem 2rem;">'
                '<h1 style="font-size:1.6rem;">🔮 Survival Prediction</h1>'
                '<p>Enter passenger details to receive a model-based survival estimate</p></div>',
                unsafe_allow_html=True)

    # Check model exists
    best_name_txt = OUTPUT_DIR / "best_model_name.txt"
    if not best_name_txt.exists():
        st.error("No trained model found. Please run `python train.py` first, then restart the app.")
        st.stop()

    st.markdown('<div class="section-header">Passenger Details</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        pclass = st.selectbox(
            "Passenger Class",
            options=[1, 2, 3],
            format_func=lambda x: f"{x}st Class" if x == 1 else (f"{x}nd Class" if x == 2 else "3rd Class"),
            help="1st = Upper deck, higher fare; 3rd = Lower deck, lowest fare",
        )
        sex = st.selectbox("Sex", ["male", "female"])
        age = st.slider("Age", min_value=1, max_value=80, value=28,
                        help="Passenger's age in years")

    with col2:
        sibsp = st.number_input(
            "Siblings / Spouses Aboard",
            min_value=0, max_value=10, value=0, step=1,
            help="Number of siblings or spouses travelling with this passenger",
        )
        parch = st.number_input(
            "Parents / Children Aboard",
            min_value=0, max_value=10, value=0, step=1,
            help="Number of parents or children travelling with this passenger",
        )
        fare = st.number_input(
            "Fare Paid (£)",
            min_value=0.0, max_value=600.0, value=30.0, step=1.0,
            help="Ticket price in British pounds",
        )

    with col3:
        embarked = st.selectbox(
            "Port of Embarkation",
            options=["S", "C", "Q"],
            format_func=lambda x: {"S": "Southampton (S)", "C": "Cherbourg (C)", "Q": "Queenstown (Q)"}[x],
        )

        family_size = int(sibsp) + int(parch) + 1
        alone = "Yes" if family_size == 1 else "No"
        st.markdown(f"""
        <div class="note-box">
          <b>Derived features:</b><br>
          Family Size = {family_size}<br>
          Travelling Alone = {alone}
        </div>
        """, unsafe_allow_html=True)

    # Predict button
    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🔮 Predict Survival", type="primary", use_container_width=True)

    if predict_btn:
        passenger = {
            "Pclass": pclass,
            "Sex": sex,
            "Age": age,
            "SibSp": int(sibsp),
            "Parch": int(parch),
            "Fare": fare,
            "Embarked": embarked,
        }

        with st.spinner("Running model inference …"):
            try:
                result = predict(passenger)
            except FileNotFoundError as e:
                st.error(str(e))
                st.stop()
            except ValueError as e:
                st.error(f"Invalid passenger input: {e}")
                st.stop()

        st.markdown("<br>", unsafe_allow_html=True)

        if result["prediction"] == 1:
            st.markdown(f"""
            <div class="result-survived">
              <h2>✅ Predicted: SURVIVED</h2>
              <p style="color:#047857; margin:0.5rem 0 0; font-size:1rem;">
                Model confidence: {result['probability']*100:.1f} %
              </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-not-survived">
              <h2>❌ Predicted: DID NOT SURVIVE</h2>
              <p style="color:#b91c1c; margin:0.5rem 0 0; font-size:1rem;">
                Model confidence: {result['probability']*100:.1f} %
              </p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="note-box">
          <b>Model used:</b> {result['model_used'].replace('_', ' ').title()}<br><br>
          ⚠️ <b>Important:</b> This is a statistical estimate based on patterns in historical
          Titanic passenger data. It reflects aggregate probabilities across similar passengers —
          not a certainty about any individual's fate. Use for educational purposes only.
        </div>
        """, unsafe_allow_html=True)

        # Show input summary
        with st.expander("📋 Passenger summary used for prediction"):
            summary = pd.DataFrame([{
                "Passenger Class": f"{pclass}{'st' if pclass==1 else 'nd' if pclass==2 else 'rd'}",
                "Sex": sex.capitalize(),
                "Age": age,
                "Siblings/Spouses": int(sibsp),
                "Parents/Children": int(parch),
                "Family Size": family_size,
                "Travelling Alone": alone,
                "Fare (£)": fare,
                "Embarked": {"S": "Southampton", "C": "Cherbourg", "Q": "Queenstown"}[embarked],
            }])
            st.dataframe(summary.T.rename(columns={0: "Value"}), use_container_width=True)
