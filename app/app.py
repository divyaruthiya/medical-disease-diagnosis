
import pickle
from pathlib import Path

import pandas as pd
import streamlit as st
from pgmpy.inference import VariableElimination

# -------------------- FILE PATHS --------------------
APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent

DATA_PATH = PROJECT_DIR / "dataset" / "final_symptom_disease.csv"
MODEL_PATH = PROJECT_DIR / "model" / "bayesian_network.pkl"

# -------------------- PAGE SETTINGS --------------------
st.set_page_config(
    page_title="Medical Disease Diagnosis",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------- CUSTOM DESIGN --------------------
st.markdown("""
<style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    .hero {
        padding: 28px;
        border-radius: 18px;
        background: linear-gradient(120deg, #102a43, #176b87);
        color: white;
        margin-bottom: 20px;
    }
    .hero h1 {
        color: white;
        margin-bottom: 8px;
    }
    .hero p {
        color: #e6f4f8;
        font-size: 16px;
    }
    .section-title {
        font-size: 23px;
        font-weight: 650;
        margin-top: 18px;
        margin-bottom: 8px;
    }
    div[data-testid="stMetric"] {
        background: var(--secondary-background-color);
        padding: 16px;
        border-radius: 12px;
        border: 1px solid rgba(128,128,128,0.2);
    }
    .footer {
        text-align: center;
        color: #888888;
        padding: 20px 0 5px 0;
        font-size: 13px;
    }
</style>
""", unsafe_allow_html=True)

# -------------------- HEADER --------------------
st.markdown("""
<div class="hero">
    <h1>🩺 Medical Disease Diagnosis</h1>
    <p>Explore symptom patterns using a Probabilistic Bayesian Network.</p>
    <p>Academic project | Artificial Intelligence and Machine Learning</p>
</div>
""", unsafe_allow_html=True)

st.warning(
    "Educational prototype only. Predictions are based on a limited "
    "dataset and have not been clinically validated. They are not a "
    "medical diagnosis and must not guide healthcare decisions."
)

# -------------------- SIDEBAR --------------------
with st.sidebar:
    st.title("🧭 Project Guide")
    st.write("**How to use this app**")
    st.write("1. Select one or more symptoms.")
    st.write("2. Click Calculate Probabilities.")
    st.write("3. Review the estimated results.")
    st.divider()
    st.write("**Model:** Bayesian Network")
    st.write("**Dataset:** Symptom–Disease CSV-Kaggle")
    st.divider()
    st.caption("Developed for academic learning.")

# -------------------- LOAD DATA AND MODEL --------------------
if not DATA_PATH.exists():
    st.error(f"Dataset file not found: {DATA_PATH}")
    st.stop()

if not MODEL_PATH.exists():
    st.error(f"Model file not found: {MODEL_PATH}")
    st.stop()

try:
    df = pd.read_csv(DATA_PATH)

    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)

    inference = VariableElimination(model)

except Exception as error:
    st.error(f"Could not load the model or dataset: {error}")
    st.stop()

symptom_columns = sorted(
    column for column in df.columns if column != "Disease"
)

# -------------------- SYMPTOM SELECTION --------------------
st.markdown(
    '<div class="section-title">🔎 Select Your Symptoms</div>',
    unsafe_allow_html=True
)
st.write(
    "Choose the symptoms you want to explore. "
    "You can search for symptoms in the selection box."
)

selected_symptoms = st.multiselect(
    "Available symptoms",
    options=symptom_columns,
    placeholder="Search and select symptoms...",
    help="Select only symptoms you want the model to consider."
)

if selected_symptoms:
    st.info(
        f"Selected {len(selected_symptoms)} symptom(s): "
        + ", ".join(selected_symptoms)
    )

calculate = st.button(
    "🧮 Calculate Probabilities",
    type="primary",
    use_container_width=True
)

# -------------------- RESULTS --------------------
if calculate:
    if not selected_symptoms:
        st.warning("Please select at least one symptom first.")

    else:
        with st.spinner("Calculating estimated probabilities..."):
            try:
                evidence = {
                    symptom: 1 for symptom in selected_symptoms
                }

                result = inference.query(
                    variables=["Disease"],
                    evidence=evidence,
                    show_progress=False
                )

                results_df = pd.DataFrame({
                    "Disease": result.state_names["Disease"],
                    "Probability (%)": result.values * 100
                })

                results_df = results_df.sort_values(
                    "Probability (%)",
                    ascending=False
                ).reset_index(drop=True)

                results_df["Probability (%)"] = (
                    results_df["Probability (%)"].round(2)
                )

                st.divider()
                st.markdown(
                    '<div class="section-title">📊 Estimated Results</div>',
                    unsafe_allow_html=True
                )

                st.caption(
                    "Rankings reflect this model and dataset only; "
                    "they do not establish the presence or absence of disease."
                )

                top_disease = results_df.iloc[0]["Disease"]
                top_probability = results_df.iloc[0]["Probability (%)"]

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        "Highest-ranked disease",
                        str(top_disease)
                    )

                with col2:
                    st.metric(
                        "Model estimate",
                        f"{top_probability:.2f}%"
                    )

                with col3:
                    st.metric(
                        "Symptoms selected",
                        len(selected_symptoms)
                    )

                st.subheader("Top 5 disease estimates")
                st.dataframe(
                    results_df.head(5),
                    use_container_width=True,
                    hide_index=True
                )

                st.subheader("Probability comparison")
                chart_data = results_df.head(5).set_index("Disease")
                st.bar_chart(chart_data["Probability (%)"])

                with st.expander("View all disease estimates"):
                    st.dataframe(
                        results_df,
                        use_container_width=True,
                        hide_index=True
                    )

            except Exception as error:
                st.error(
                    f"Could not calculate probabilities: {error}"
                )

# -------------------- FOOTER --------------------
st.divider()
st.markdown("""
<div class="footer">
    Medical Disease Diagnosis Using Bayesian Network<br>
    Academic project • For educational use only
</div>
""", unsafe_allow_html=True)
