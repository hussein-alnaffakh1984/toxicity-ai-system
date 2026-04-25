import os
import json
import joblib
import gdown
import torch
import numpy as np
import pandas as pd
import streamlit as st

from scipy.stats import entropy
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ===============================
# App Configuration
# ===============================
st.set_page_config(
    page_title="AI Toxicity Screening System",
    page_icon="🧪",
    layout="wide"
)

DRIVE_FOLDER_URL = "https://drive.google.com/drive/folders/1LVaMkGMzFRb1_GiDlGTeHS3dX6Ulyi0v?usp=drive_link"
MODEL_DIR = "toxicity_system"


# ===============================
# Download Model Files
# ===============================
@st.cache_resource
def download_files():
    if not os.path.exists(MODEL_DIR) or len(os.listdir(MODEL_DIR)) == 0:
        os.makedirs(MODEL_DIR, exist_ok=True)
        gdown.download_folder(
            DRIVE_FOLDER_URL,
            output=MODEL_DIR,
            quiet=False,
            use_cookies=False
        )


# ===============================
# Load System
# ===============================
@st.cache_resource
def load_system():
    download_files()

    tokenizer = AutoTokenizer.from_pretrained(f"{MODEL_DIR}/chemberta_model")
    bert_model = AutoModelForSequenceClassification.from_pretrained(f"{MODEL_DIR}/chemberta_model")
    bert_model.eval()

    vectorizer = joblib.load(f"{MODEL_DIR}/tfidf_vectorizer.pkl")

    with open(f"{MODEL_DIR}/config.json", "r") as f:
        config = json.load(f)

    tox_cols = config["toxicity_tasks"]
    bert_w = config["bert_weight"]
    rf_w = config["rf_weight"]

    rf_models = []
    for task in tox_cols:
        rf_models.append(joblib.load(f"{MODEL_DIR}/rf_{task}.pkl"))

    return tokenizer, bert_model, vectorizer, tox_cols, bert_w, rf_w, rf_models


tokenizer, bert_model, vectorizer, tox_cols, bert_w, rf_w, rf_models = load_system()


# ===============================
# Adaptive Risk Function
# ===============================
def compute_adaptive_risk(probs):
    max_risk = np.max(probs)
    mean_risk = np.mean(probs)
    var_risk = np.var(probs)

    eps = 1e-9
    ent = entropy(probs + eps)

    adaptive_score = (
        0.4 * max_risk +
        0.3 * mean_risk +
        0.2 * var_risk +
        0.1 * ent
    )

    return float(adaptive_score)


# ===============================
# Prediction Function
# ===============================
def predict_smiles(smiles):
    inputs = tokenizer(
        smiles,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )

    with torch.no_grad():
        outputs = bert_model(**inputs)

    bert_probs = torch.sigmoid(outputs.logits).numpy()[0]

    x_vec = vectorizer.transform([smiles])

    rf_probs = []
    for rf in rf_models:
        rf_probs.append(rf.predict_proba(x_vec)[0][1])

    rf_probs = np.array(rf_probs)

    final_probs = bert_w * bert_probs + rf_w * rf_probs

    max_risk = float(np.max(final_probs))
    avg_risk = float(np.mean(final_probs))
    adaptive_risk = compute_adaptive_risk(final_probs)

    if adaptive_risk >= 0.60:
        level = "HIGH RISK 🔴"
    elif adaptive_risk >= 0.40:
        level = "MEDIUM RISK 🟡"
    elif adaptive_risk >= 0.25:
        level = "LOW-MODERATE RISK 🟠"
    else:
        level = "LOW RISK 🟢"

    return final_probs, level, max_risk, avg_risk, adaptive_risk


# ===============================
# User Interface
# ===============================
st.title("🧪 AI Toxicity Screening System")
st.markdown(
    "### Multi-endpoint Tox21 toxicity prediction using ChemBERTa + TF-IDF Random Forest Ensemble"
)

st.info(
    "Enter a SMILES molecular string to predict toxicity probabilities across 12 Tox21 endpoints. "
    "The final decision uses an adaptive risk score based on maximum toxicity, average toxicity, variance, and entropy."
)

smiles = st.text_input(
    "Enter SMILES:",
    value="CCOc1ccc2nc(S(N)(=O)=O)sc2c1"
)

if st.button("Predict Toxicity"):
    if smiles.strip() == "":
        st.error("Please enter a valid SMILES string.")
    else:
        probs, level, max_risk, avg_risk, adaptive_risk = predict_smiles(smiles.strip())

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Overall Risk", level)
        col2.metric("Adaptive Risk", round(adaptive_risk, 3))
        col3.metric("Max Endpoint Toxicity", round(max_risk, 3))
        col4.metric("Average Toxicity", round(avg_risk, 3))

        results_df = pd.DataFrame({
            "Endpoint": tox_cols,
            "Predicted Probability": probs
        }).sort_values("Predicted Probability", ascending=False)

        st.subheader("Top Risk Endpoints")
        st.dataframe(results_df.head(3), use_container_width=True)

        st.subheader("All Tox21 Endpoints")
        st.dataframe(results_df, use_container_width=True)

        st.warning(
            "⚠️ This tool predicts Tox21 endpoint-related toxicity only. "
            "It is not a substitute for laboratory, clinical, or regulatory toxicology assessment."
        )

st.sidebar.header("Model Information")
st.sidebar.write("Model: ChemBERTa + TF-IDF Random Forest Ensemble")
st.sidebar.write("Tasks: 12 Tox21 endpoints")
st.sidebar.write(f"ChemBERTa weight: {bert_w}")
st.sidebar.write(f"Random Forest weight: {rf_w}")

st.sidebar.header("Decision Strategy")
st.sidebar.write("Adaptive Risk = 0.4×Max + 0.3×Mean + 0.2×Variance + 0.1×Entropy")
