import streamlit as st
import joblib
import json
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
import gdown

DRIVE_FOLDER_URL = "https://drive.google.com/drive/folders/1LVaMkGMzFRb1_GiDlGTeHS3dX6Ulyi0v?usp=drive_link"
path = "toxicity_system"

if not os.path.exists(path):
    os.makedirs(path, exist_ok=True)
    gdown.download_folder(
        DRIVE_FOLDER_URL,
        output=path,
        quiet=False,
        use_cookies=False
    )

path = "toxicity_system"

@st.cache_resource
def load_system():
    tokenizer = AutoTokenizer.from_pretrained(f"{path}/chemberta_model")
    bert_model = AutoModelForSequenceClassification.from_pretrained(f"{path}/chemberta_model")
    vectorizer = joblib.load(f"{path}/tfidf_vectorizer.pkl")

    with open(f"{path}/config.json") as f:
        config = json.load(f)

    tox_cols = config["toxicity_tasks"]
    bert_w = config["bert_weight"]
    rf_w = config["rf_weight"]

    rf_models = []
    for task in tox_cols:
        rf_models.append(joblib.load(f"{path}/rf_{task}.pkl"))

    return tokenizer, bert_model, vectorizer, config, tox_cols, bert_w, rf_w, rf_models

tokenizer, bert_model, vectorizer, config, tox_cols, bert_w, rf_w, rf_models = load_system()

def predict_smiles(smiles):
    inputs = tokenizer(smiles, return_tensors="pt", truncation=True, padding=True, max_length=256)

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

    if max_risk >= 0.70:
        level = "HIGH RISK 🔴"
    elif max_risk >= 0.40:
        level = "MEDIUM RISK 🟡"
    elif max_risk >= 0.20:
        level = "LOW-MODERATE RISK 🟠"
    else:
        level = "LOW RISK 🟢"

    return final_probs, level, max_risk, avg_risk

st.title("🧪 AI Toxicity Screening System")

smiles = st.text_input("Enter SMILES:")

if st.button("Predict"):
    probs, level, max_risk, avg_risk = predict_smiles(smiles)

    st.write("### Result")
    st.write("Risk:", level)
    st.write("Max:", max_risk)
    st.write("Avg:", avg_risk)

    df = pd.DataFrame({
        "Endpoint": tox_cols,
        "Probability": probs
    }).sort_values("Probability", ascending=False)

    st.dataframe(df)

    st.warning("⚠️ This is Tox21 prediction only.")
