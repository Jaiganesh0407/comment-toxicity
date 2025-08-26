import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle, os
from tensorflow.keras.preprocessing.sequence import pad_sequences

MODEL_PATH = "toxicity_model_glove.h5"
TOKENIZER_PATH = "tokenizer.pickle"
LABELS = ["toxic","severe_toxic","obscene","threat","insult","identity_hate"]
MAX_LEN = 200

st.set_page_config(page_title="Toxic Comment Detection", page_icon="🛡️")
st.title("🛡️ Toxic Comment Detection (Keras H5 Model)")

@st.cache_resource
def load_tokenizer():
    with open(TOKENIZER_PATH, "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

tokenizer = load_tokenizer()
model = load_model()

def predict_texts(texts):
    seqs = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(seqs, maxlen=MAX_LEN, padding="post")
    preds = model.predict(padded)
    return preds

st.subheader("🔹 Test a Single Comment")
txt = st.text_area("Enter a comment:")
threshold = st.slider("Decision threshold", 0.0, 1.0, 0.5, 0.01)

if st.button("Predict"):
    if txt.strip():
        probs = predict_texts([txt])[0]
        preds = (probs >= threshold).astype(int)
        st.write("### Probabilities")
        for k, p in zip(LABELS, probs):
            st.write(f"- **{k}**: {float(p):.3f}")
        st.write("### Predicted labels")
        st.json({k:int(v) for k,v in zip(LABELS,preds)})
    else:
        st.warning("Please enter a comment.")

st.subheader("📂 Bulk Prediction from CSV")
up = st.file_uploader("Upload a CSV with 'comment_text' column", type=["csv"])
if up is not None:
    df = pd.read_csv(up)
    if "comment_text" not in df.columns:
        st.error("CSV must contain 'comment_text' column.")
    else:
        probs = predict_texts(df["comment_text"].astype(str).tolist())
        preds = (probs >= threshold).astype(int)
        probs_df = pd.DataFrame(probs, columns=[f"prob_{k}" for k in LABELS])
        preds_df = pd.DataFrame(preds, columns=LABELS)
        result = pd.concat([df.reset_index(drop=True), probs_df, preds_df], axis=1)
        st.dataframe(result.head(20))
        st.download_button("📥 Download predictions", result.to_csv(index=False).encode("utf-8"),
                           file_name="predictions.csv", mime="text/csv")
