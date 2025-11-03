import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import pennylane as qml
from sklearn.decomposition import PCA

st.set_page_config(page_title="Energy Forecasting App", layout="wide")
st.title("⚡ Household Power Consumption Forecasting")
st.write("Predict next-hour power consumption using ML, Deep Learning & Quantum Models")

@st.cache_resource
def load_all_models():
    models = {
        "Ridge": joblib.load("models/Ridge_model.pkl"),
        "RandomForest": joblib.load("models/RandomForest_model.pkl"),
        "XGBoost": joblib.load("models/XGBoost_model.pkl"),
    }
    dl_models = {
        "LSTM": load_model("models/LSTM.h5"),
        "GRU": load_model("models/GRU.h5"),
        "BiLSTM": load_model("models/BiLSTM.h5"),
        "CNN_LSTM": load_model("models/CNN_LSTM.h5"),
        "Transformer": load_model("models/Transformer.h5")
    }
    quantum_weights = joblib.load("models/quantum_weights.pkl")
    quantum_pca = joblib.load("models/quantum_pca.pkl")
    return models, dl_models, quantum_weights, quantum_pca

models, dl_models, q_weights, q_pca = load_all_models()

n_qubits = 8
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev)
def quantum_circuit(weights, x):
    qml.templates.AngleEmbedding(x, wires=range(n_qubits), rotation="X")
    qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return qml.expval(qml.PauliZ(0))

def quantum_predict(X):
    X = q_pca.transform(X)
    X = (X - X.min()) / (X.max() - X.min() + 1e-9) * (2 * np.pi) - np.pi
    return np.array([quantum_circuit(q_weights, xi) for xi in X])


st.subheader("Enter Last 24-Hour Power Usage")

DEFAULT_VALUES = [1.25,1.30,1.28,1.35,1.33,1.32,1.40,1.42,
                  1.38,1.50,1.48,1.55,1.52,1.45,1.40,1.38,
                  1.42,1.47,1.50,1.55,1.53,1.48,1.45,1.40]

user_input = st.text_area("Enter comma-separated values:", ",".join(map(str, DEFAULT_VALUES)))

if st.button("Predict"):
    try:
        values = list(map(float, user_input.split(",")))

        if len(values) < 24:
            values += [values[-1]] * (24 - len(values))
        values = values[:24]

        X_input = np.array(values).reshape(1, -1)
        X_rnn = np.array(values).reshape(1, 24, 1)

        predictions = {
            "Ridge": models["Ridge"].predict(X_input)[0],
            "Random Forest": models["RandomForest"].predict(X_input)[0],
            "XGBoost": models["XGBoost"].predict(X_input)[0],
            "LSTM": dl_models["LSTM"].predict(X_rnn)[0][0],
            "GRU": dl_models["GRU"].predict(X_rnn)[0][0],
            "BiLSTM": dl_models["BiLSTM"].predict(X_rnn)[0][0],
            "CNN-LSTM": dl_models["CNN_LSTM"].predict(X_rnn)[0][0],
            "Transformer": dl_models["Transformer"].predict(X_rnn)[0][0],
            "Quantum Model": quantum_predict(X_input)[0]
        }

        st.subheader("Model Predictions")
        results_df = pd.DataFrame(predictions.items(), columns=["Model", "Predicted Value"])
        st.table(results_df)

        st.subheader("Predictions Comparison")
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.bar(results_df["Model"], results_df["Predicted Value"])
        plt.xticks(rotation=45)
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error: {e}")
