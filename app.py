import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import shap
import time

st.set_page_config(page_title="Fraud Detection Pro", layout="wide")

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    data = pickle.load(open("best_model.pkl", "rb"))
    return data["params"], data["scaler"]

params, scaler = load_model()

# =========================
# NN FUNCTIONS
# =========================
def sigmoid(z): return 1/(1+np.exp(-z))
def relu(z): return np.maximum(0,z)

def forward(X):
    A = X
    L = len(params)//2
    for l in range(1, L):
        A = relu(np.dot(params[f"W{l}"], A) + params[f"B{l}"])
    return sigmoid(np.dot(params[f"W{L}"], A) + params[f"B{L}"])

def predict(X, threshold=0.5):
    probs = forward(X)
    return (probs > threshold).astype(int), probs

# =========================
# SHAP
# =========================
def shap_explain(X_sample):
    def model_fn(x):
        return forward(x.T).flatten()
    explainer = shap.Explainer(model_fn, np.zeros((1, X_sample.shape[0])))
    return explainer(X_sample.T)

# =========================
# RISK
# =========================
def risk_level(prob):
    if prob > 0.8:
        return "🚨 HIGH FRAUD RISK"
    elif prob > 0.5:
        return "⚠️ MEDIUM RISK"
    else:
        return "✅ SAFE"

# =========================
# TITLE
# =========================
st.title("💳 Smart Fraud Detection System")

threshold = st.sidebar.slider("Threshold", 0.0, 1.0, 0.5)

# =========================
# DATASET DASHBOARD
# =========================
st.sidebar.subheader("Dataset Dashboard")
dataset_file = st.sidebar.file_uploader("Upload Dataset", type=["csv"])

if dataset_file:
    df = pd.read_csv(dataset_file)

    st.subheader("📊 Dataset Overview")
    st.write(df.head())

    if "Class" in df.columns:
        st.metric("Fraud Rate", f"{df['Class'].mean():.4f}")

        fig, ax = plt.subplots()
        df["Class"].value_counts().plot(kind="bar", ax=ax)
        ax.set_title("Fraud vs Legit")
        st.pyplot(fig)

# =========================
# SMART INPUT
# =========================
st.subheader("🧠 Smart Transaction Input")

col1, col2 = st.columns(2)

with col1:
    amount = st.number_input("Amount", value=100.0)
    txn_type = st.selectbox("Transaction Type", ["Online", "POS", "ATM"])

with col2:
    location = st.selectbox("Location", ["Same City", "Different City"])
    time_period = st.selectbox("Time", ["Day", "Night"])

# =========================
# FEATURE ENGINEERING
# =========================
def create_features(amount, txn_type, location, time_period):
    X = np.zeros(scaler.mean_.shape[0])

    X[0] = amount / 1000

    if txn_type == "Online":
        X[1] = 1
    elif txn_type == "ATM":
        X[2] = 1

    if location == "Different City":
        X[3] = 1

    if time_period == "Night":
        X[4] = 1

    X[5:] = np.random.normal(0, 1, len(X)-5)

    return X.reshape(1, -1)

# =========================
# PREDICT
# =========================
if st.button("🔍 Analyze Transaction"):

    X = create_features(amount, txn_type, location, time_period)
    X_scaled = scaler.transform(X).T

    pred, prob = predict(X_scaled, threshold)
    prob_val = prob[0][0]

    st.subheader("📊 Result")

    if prob_val > 0.8:
        st.error("🚨 Fraud Detected")
    elif prob_val > 0.5:
        st.warning("⚠️ Suspicious Transaction")
    else:
        st.success("✅ Legit Transaction")

    st.metric("Fraud Probability", f"{prob_val:.4f}")

    # =========================
    # VISUAL 1: PROBABILITY
    # =========================
    fig, ax = plt.subplots()
    ax.bar(["Fraud Risk"], [prob_val])
    ax.set_ylim(0,1)
    st.pyplot(fig)

    # =========================
    # VISUAL 2: INPUT FEATURES
    # =========================
    st.subheader("📈 Risk Factors")

    factors = {
        "Amount": amount,
        "Online": 1 if txn_type=="Online" else 0,
        "Location": 1 if location=="Different City" else 0,
        "Night": 1 if time_period=="Night" else 0
    }

    fig2, ax2 = plt.subplots()
    ax2.bar(factors.keys(), factors.values())
    st.pyplot(fig2)

    # =========================
    # SHAP EXPLANATION
    # =========================
    st.subheader("🧠 SHAP Explanation")

    try:
        shap_values = shap_explain(X_scaled)
        fig_shap, ax_shap = plt.subplots()
        shap.plots.waterfall(shap_values[0], show=False)
        st.pyplot(fig_shap)
    except:
        st.warning("SHAP visualization not available")

# =========================
# LIVE SIMULATION
# =========================
st.subheader("⚡ Live Monitoring")

if st.button("Start Simulation"):

    placeholder = st.empty()
    history = []

    for i in range(30):
        X = create_features(np.random.randint(10,5000), "Online", "Different City", "Night")
        X_scaled = scaler.transform(X).T

        _, prob = predict(X_scaled, threshold)
        prob_val = prob[0][0]
        history.append(prob_val)

        with placeholder.container():
            st.write(f"Transaction {i}")

            if prob_val > 0.8:
                st.error("🚨 Fraud")
            elif prob_val > 0.5:
                st.warning("⚠️ Suspicious")
            else:
                st.success("✅ Safe")

            fig, ax = plt.subplots()
            ax.plot(history)
            ax.set_ylim(0,1)
            ax.set_title("Fraud Trend")
            st.pyplot(fig)

        time.sleep(0.3)