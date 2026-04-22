# 💳 Neural Network From Scratch (Fraud Detection)

Minimal, from-scratch neural network built using **NumPy only**, applied to a real-world problem: **credit card fraud detection**.

---

## 🚀 Problem

Fraud detection is hard because:

- Data is **extremely imbalanced**
- Accuracy is misleading  
- Default thresholds fail  
- Frameworks hide core logic  

👉 Example: only ~492 frauds in 284,807 transactions (~0.17%) :contentReference[oaicite:0]{index=0}

---

## 💡 Solution

- Neural network built **from scratch**
- Manual forward + backprop
- **Adam optimizer**
- **ROC-based threshold tuning**
- Evaluated using **F1 score**

---

## ⚙️ Features

- Pure NumPy implementation  
- Experiment runner (multi configs)  
- Train / validation tracking  
- Best model selection  
- Model saving (`.pkl`)  

---

## 📂 Dataset

Credit Card Fraud Dataset (Kaggle)  
👉 https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud  

- 284K+ transactions  
- Highly imbalanced  
- Anonymized features (V1–V28)  

> Place `creditcard.csv` in project root before running

---

## ▶️ Run

```bash
git clone https://github.com/VasuML07/NeuralNetworkfromscratch
cd NeuralNetworkfromscratch
python main.py
