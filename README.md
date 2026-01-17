🧠 Breast Cancer Detection — Neural Network From Scratch (NumPy Only)
🚀 Project Overview

This project implements a fully connected neural network (Multi-Layer Perceptron) from first principles, using only Python and NumPy.
No TensorFlow. No PyTorch. No shortcuts.

The model classifies tumors as Malignant or Benign using the Wisconsin Breast Cancer Dataset, proving that deep learning is just math + logic + iteration, not magic.

The goal is simple but ambitious:

Demystify neural networks by building every component manually.

🧩 Why This Project Exists (Read This If You’re Serious About ML)

Most ML projects hide the learning process behind high-level APIs.
This one does the opposite.

By implementing everything from scratch, this repository demonstrates:

How neural networks actually compute

How gradients flow via calculus (Chain Rule)

How optimization updates weights using linear algebra

Why neural networks are not black boxes if you understand the math

If you can explain this project, you can explain deep learning fundamentals confidently.

📊 Dataset

Wisconsin Breast Cancer Dataset

Samples: 569

Features: 30 real-valued tumor characteristics
(radius, texture, smoothness, concavity, symmetry, etc.)

Labels:

0 → Malignant

1 → Benign

All features are standardized to ensure stable gradient descent.

🏗️ Neural Network Architecture
Input Layer (30 features)
        ↓
Hidden Layer (16 neurons)
[Linear → ReLU]
        ↓
Output Layer (1 neuron)
[Linear → Sigmoid]
        ↓
Binary Prediction (0 or 1)

Layer Breakdown

Input Layer:
Receives normalized feature vectors

Hidden Layer:
Learns non-linear feature interactions using ReLU

Output Layer:
Outputs probability of malignancy using Sigmoid



ReLU(z)=max(0,z)

σ(z)= 1/(1+e^(-x))

loss = - (y * np.log(Y + 1e-8) + (1 - y) * np.log(1 - Y + 1e-8))
​

2️⃣ Loss Function — Binary Cross-Entropy

This measures how wrong the prediction is:



Lower loss = better predictions.

3️⃣ Backpropagation (Chain Rule in Action)

Gradients are computed manually for every parameter.

Output Layer Gradient
​
dZ2 = A2 - Y

m = Y.shape[1]   # number of samples
dW2 = (1 / m) * np.dot(dZ2, A1.T)

db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)

This is pure calculus + matrix multiplication.

4️⃣ Optimization — Gradient Descent

Each parameter is updated as:

para_new = para_old-(learning-rate)*(gradient)
	​

 = computed gradient

Repeat this for 1000 epochs, and the network learns.

🔁 Training Workflow

Load dataset

Standardize features

Split into train/test sets

Forward propagation

Compute loss

Backpropagation

Update weights

Repeat until convergence

This loop is the heartbeat of deep learning.

📈 Evaluation Metrics

The model is evaluated using:

Accuracy

Precision

Recall

F1-Score

Sample Results
Training Accuracy: ~98%
Testing Accuracy:  ~96%


High accuracy without overfitting — achieved without any DL framework.

🧪 Example Prediction
Input: [feature vector]
Actual Label: 1 (Benign)
Predicted Label: 1 (Benign)


The model outputs a probability and applies a 0.5 threshold for classification.

🛠️ Tech Stack

Python 🐍

NumPy (matrix math)

Scikit-learn (data + metrics only)

No deep learning libraries used.

🎯 What This Project Proves

This repository demonstrates strong understanding of:

Neural network internals

Linear algebra for ML

Gradient-based optimization

Loss functions and activations

End-to-end ML pipelines

In short:
You don’t just use neural networks — you understand them.

📌 Future Improvements

Add multi-layer support

Implement Adam optimizer manually

Visualize loss curves

Extend to multiclass classification
