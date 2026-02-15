# 🧠 Breast Cancer Detection  
### Neural Network From Scratch (NumPy Only)

A fully connected neural network (Multi-Layer Perceptron) implemented entirely from first principles using Python and NumPy.

No TensorFlow.  
No PyTorch.  
No abstraction layers.

This project proves that deep learning is linear algebra + calculus + optimization — not magic.

---

## 🚀 Project Overview

The model classifies tumors as:

- 0 → Malignant  
- 1 → Benign  

Using the Wisconsin Breast Cancer Dataset.

The goal is to demystify neural networks by manually implementing:

- Forward propagation  
- Backpropagation  
- Gradient descent  
- Binary cross-entropy loss  

---

## 📊 Dataset

**Wisconsin Breast Cancer Dataset**

- Samples: 569  
- Features: 30 real-valued tumor characteristics  
  (radius, texture, smoothness, concavity, symmetry, etc.)  

All features are standardized for stable gradient descent.

---

## 🏗️ Neural Network Architecture

Input Layer (30 features)
↓
Hidden Layer (16 neurons)
[Linear → ReLU]
↓
Output Layer (1 neuron)
[Linear → Sigmoid]
↓
Binary Prediction (0 or 1)


---

## ⚙️ Mathematical Foundations

### 1️⃣ Activation Functions

ReLU:
ReLU(z) = max(0, z)


Sigmoid:
σ(z) = 1 / (1 + e^(-z))


---

### 2️⃣ Loss Function — Binary Cross-Entropy

Loss = - ( y log(Y) + (1 - y) log(1 - Y) )


Lower loss indicates better predictions.

---

### 3️⃣ Backpropagation (Chain Rule)

Output layer gradient:

```python
dZ2 = A2 - Y
m = Y.shape[1]
dW2 = (1 / m) * np.dot(dZ2, A1.T)
db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
This is pure calculus expressed in matrix form.

4️⃣ Optimization — Gradient Descent
parameter_new = parameter_old - learning_rate × gradient
Repeated over ~1000 epochs until convergence.

🔁 Training Workflow
Load dataset

Standardize features

Train-test split

Forward propagation

Compute loss

Backpropagation

Update weights

Repeat

This loop is the core engine of deep learning.

📈 Evaluation Metrics
Accuracy

Precision

Recall

F1-Score

Sample Results
Training Accuracy: ~98%

Testing Accuracy: ~96%

High generalization achieved without any deep learning framework.

🧪 Example Prediction
Input: Feature vector
Actual Label: 1 (Benign)
Predicted Label: 1 (Benign)

The model outputs probability and applies a 0.5 classification threshold.

🛠 Tech Stack
💻 Core Language

📊 Numerical Computation

📈 Dataset & Metrics

🎯 What This Project Demonstrates
Manual implementation of neural networks

Deep understanding of backpropagation

Linear algebra for ML

Gradient-based optimization

End-to-end ML pipeline development

This project shows not just usage of neural networks — but comprehension of their internals.

🔮 Future Improvements
Add deeper multi-layer support

Implement Adam optimizer manually

Visualize loss curves

Extend to multiclass classification
