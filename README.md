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

	​

2️⃣ Loss Function — Binary Cross-Entropy

This measures how wrong the prediction is:

𝐿
=
−
1
𝑚
∑
𝑖
=
1
𝑚
[
𝑦
(
𝑖
)
log
⁡
(
𝑦
^
(
𝑖
)
)
+
(
1
−
𝑦
(
𝑖
)
)
log
⁡
(
1
−
𝑦
^
(
𝑖
)
)
]
L=−
m
1
	​

i=1
∑
m
	​

[y
(i)
log(
y
^
	​

(i)
)+(1−y
(i)
)log(1−
y
^
	​

(i)
)]

Lower loss = better predictions.

3️⃣ Backpropagation (Chain Rule in Action)

Gradients are computed manually for every parameter.

Output Layer Gradient
𝑑
𝑍
[
2
]
=
𝐴
[
2
]
−
𝑌
dZ
[2]
=A
[2]
−Y
Weight Updates
𝑑
𝑊
[
𝑙
]
=
1
𝑚
𝑑
𝑍
[
𝑙
]
𝐴
[
𝑙
−
1
]
𝑇
dW
[l]
=
m
1
	​

dZ
[l]
A
[l−1]
T
Bias Updates
𝑑
𝑏
[
𝑙
]
=
1
𝑚
∑
𝑑
𝑍
[
𝑙
]
db
[l]
=
m
1
	​

∑dZ
[l]

ReLU derivative:

𝑑
𝑑
𝑧
ReLU
(
𝑧
)
=
{
1
	
𝑧
>
0


0
	
𝑧
≤
0
dz
d
	​

ReLU(z)={
1
0
	​

z>0
z≤0
	​


This is pure calculus + matrix multiplication.

4️⃣ Optimization — Gradient Descent

Each parameter is updated as:

𝜃
:
=
𝜃
−
𝛼
⋅
∇
𝜃
θ:=θ−α⋅∇
θ
	​


Where:

𝛼
α = learning rate

∇
𝜃
∇
θ
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
