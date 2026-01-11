1. Project Overview
Title: Breast Cancer Detection: Neural Networks from First Principles

Description: This repository bridges the gap between theory and practice by implementing a neural network without the aid of deep learning frameworks like TensorFlow or PyTorch. The goal is to classify breast cancer tumors as either Malignant or Benign using the Wisconsin Breast Cancer dataset.

While we leverage scikit-learn to handle the logistics of loading the data and splitting it into training/testing sets, the actual "brain" of the model—the learning algorithms, weight updates, and predictions—is built entirely using raw mathematics and NumPy matrix operations.

2. The Architecture
The model creates a fully connected neural network (Multi-Layer Perceptron) designed to process the 30 distinct features provided in the dataset (such as tumor radius, texture, and smoothness).

Input Layer: Receives the normalized data from the dataset.

Hidden Layers: Performs matrix multiplication (Dot Product) between inputs and weights, adds a bias term, and applies a non-linear activation function (ReLU) to capture complex patterns.

Output Layer: Compresses the final signal into a probability between 0 and 1 using the Sigmoid activation function, predicting the likelihood of malignancy.

3. The Mathematics (The "From Scratch" Part)
The core of this repository is the manual implementation of the learning cycle:

Forward Propagation: Data flows through the network, transforming inputs into a prediction.

Loss Calculation: We calculate "Binary Cross-Entropy" to mathematically quantify the difference between the model's prediction  the actual diagnosis.

Backpropagation: Using the Chain Rule of Calculus, we compute the gradient of the loss with respect to every single weight and bias in the network. This tells us exactly how much to adjust each parameter to reduce error.

Optimization: We implement an optimizer (like Adam or Stochastic Gradient Descent) to update the weights based on the calculated gradients, effectively "learning" from the data.

4. Data Workflow
Loading: The dataset is fetched using standard data science tools.

Preprocessing: The data is standardized (scaled) so that features with larger ranges (like Area) don't overpower features with smaller ranges (like Symmetry).

Splitting: The data is separated to ensure the model is tested on unseen examples, verifying it hasn't just memorized the training data.

Training: The NumPy model iterates through the training data, refining its accuracy over thousands of epochs.

5. Why This Matters
Building this from scratch proves that the model is not a "black box." It demonstrates a foundational understanding of:

How computers process biological data.

The calculus behind error minimization.

The linear algebra required for high-speed computation.
