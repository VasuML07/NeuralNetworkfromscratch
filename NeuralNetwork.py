import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def relu(z):
    return np.maximum(0, z)

def der_relu(z):
    return (z > 0).astype(float)

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.params = {
            'W1': np.random.randn(hidden_size, input_size) * 0.01,
            'B1': np.zeros((hidden_size, 1)),
            'W2': np.random.randn(output_size, hidden_size) * 0.01,
            'B2': np.zeros((output_size, 1))
        }

    def forward_propagation(self, X):
        Z1 = np.dot(self.params['W1'], X) + self.params['B1']
        A1 = relu(Z1)
        Z2 = np.dot(self.params['W2'], A1) + self.params['B2']
        A2 = sigmoid(Z2)
        cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
        return A2, cache

    def loss(self, A2, Y):
        m = Y.shape[1]
        epsilon = 1e-15
        cost = - (1/m) * np.sum(Y * np.log(A2 + epsilon) + (1-Y) * np.log(1-A2 + epsilon))
        return np.squeeze(cost)

    def backward_propagation(self, X, Y, cache):
        m = Y.shape[1]
        A1 = cache['A1']
        A2 = cache['A2']
        Z1 = cache['Z1']
        W2 = self.params['W2']
        dZ2 = A2 - Y
        dW2 = (1/m) * np.dot(dZ2, A1.T)
        db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)
        dA1 = np.dot(W2.T, dZ2)
        dZ1 = dA1 * der_relu(Z1)
        dW1 = (1/m) * np.dot(dZ1, X.T)
        db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)
        gradients = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
        return gradients

    def update_parameters(self, gradients):
        self.params['W1'] -= self.learning_rate * gradients['dW1']
        self.params['B1'] -= self.learning_rate * gradients['db1']
        self.params['W2'] -= self.learning_rate * gradients['dW2']
        self.params['B2'] -= self.learning_rate * gradients['db2']

    def train(self, X, Y, epochs=1000, print_cost=True):
        for i in range(epochs):
            A2, cache = self.forward_propagation(X)
            cost = self.loss(A2, Y)
            gradients = self.backward_propagation(X, Y, cache)
            self.update_parameters(gradients)
            if print_cost and i % 100 == 0:
                print(f"Cost after epoch {i}: {cost:.4f}")

    def predict(self, X):
        A2, _ = self.forward_propagation(X)
        predictions = (A2 > 0.5).astype(int)
        return predictions

data = load_breast_cancer()
X_orig = data.data
y_orig = data.target.reshape(1, -1)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_orig)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_orig.T, test_size=0.2, random_state=42)

X_train = X_train.T
X_test = X_test.T
y_train = y_train.T
y_test = y_test.T

print(f"Training Data Shape: {X_train.shape}")
print(f"Testing Data Shape: {X_test.shape}")

model = NeuralNetwork(input_size=30, hidden_size=16, output_size=1, learning_rate=0.01)
print("\n -> Starting Training...")
model.train(X_train, y_train, epochs=1000)

print("\n--- Evaluation ---")
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

train_acc = accuracy_score(y_train.flatten(), y_pred_train.flatten())
test_acc = accuracy_score(y_test.flatten(), y_pred_test.flatten())

print(f"Training Accuracy: {train_acc * 100:.2f}%")
print(f"Testing Accuracy:  {test_acc * 100:.2f}%")

sample = X_test[:, 5].reshape(-1, 1)
print(f"Input: {sample.flatten()}")
print(f"Actual: {y_test[0, 5]}")
print(f"Predicted: {model.predict(sample)[0, 0]}") add more features amke it some reasearch oriented and more features
