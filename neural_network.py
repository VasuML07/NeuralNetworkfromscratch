import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix, classification_report
import json
import pickle
from datetime import datetime
from typing import Dict, Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')
def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
def relu(z):
    return np.maximum(0, z)
def der_relu(z):
    return (z > 0).astype(float)
def he_initializer(fan_in, fan_out):
    return np.random.randn(fan_out, fan_in) * np.sqrt(2. / fan_in)
def xavier_initializer(fan_in, fan_out):
    return np.random.randn(fan_out, fan_in) * np.sqrt(1. / fan_in)
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01, l2_lambda=0.001, dropout_rate=0.0, initialization='he', seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.initial_lr = learning_rate
        self.learning_rate = learning_rate
        self.l2_lambda = l2_lambda
        self.dropout_rate = dropout_rate
        self.initialization = initialization
        self.history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}
        self.best_weights = None
        self.best_val_loss = float('inf')
        if initialization == 'he':
            W1 = he_initializer(input_size, hidden_size)
            W2 = he_initializer(hidden_size, output_size)
        else:
            W1 = xavier_initializer(input_size, hidden_size)
            W2 = xavier_initializer(hidden_size, output_size)
        self.params = {
            'W1': W1,
            'B1': np.zeros((hidden_size, 1)),
            'W2': W2,
            'B2': np.zeros((output_size, 1))
        }
    def forward_propagation(self, X, training=True):
        Z1 = np.dot(self.params['W1'], X) + self.params['B1']
        A1 = relu(Z1)
        if training and self.dropout_rate > 0:
            dropout_mask = (np.random.rand(*A1.shape) > self.dropout_rate).astype(float)
            A1 = A1 * dropout_mask / (1 - self.dropout_rate)
        else:
            dropout_mask = None
        Z2 = np.dot(self.params['W2'], A1) + self.params['B2']
        A2 = sigmoid(Z2)
        cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2, "dropout_mask": dropout_mask}
        return A2, cache
    def loss(self, A2, Y, m):
        epsilon = 1e-15
        A2_clipped = np.clip(A2, epsilon, 1 - epsilon)
        base_loss = - (1/m) * np.sum(Y * np.log(A2_clipped) + (1-Y) * np.log(1-A2_clipped))
        l2_reg = (self.l2_lambda / (2*m)) * (np.sum(self.params['W1']**2) + np.sum(self.params['W2']**2))
        return base_loss + l2_reg
    def backward_propagation(self, X, Y, cache, m):
        A1 = cache['A1']
        A2 = cache['A2']
        Z1 = cache['Z1']
        W2 = self.params['W2']
        dropout_mask = cache.get('dropout_mask')
        dZ2 = A2 - Y
        dW2 = (1/m) * np.dot(dZ2, A1.T) + (self.l2_lambda/m) * W2
        db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)
        dA1 = np.dot(W2.T, dZ2)
        if dropout_mask is not None:
            dA1 = dA1 * dropout_mask / (1 - self.dropout_rate)
        dZ1 = dA1 * der_relu(Z1)
        dW1 = (1/m) * np.dot(dZ1, X.T) + (self.l2_lambda/m) * self.params['W1']
        db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)
        return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
    def update_parameters(self, gradients):
        self.params['W1'] -= self.learning_rate * gradients['dW1']
        self.params['B1'] -= self.learning_rate * gradients['db1']
        self.params['W2'] -= self.learning_rate * gradients['dW2']
        self.params['B2'] -= self.learning_rate * gradients['db2']
    def adjust_learning_rate(self, epoch, strategy='exponential', decay_rate=0.95, patience=50):
        if strategy == 'exponential':
            self.learning_rate = self.initial_lr * (decay_rate ** (epoch // 100))
        elif strategy == 'step':
            if epoch % 200 == 0 and epoch > 0:
                self.learning_rate *= 0.5
        elif strategy == 'plateau':
            if len(self.history['val_loss']) > patience:
                recent = self.history['val_loss'][-patience:]
                if all(recent[i] >= recent[i-1] for i in range(1, len(recent))):
                    self.learning_rate *= 0.7
    def train(self, X, Y, X_val=None, Y_val=None, epochs=1000, batch_size=None, print_cost=True, early_stopping=True, early_stopping_patience=100, lr_schedule='exponential'):
        m = X.shape[1]
        if batch_size is None:
            batch_size = m
        best_patience_counter = 0
        for epoch in range(epochs):
            indices = np.random.permutation(m)
            X_shuffled = X[:, indices]
            Y_shuffled = Y[:, indices]
            epoch_losses = []
            for mini_batch_start in range(0, m, batch_size):
                mini_batch_end = min(mini_batch_start + batch_size, m)
                X_batch = X_shuffled[:, mini_batch_start:mini_batch_end]
                Y_batch = Y_shuffled[:, mini_batch_start:mini_batch_end]
                A2, cache = self.forward_propagation(X_batch, training=True)
                batch_loss = self.loss(A2, Y_batch, X_batch.shape[1])
                epoch_losses.append(batch_loss)
                gradients = self.backward_propagation(X_batch, Y_batch, cache, X_batch.shape[1])
                self.update_parameters(gradients)
            avg_train_loss = np.mean(epoch_losses)
            self.history['loss'].append(avg_train_loss)
            train_pred = (self.predict(X) > 0.5).flatten()
            train_acc = accuracy_score(Y.flatten(), train_pred)
            self.history['accuracy'].append(train_acc)
            if X_val is not None and Y_val is not None:
                val_pred_prob, _ = self.forward_propagation(X_val, training=False)
                val_loss = self.loss(val_pred_prob, Y_val, X_val.shape[1])
                self.history['val_loss'].append(val_loss)
                val_pred = (val_pred_prob > 0.5).flatten()
                val_acc = accuracy_score(Y_val.flatten(), val_pred)
                self.history['val_accuracy'].append(val_acc)
                if early_stopping and val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.best_weights = {k: v.copy() for k, v in self.params.items()}
                    best_patience_counter = 0
                elif early_stopping:
                    best_patience_counter += 1
            self.adjust_learning_rate(epoch, strategy=lr_schedule)
            if print_cost and epoch % 100 == 0:
                if X_val is not None:
                    print(f"Epoch {epoch:4d} | Loss: {avg_train_loss:.4f} | Val Loss: {self.history['val_loss'][-1]:.4f} | Acc: {train_acc:.4f} | Val Acc: {self.history['val_accuracy'][-1]:.4f} | LR: {self.learning_rate:.6f}")
                else:
                    print(f"Epoch {epoch:4d} | Loss: {avg_train_loss:.4f} | Acc: {train_acc:.4f} | LR: {self.learning_rate:.6f}")
            if early_stopping and best_patience_counter >= early_stopping_patience:
                if print_cost:
                    print(f"Early stopping at epoch {epoch}")
                if self.best_weights is not None:
                    self.params = self.best_weights
                break
        if self.best_weights is not None and early_stopping:
            self.params = self.best_weights
        return self.history
    def predict(self, X):
        A2, _ = self.forward_propagation(X, training=False)
        return A2
    def predict_classes(self, X, threshold=0.5):
        return (self.predict(X) >= threshold).astype(int)
    def evaluate(self, X, Y):
        Y_pred_prob = self.predict(X).flatten()
        Y_pred = (Y_pred_prob >= 0.5).astype(int)
        Y_true = Y.flatten()
        return {
            'accuracy': accuracy_score(Y_true, Y_pred),
            'precision': precision_score(Y_true, Y_pred, zero_division=0),
            'recall': recall_score(Y_true, Y_pred, zero_division=0),
            'f1_score': f1_score(Y_true, Y_pred, zero_division=0),
            'roc_auc': roc_auc_score(Y_true, Y_pred_prob) if len(np.unique(Y_true)) > 1 else 0.5
        }
    def get_roc_data(self, X, Y):
        Y_pred_prob = self.predict(X).flatten()
        fpr, tpr, _ = roc_curve(Y.flatten(), Y_pred_prob)
        auc_score = roc_auc_score(Y.flatten(), Y_pred_prob)
        return fpr, tpr, auc_score
    def get_confusion_matrix(self, X, Y):
        Y_pred = self.predict_classes(X).flatten()
        return confusion_matrix(Y.flatten(), Y_pred)
    def save_model(self, filepath, metadata=None):
        model_data = {
            'params': {k: v.tolist() for k, v in self.params.items()},
            'hyperparameters': {
                'input_size': self.input_size,
                'hidden_size': self.hidden_size,
                'output_size': self.output_size,
                'learning_rate': self.initial_lr,
                'l2_lambda': self.l2_lambda,
                'dropout_rate': self.dropout_rate,
                'initialization': self.initialization
            },
            'history': self.history,
            'metadata': metadata or {'timestamp': datetime.now().isoformat()}
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    @classmethod
    def load_model(cls, filepath):
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        hp = model_data['hyperparameters']
        model = cls(
            input_size=hp['input_size'],
            hidden_size=hp['hidden_size'],
            output_size=hp['output_size'],
            learning_rate=hp['learning_rate'],
            l2_lambda=hp['l2_lambda'],
            dropout_rate=hp['dropout_rate'],
            initialization=hp['initialization']
        )
        model.params = {k: np.array(v) for k, v in model_data['params'].items()}
        model.history = model_data['history']
        return model
    def get_feature_importance(self, X, Y, n_permutations=10):
        base_metrics = self.evaluate(X, Y)
        base_score = base_metrics['f1_score']
        importance = np.zeros(self.input_size)
        for feat_idx in range(self.input_size):
            scores = []
            for _ in range(n_permutations):
                X_permuted = X.copy()
                np.random.shuffle(X_permuted[feat_idx, :])
                perm_metrics = self.evaluate(X_permuted, Y)
                scores.append(perm_metrics['f1_score'])
            importance[feat_idx] = base_score - np.mean(scores)
        return importance
def cross_validate(X, Y, hidden_size=16, n_folds=5, epochs=500, **nn_kwargs):
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    results = {'accuracy': [], 'precision': [], 'recall': [], 'f1_score': [], 'roc_auc': []}
    for fold, (train_idx, test_idx) in enumerate(kf.split(X, Y.flatten())):
        X_train, X_test = X[train_idx], X[test_idx]
        Y_train, Y_test = Y[:, train_idx], Y[:, test_idx]
        model = NeuralNetwork(
            input_size=X_train.shape[0],
            hidden_size=hidden_size,
            output_size=1,
            seed=42 + fold,
            **nn_kwargs
        )
        val_split = 0.15
        n_val = int(X_train.shape[1] * val_split)
        indices = np.random.permutation(X_train.shape[1])
        X_tr, X_val = X_train[:, indices[n_val:]], X_train[:, indices[:n_val]]
        Y_tr, Y_val = Y_train[:, indices[n_val:]], Y_train[:, indices[:n_val]]
        model.train(X_tr, Y_tr, X_val, Y_val, epochs=epochs, print_cost=False)
        metrics = model.evaluate(X_test, Y_test)
        for key in results:
            results[key].append(metrics[key])
    return {k: v for k, v in results.items()}
def hyperparameter_search(X, Y, param_grid, n_folds=3, epochs=300):
    best_score = -1
    best_params = None
    best_model = None
    for lr in param_grid.get('learning_rate', [0.01]):
        for l2 in param_grid.get('l2_lambda', [0.001]):
            for hidden in param_grid.get('hidden_size', [16]):
                for dropout in param_grid.get('dropout_rate', [0.0]):
                    cv_results = cross_validate(
                        X, Y,
                        hidden_size=hidden,
                        learning_rate=lr,
                        l2_lambda=l2,
                        dropout_rate=dropout,
                        n_folds=n_folds,
                        epochs=epochs
                    )
                    mean_f1 = np.mean(cv_results['f1_score'])
                    if mean_f1 > best_score:
                        best_score = mean_f1
                        best_params = {'learning_rate': lr, 'l2_lambda': l2, 'hidden_size': hidden, 'dropout_rate': dropout}
    best_model = NeuralNetwork(
        input_size=X.shape[0],
        hidden_size=best_params['hidden_size'],
        output_size=1,
        learning_rate=best_params['learning_rate'],
        l2_lambda=best_params['l2_lambda'],
        dropout_rate=best_params['dropout_rate'],
        seed=42
    )
    X_train, X_test, Y_train, Y_test = train_test_split(
        X.T, Y.T, test_size=0.2, stratify=Y.T, random_state=42
    )
    best_model.train(
        X_train.T, Y_train.T,
        X_test.T, Y_test.T,
        epochs=epochs * 2,
        print_cost=True
    )
    return best_params, best_model
