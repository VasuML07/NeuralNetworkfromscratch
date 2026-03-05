from neural_network import NeuralNetwork
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
data = load_breast_cancer()
X_orig = data.data
y_orig = data.target.reshape(1, -1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_orig)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_orig.T, test_size=0.2, stratify=y_orig.T, random_state=42
)
X_train = X_train.T
X_test = X_test.T
y_train = y_train.T
y_test = y_test.T
model = NeuralNetwork.load_model('models/breast_cancer_nn_model.pkl')
print("="*70)
print("MODEL EVALUATION REPORT")
print("="*70)
test_metrics = model.evaluate(X_test, y_test)
print(f"\nTest Set Performance:")
print(f"  Accuracy:  {test_metrics['accuracy']*100:.2f}%")
print(f"  Precision: {test_metrics['precision']*100:.2f}%")
print(f"  Recall:    {test_metrics['recall']*100:.2f}%")
print(f"  F1-Score:  {test_metrics['f1_score']*100:.2f}%")
print(f"  ROC-AUC:   {test_metrics['roc_auc']:.4f}")
cm = model.get_confusion_matrix(X_test, y_test)
print(f"\nConfusion Matrix:\n{cm}")
fpr, tpr, auc_val = model.get_roc_data(X_test, y_test)
print(f"\nROC-AUC Score: {auc_val:.4f}")
print("\n" + "="*70)
