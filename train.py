
from neural_network import NeuralNetwork, cross_validate, hyperparameter_search
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import numpy as np

data = load_breast_cancer()
X_orig = data.data
y_orig = data.target.reshape(1, -1)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_orig)

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X_scaled, y_orig.T, test_size=0.2, stratify=y_orig.T, random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.15, stratify=y_train_full, random_state=42
)

X_train = X_train.T
X_val = X_val.T
X_test = X_test.T
y_train = y_train.T
y_val = y_val.T
y_test = y_test.T

print("="*70)
print("BREAST CANCER CLASSIFICATION - NEURAL NETWORK TRAINING")
print("="*70)
print(f"Training Samples: {X_train.shape[1]}")
print(f"Validation Samples: {X_val.shape[1]}")
print(f"Test Samples: {X_test.shape[1]}")
print(f"Features: {X_train.shape[0]}")
print("="*70)

model = NeuralNetwork(
    input_size=30,
    hidden_size=16,
    output_size=1,
    learning_rate=0.01,
    l2_lambda=0.001,
    dropout_rate=0.3,
    initialization='he',
    seed=42
)

training_history = model.train(
    X_train, y_train,
    X_val=X_val, Y_val=y_val,
    epochs=1500,
    batch_size=32,
    print_cost=True,
    early_stopping=True,
    early_stopping_patience=150,
    lr_schedule='exponential'
)

print("\n" + "="*70)
print("FINAL EVALUATION")
print("="*70)

train_metrics = model.evaluate(X_train, y_train)
val_metrics = model.evaluate(X_val, y_val)
test_metrics = model.evaluate(X_test, y_test)

print(f"\n{'Metric':<12} {'Train':>10} {'Validation':>12} {'Test':>10}")
print("-"*50)
for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']:
    print(f"{metric:<12} {train_metrics[metric]:>10.4f} {val_metrics[metric]:>12.4f} {test_metrics[metric]:>10.4f}")

fpr, tpr, auc_val = model.get_roc_data(X_test, y_test)
print(f"\nTest ROC-AUC: {auc_val:.4f}")

cm = model.get_confusion_matrix(X_test, y_test)
print(f"\nConfusion Matrix:\n{cm}")

sample_idx = 5
sample = X_test[:, sample_idx:sample_idx+1]
prediction_prob = model.predict(sample)[0, 0]
prediction_class = int(prediction_prob >= 0.5)
actual_class = int(y_test[0, sample_idx])

print(f"\nSample Prediction (index {sample_idx}):")
print(f"  Probability: {prediction_prob:.4f}")
print(f"  Predicted Class: {prediction_class} ({'Malignant' if prediction_class else 'Benign'})")
print(f"  Actual Class: {actual_class} ({'Malignant' if actual_class else 'Benign'})")
print(f"  Correct: {'✓' if prediction_class == actual_class else '✗'}")

feature_importance = model.get_feature_importance(X_test, y_test, n_permutations=5)
top_features = np.argsort(feature_importance)[::-1][:10]
print(f"\nTop 10 Most Important Features:")
for rank, idx in enumerate(top_features, 1):
    print(f"  {rank}. Feature {idx}: Importance = {feature_importance[idx]:.4f}")

model.save_model('models/breast_cancer_nn_model.pkl', metadata={
    'dataset': 'Breast Cancer Wisconsin (Diagnostic)',
    'n_samples_train': X_train.shape[1],
    'n_samples_test': X_test.shape[1],
    'final_test_accuracy': test_metrics['accuracy']
})

print(f"\n✓ Model saved to 'models/breast_cancer_nn_model.pkl'")
print(f"✓ Training completed at {datetime.now().isoformat()}")
