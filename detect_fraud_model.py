"""
Credit Card Fraud Detection
Model Comparison: Custom Backpropagation MLP vs Keras MLP
Using Stratified K-Fold Cross-Validation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("=" * 60)
print("CREDIT CARD FRAUD DETECTION - MODEL COMPARISON")
print("=" * 60)

# ============================================
# STEP 1: LOAD AND EXPLORE DATA
# ============================================
print("\n[Step 1] Loading dataset...")

# Load data (update path to your file location)
df = pd.read_csv('creditcard.csv')

print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"\nClass distribution:")
print(df['Class'].value_counts())
print(f"Fraud percentage: {df['Class'].mean() * 100:.4f}%")

# ============================================
# STEP 2: PREPROCESSING
# ============================================
print("\n[Step 2] Preprocessing data...")

# Scale 'Time' and 'Amount' features (V1-V28 are already scaled from PCA)
scaler = StandardScaler()
df['scaled_amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
df['scaled_time'] = scaler.fit_transform(df['Time'].values.reshape(-1, 1))

# Drop original unscaled columns
df.drop(['Time', 'Amount'], axis=1, inplace=True)

# Prepare features and target
X = df.drop('Class', axis=1).values
y = df['Class'].values

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"Feature names: {df.drop('Class', axis=1).columns.tolist()}")

# ============================================
# STEP 3: HANDLE CLASS IMBALANCE WITH SMOTE
# ============================================
print("\n[Step 3] Applying SMOTE to handle class imbalance...")

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

print(f"Original shape: {X.shape}")
print(f"Resampled shape: {X_resampled.shape}")
print(f"Resampled class distribution:")
print(pd.Series(y_resampled).value_counts())

# ============================================
# STEP 4: CUSTOM BACKPROPAGATION MLP IMPLEMENTATION
# ============================================
print("\n[Step 4] Implementing Custom MLP with Backpropagation...")


class CustomMLP:
    """
    Multi-Layer Perceptron with manual backpropagation implementation.
    """

    def __init__(self, layer_sizes, learning_rate=0.001, epochs=50, batch_size=32):
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.weights = []
        self.biases = []
        self.losses = []

        # Initialize weights and biases
        for i in range(len(layer_sizes) - 1):
            limit = np.sqrt(6 / (layer_sizes[i] + layer_sizes[i + 1]))
            w = np.random.uniform(-limit, limit, (layer_sizes[i], layer_sizes[i + 1]))
            b = np.zeros((1, layer_sizes[i + 1]))
            self.weights.append(w)
            self.biases.append(b)

    def _sigmoid(self, x):
        """Sigmoid activation function."""
        # Clip to prevent overflow
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    def _sigmoid_derivative(self, x):
        """Derivative of sigmoid."""
        s = self._sigmoid(x)
        return s * (1 - s)

    def _relu(self, x):
        """ReLU activation function."""
        return np.maximum(0, x)

    def _relu_derivative(self, x):
        """Derivative of ReLU."""
        return (x > 0).astype(float)

    def _binary_cross_entropy(self, y_true, y_pred):
        """Binary cross-entropy loss - FIXED for proper shapes."""
        # Ensure inputs are 1D or properly shaped
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        # Clip to avoid log(0)
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss

    def forward(self, X, return_activations=False):
        """Forward pass."""
        activations = [X]
        current_input = X

        for i in range(len(self.weights) - 1):
            z = np.dot(current_input, self.weights[i]) + self.biases[i]
            a = self._relu(z)
            activations.append(a)
            current_input = a

        # Output layer
        z_output = np.dot(current_input, self.weights[-1]) + self.biases[-1]
        output = self._sigmoid(z_output)
        activations.append(output)

        if return_activations:
            return output, activations
        return output

    def backward(self, X, y, output, activations):
        """Backward pass."""
        m = X.shape[0]
        gradients_w = [np.zeros_like(w) for w in self.weights]
        gradients_b = [np.zeros_like(b) for b in self.biases]

        # Output layer error - ensure y is correct shape
        y_reshaped = y.reshape(-1, 1)
        delta = output - y_reshaped

        # Output layer gradients
        gradients_w[-1] = np.dot(activations[-2].T, delta) / m
        gradients_b[-1] = np.sum(delta, axis=0, keepdims=True) / m

        # Backpropagate through hidden layers
        for i in range(len(self.weights) - 2, -1, -1):
            delta = np.dot(delta, self.weights[i + 1].T) * self._relu_derivative(activations[i + 1])
            gradients_w[i] = np.dot(activations[i].T, delta) / m
            gradients_b[i] = np.sum(delta, axis=0, keepdims=True) / m

        return gradients_w, gradients_b

    def update_parameters(self, gradients_w, gradients_b):
        """Update weights and biases."""
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * gradients_w[i]
            self.biases[i] -= self.learning_rate * gradients_b[i]

    def train(self, X, y, X_val=None, y_val=None, verbose=True):
        """Train the neural network."""
        n_samples = X.shape[0]

        # Convert to float32 to save memory
        X = X.astype(np.float32)
        y = y.astype(np.float32)

        val_losses = []  # Initialize val_losses list

        for epoch in range(self.epochs):
            epoch_loss = 0
            num_batches = 0

            # Mini-batch training
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for i in range(0, n_samples, self.batch_size):
                X_batch = X_shuffled[i:i + self.batch_size]
                y_batch = y_shuffled[i:i + self.batch_size]

                # Forward pass
                output, activations = self.forward(X_batch, return_activations=True)

                # Compute loss
                loss = self._binary_cross_entropy(y_batch, output)
                epoch_loss += loss
                num_batches += 1

                # Backward pass
                gradients_w, gradients_b = self.backward(X_batch, y_batch, output, activations)

                # Update parameters
                self.update_parameters(gradients_w, gradients_b)

            avg_loss = epoch_loss / num_batches
            self.losses.append(avg_loss)

            # Validation
            val_loss = None
            if X_val is not None:
                val_pred = self.predict(X_val)
                val_loss = self._binary_cross_entropy(y_val, val_pred)
                val_losses.append(val_loss)  # Append to val_losses list

            if verbose and (epoch % 10 == 0 or epoch == self.epochs - 1):
                train_pred = self.predict(X)
                train_acc = accuracy_score(y, (train_pred > 0.5).astype(int))
                if X_val is not None:
                    val_acc = accuracy_score(y_val, (val_pred > 0.5).astype(int))
                    print(f"Epoch {epoch}: Loss = {avg_loss:.4f}, Train Acc = {train_acc:.4f}, Val Acc = {val_acc:.4f}")
                else:
                    print(f"Epoch {epoch}: Loss = {avg_loss:.4f}, Train Acc = {train_acc:.4f}")

        return self.losses, val_losses if X_val is not None else None

    def predict(self, X):
        """Predict probabilities."""
        X = X.astype(np.float32)
        return self.forward(X)

    def predict_binary(self, X, threshold=0.5):
        """Predict binary classes."""
        return (self.predict(X) > threshold).astype(int)

# ============================================
# STEP 5: STRATIFIED K-FOLD CROSS-VALIDATION
# ============================================
print("\n[Step 5] Setting up Stratified 5-Fold Cross-Validation...")

# Use SMOTE-resampled data for fair comparison
X_data = X_resampled
y_data = y_resampled

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Store results
custom_results = {
    'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'auc': []
}
keras_results = {
    'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'auc': []
}

# ================================    ============
# STEP 6: TRAIN AND EVALUATE WITH K-FOLD
# ============================================
print("\n[Step 6] Training models with 5-fold CV...")
print("-" * 60)

fold = 1
for train_idx, val_idx in skf.split(X_data, y_data):
    print(f"\n{'='*40}")
    print(f"FOLD {fold}/5")
    print(f"{'='*40}")
    
    X_train, X_val = X_data[train_idx], X_data[val_idx]
    y_train, y_val = y_data[train_idx], y_data[val_idx]
    
    print(f"Training samples: {X_train.shape[0]}, Validation samples: {X_val.shape[0]}")
    
    # ---------- Custom MLP ----------
    print("\n>>> Training Custom MLP (Hand-coded Backpropagation)...")
    
    # Architecture: Input (30) -> Hidden1 (32) -> Hidden2 (16) -> Output (1)
    custom_mlp = CustomMLP(
        layer_sizes=[30, 32, 16, 1],  # Smaller hidden layers
        learning_rate=0.001,
        epochs=50,
        batch_size=64
    )
    
    train_losses, val_losses = custom_mlp.train(
        X_train, y_train, X_val, y_val, verbose=False
    )
    
    # Evaluate Custom MLP
    y_pred_proba = custom_mlp.predict(X_val)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    custom_results['accuracy'].append(accuracy_score(y_val, y_pred))
    custom_results['precision'].append(precision_score(y_val, y_pred))
    custom_results['recall'].append(recall_score(y_val, y_pred))
    custom_results['f1'].append(f1_score(y_val, y_pred))
    custom_results['auc'].append(roc_auc_score(y_val, y_pred_proba))
    
    print(f"  Custom MLP - Acc: {custom_results['accuracy'][-1]:.4f}, "
          f"Prec: {custom_results['precision'][-1]:.4f}, "
          f"Rec: {custom_results['recall'][-1]:.4f}, "
          f"F1: {custom_results['f1'][-1]:.4f}, "
          f"AUC: {custom_results['auc'][-1]:.4f}")
    
    # ---------- Keras MLP ----------
    print("\n>>> Training Keras MLP...")
    
    keras_model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])
    
    keras_model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Early stopping to prevent overfitting
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True
    )
    
    history = keras_model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=64,
        validation_data=(X_val, y_val),
        verbose=0,
        callbacks=[early_stop]
    )
    
    # Evaluate Keras MLP
    y_pred_proba_keras = keras_model.predict(X_val, verbose=0)
    y_pred_keras = (y_pred_proba_keras > 0.5).astype(int)
    
    keras_results['accuracy'].append(accuracy_score(y_val, y_pred_keras))
    keras_results['precision'].append(precision_score(y_val, y_pred_keras))
    keras_results['recall'].append(recall_score(y_val, y_pred_keras))
    keras_results['f1'].append(f1_score(y_val, y_pred_keras))
    keras_results['auc'].append(roc_auc_score(y_val, y_pred_proba_keras))
    
    print(f"  Keras MLP  - Acc: {keras_results['accuracy'][-1]:.4f}, "
          f"Prec: {keras_results['precision'][-1]:.4f}, "
          f"Rec: {keras_results['recall'][-1]:.4f}, "
          f"F1: {keras_results['f1'][-1]:.4f}, "
          f"AUC: {keras_results['auc'][-1]:.4f}")
    
    fold += 1

# ============================================
# STEP 7: BASELINE COMPARISON (Logistic Regression)
# ============================================
print("\n[Step 7] Training Baseline Model (Logistic Regression)...")

# Use the same CV for baseline
baseline_results = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'auc': []}

for train_idx, val_idx in skf.split(X_data, y_data):
    X_train, X_val = X_data[train_idx], X_data[val_idx]
    y_train, y_val = y_data[train_idx], y_data[val_idx]
    
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    
    y_pred = lr.predict(X_val)
    y_pred_proba = lr.predict_proba(X_val)[:, 1]
    
    baseline_results['accuracy'].append(accuracy_score(y_val, y_pred))
    baseline_results['precision'].append(precision_score(y_val, y_pred))
    baseline_results['recall'].append(recall_score(y_val, y_pred))
    baseline_results['f1'].append(f1_score(y_val, y_pred))
    baseline_results['auc'].append(roc_auc_score(y_val, y_pred_proba))

# ============================================
# STEP 8: VISUALIZATION AND RESULTS
# ============================================
print("\n" + "=" * 60)
print("FINAL RESULTS - 5-FOLD CROSS-VALIDATION")
print("=" * 60)

# Summary table
summary = pd.DataFrame({
    'Model': ['Logistic Regression', 'Custom MLP (Hand-coded BP)', 'Keras MLP'],
    'Accuracy': [
        f"{np.mean(baseline_results['accuracy']):.4f} ± {np.std(baseline_results['accuracy']):.4f}",
        f"{np.mean(custom_results['accuracy']):.4f} ± {np.std(custom_results['accuracy']):.4f}",
        f"{np.mean(keras_results['accuracy']):.4f} ± {np.std(keras_results['accuracy']):.4f}"
    ],
    'Precision': [
        f"{np.mean(baseline_results['precision']):.4f} ± {np.std(baseline_results['precision']):.4f}",
        f"{np.mean(custom_results['precision']):.4f} ± {np.std(custom_results['precision']):.4f}",
        f"{np.mean(keras_results['precision']):.4f} ± {np.std(keras_results['precision']):.4f}"
    ],
    'Recall': [
        f"{np.mean(baseline_results['recall']):.4f} ± {np.std(baseline_results['recall']):.4f}",
        f"{np.mean(custom_results['recall']):.4f} ± {np.std(custom_results['recall']):.4f}",
        f"{np.mean(keras_results['recall']):.4f} ± {np.std(keras_results['recall']):.4f}"
    ],
    'F1-Score': [
        f"{np.mean(baseline_results['f1']):.4f} ± {np.std(baseline_results['f1']):.4f}",
        f"{np.mean(custom_results['f1']):.4f} ± {np.std(custom_results['f1']):.4f}",
        f"{np.mean(keras_results['f1']):.4f} ± {np.std(keras_results['f1']):.4f}"
    ],
    'AUC': [
        f"{np.mean(baseline_results['auc']):.4f} ± {np.std(baseline_results['auc']):.4f}",
        f"{np.mean(custom_results['auc']):.4f} ± {np.std(custom_results['auc']):.4f}",
        f"{np.mean(keras_results['auc']):.4f} ± {np.std(keras_results['auc']):.4f}"
    ]
})

print("\n")
print(summary.to_string(index=False))

# ============================================
# STEP 9: CREATE VISUALIZATIONS
# ============================================
print("\n[Step 9] Generating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Plot 1: Bar chart comparison
ax1 = axes[0, 0]
metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']
x = np.arange(len(metrics))
width = 0.25

baseline_means = [np.mean(baseline_results[m.lower()]) for m in metrics]
custom_means = [np.mean(custom_results[m.lower()]) for m in metrics]
keras_means = [np.mean(keras_results[m.lower()]) for m in metrics]

ax1.bar(x - width, baseline_means, width, label='Logistic Regression', color='gray', alpha=0.7)
ax1.bar(x, custom_means, width, label='Custom MLP (Hand-coded)', color='blue', alpha=0.7)
ax1.bar(x + width, keras_means, width, label='Keras MLP', color='green', alpha=0.7)
ax1.set_xlabel('Metrics')
ax1.set_ylabel('Score')
ax1.set_title('Model Performance Comparison (5-Fold CV)')
ax1.set_xticks(x)
ax1.set_xticklabels(metrics)
ax1.legend()
ax1.set_ylim([0, 1.05])

# Plot 2: Box plots for F1-Score across folds
ax2 = axes[0, 1]
data_to_plot = [
    baseline_results['f1'],
    custom_results['f1'],
    keras_results['f1']
]
bp = ax2.boxplot(data_to_plot, labels=['Logistic\nRegression', 'Custom MLP\n(Hand-coded)', 'Keras MLP'], patch_artist=True)
colors = ['gray', 'blue', 'green']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax2.set_ylabel('F1-Score')
ax2.set_title('F1-Score Distribution Across Folds')
ax2.set_ylim([0, 1.05])

# Plot 3: Learning curves comparison (one representative fold)
ax3 = axes[1, 0]
# Retrain on full data to get learning curves
from sklearn.model_selection import train_test_split

# FIXED: Use 30 features, not 29
X_train_full, X_val_full, y_train_full, y_val_full = train_test_split(
    X_data, y_data, test_size=0.2, random_state=42
)

# FIXED: Use 30 input features
custom_full = CustomMLP(
    layer_sizes=[30, 32, 16, 1],  # Changed from 29 to 30
    learning_rate=0.001,
    epochs=50,
    batch_size=64
)
train_losses, val_losses = custom_full.train(
    X_train_full, y_train_full, X_val_full, y_val_full, verbose=False
)

ax3.plot(train_losses, 'b-', label='Custom MLP Training Loss', linewidth=2)
if val_losses:  # Check if val_losses exists
    ax3.plot(val_losses, 'b--', label='Custom MLP Validation Loss', linewidth=2)

keras_full = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train_full.shape[1],)),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
])
keras_full.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = keras_full.fit(
    X_train_full, y_train_full,
    epochs=50,
    batch_size=64,
    validation_data=(X_val_full, y_val_full),
    verbose=0
)

ax3.plot(history.history['loss'], 'g-', label='Keras MLP Training Loss', linewidth=2)
ax3.plot(history.history['val_loss'], 'g--', label='Keras MLP Validation Loss', linewidth=2)

ax3.set_xlabel('Epoch')
ax3.set_ylabel('Loss')
ax3.set_title('Learning Curves (Training vs Validation)')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: ROC Curves comparison
ax4 = axes[1, 1]
# Calculate ROC curves for the last fold
from sklearn.metrics import roc_curve

# FIXED: Use the actual trained models from the last fold
y_pred_custom = custom_mlp.predict(X_val)
y_pred_keras = keras_model.predict(X_val, verbose=0)

# FIXED: Get predictions from the last trained logistic regression model
# The lr variable from the last iteration is available
y_pred_lr = lr.predict_proba(X_val)[:, 1]

fpr_custom, tpr_custom, _ = roc_curve(y_val, y_pred_custom)
fpr_keras, tpr_keras, _ = roc_curve(y_val, y_pred_keras)
fpr_lr, tpr_lr, _ = roc_curve(y_val, y_pred_lr)

ax4.plot(fpr_custom, tpr_custom, 'b-', label=f'Custom MLP (AUC = {np.mean(custom_results["auc"]):.3f})', linewidth=2)
ax4.plot(fpr_keras, tpr_keras, 'g-', label=f'Keras MLP (AUC = {np.mean(keras_results["auc"]):.3f})', linewidth=2)
ax4.plot(fpr_lr, tpr_lr, 'gray', label=f'Logistic Regression (AUC = {np.mean(baseline_results["auc"]):.3f})', linewidth=2)
ax4.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
ax4.set_xlabel('False Positive Rate')
ax4.set_ylabel('True Positive Rate')
ax4.set_title('ROC Curves Comparison')
ax4.legend(loc='lower right')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('fraud_detection_results.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================
# STEP 10: CONFUSION MATRIX (Final Model)
# ============================================
print("\n[Step 10] Confusion Matrix for Best Model (Keras MLP)...")

# Train final Keras model on all data
final_model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_data.shape[1],)),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
])
final_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
final_model.fit(X_data, y_data, epochs=30, batch_size=64, verbose=0)

# Evaluate on original test data (hold-out set from original dataset)
# For demonstration, we'll use a 80/20 split
from sklearn.model_selection import train_test_split
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Note: Test on ORIGINAL imbalanced data to see real-world performance
y_test_pred = (final_model.predict(X_test, verbose=0) > 0.5).astype(int)

cm = confusion_matrix(y_test, y_test_pred)

fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm,
            xticklabels=['Legitimate', 'Fraud'],
            yticklabels=['Legitimate', 'Fraud'])
ax_cm.set_xlabel('Predicted')
ax_cm.set_ylabel('Actual')
ax_cm.set_title('Confusion Matrix - Keras MLP on Imbalanced Test Set')

plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "=" * 60)
print("PROJECT COMPLETE!")
print("=" * 60)
print("\nKey Takeaways:")
print("1. Custom backpropagation implementation demonstrates understanding of gradient flow")
print("2. Stratified K-Fold ensures class distribution is preserved across folds")
print("3. SMOTE addresses class imbalance before training")
print("4. Keras MLP typically outperforms hand-coded version due to optimizations (Adam, Dropout, EarlyStopping)")
print("5. Fraud detection is about RECALL - catching fraudulent transactions matters most")
