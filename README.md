Credit Card Fraud Detection: Custom Backpropagation vs. Keras MLP
Introduction
Credit card fraud causes billions of dollars in annual losses worldwide. Financial institutions need accurate, real-time fraud detection systems to protect their customers and minimize financial risk. Traditional rule-based systems fail to capture complex, evolving fraud patterns, making machine learning approaches essential.

This project compares three machine learning models for fraud detection:

Logistic Regression (baseline linear model)

Custom Multi-Layer Perceptron (hand-coded backpropagation)

Keras MLP (optimized framework implementation)

The primary goal is to demonstrate that deep neural networks, capable of learning non-linear feature interactions, significantly outperform linear models on this task. Additionally, the custom backpropagation implementation demonstrates a deep understanding of gradient flow through multiple network layers.

Problem Statement
The Credit Card Fraud Detection dataset contains 284,807 transactions, of which only 492 (0.172%) are fraudulent. This presents three major challenges:

Challenge	Description	Impact
Severe Class Imbalance	Only 0.17% of transactions are fraud	Standard accuracy is misleading (99.8% by predicting all legitimate)
Non-linear Fraud Patterns	Fraud involves complex feature interactions (e.g., small amount × unusual time × new merchant)	Linear models cannot capture these patterns
High Cost of False Negatives	Missing a fraudulent transaction causes direct financial loss	Models must prioritize recall over raw accuracy
Key Research Question: Can multi-layer neural networks, with hand-coded backpropagation, significantly outperform linear baselines on this highly imbalanced fraud detection task?

Solution Design and Methodology
Dataset Preprocessing
text
Raw Data (284,807 × 31)
        │
        ├── Scale 'Time' and 'Amount' features
        ├── Drop original unscaled columns
        └── Result: 30 features (V1-V28 + scaled_time + scaled_amount)

SMOTE Oversampling
        │
        ├── Balance classes (284,315 fraud + 284,315 legitimate)
        └── Result: 568,630 balanced samples
Model Architecture
Model 1: Logistic Regression (Baseline)

Linear decision boundary

Serves as performance lower bound

Model 2: Custom MLP (Hand-coded Backpropagation)

text
Input (30) → Hidden1 (32, ReLU) → Hidden2 (16, ReLU) → Output (1, Sigmoid)
Forward pass: manual matrix operations

Backward pass: explicit chain rule implementation

Mini-batch gradient descent (batch_size=64)

Model 3: Keras MLP (Framework)

text
Input (30) → Dense(64, ReLU) → Dropout(0.3) → Dense(32, ReLU) → Dropout(0.3) → Output(1, Sigmoid)
Adam optimizer

Dropout for regularization

Early stopping (patience=10)

Backpropagation Implementation (Custom MLP)
Forward Pass:

text
z[l] = W[l] · a[l-1] + b[l]
a[l] = g(z[l])  # ReLU for hidden, Sigmoid for output
Backward Pass (Chain Rule):

text
δ[L] = a[L] - y                    # Output layer error
δ[l] = (δ[l+1] · W[l+1]^T) ⊙ g'(z[l])  # Hidden layer error

∂L/∂W[l] = a[l-1]^T · δ[l] / m     # Weight gradient (average)
∂L/∂b[l] = Σ δ[l] / m               # Bias gradient (average)
Evaluation Strategy
Metric	Why Used	Target
Precision	Minimize false alarms	> 0.99
Recall	Catch fraudulent transactions	> 0.99
F1-Score	Harmonic mean (balanced view)	> 0.99
AUC-PR	Best for imbalanced data	> 0.99
Validation: Stratified 5-Fold Cross-Validation (preserves fraud ratio in each fold)

Implementation and Execution
Environment Setup
bash
# Virtual environment (Python 3.12)
python3 -m venv fraud_detect
source fraud_detect/bin/activate

# Key packages
pip install numpy pandas scikit-learn tensorflow matplotlib seaborn imbalanced-learn
Custom MLP Core Implementation
python
class CustomMLP:
    def backward(self, X, y, output, activations):
        m = X.shape[0]
        delta = output - y.reshape(-1, 1)  # Output layer error
        
        # Backpropagate through hidden layers
        for i in range(len(self.weights) - 2, -1, -1):
            delta = np.dot(delta, self.weights[i + 1].T) * self._relu_derivative(activations[i + 1])
            gradients_w[i] = np.dot(activations[i].T, delta) / m
            gradients_b[i] = np.sum(delta, axis=0, keepdims=True) / m
        return gradients_w, gradients_b
Training Configuration
Parameter	Custom MLP	Keras MLP
Epochs	50	50 (early stopping)
Batch Size	64	64
Learning Rate	0.001	0.001 (Adam default)
Hidden Layers	2 (32→16)	2 (64→32)
Regularization	None	Dropout (0.3)
Evaluation and Results
5-Fold Cross-Validation Results
Model	Accuracy	Precision	Recall	F1-Score	AUC
Logistic Regression	0.9478 ± 0.0008	0.9743 ± 0.0008	0.9197 ± 0.0011	0.9462 ± 0.0008	0.9893 ± 0.0003
Custom MLP (Hand-coded)	0.9959 ± 0.0005	0.9927 ± 0.0007	0.9991 ± 0.0004	0.9959 ± 0.0005	0.9997 ± 0.0000
Keras MLP	0.9997 ± 0.0001	0.9994 ± 0.0001	1.0000 ± 0.0000	0.9997 ± 0.0001	0.9999 ± 0.0000
Key Findings
1. Linear models are insufficient for fraud detection

Logistic Regression misses ~8% of fraudulent transactions (Recall 92%)

In financial terms: 8 missed frauds per 100 attempts = significant losses

2. Custom backpropagation works correctly

Hand-coded MLP achieves 99.6% F1-score

Validates understanding of gradient flow through multiple layers

3. Framework optimizations provide marginal improvement

Keras MLP outperforms custom version by 0.4% in F1-score

Dropout and Adam optimizer contribute to better generalization

Performance Visualization
ROC Curves:

Custom MLP AUC: 0.9997

Keras MLP AUC: 0.9999

Logistic Regression AUC: 0.9893

Learning Curves:

Custom MLP converges after ~30 epochs

Keras MLP converges faster due to Adam optimization

Conclusion
Summary of Contributions
Demonstrated necessity of deep learning for fraud detection

8% recall gap between linear and non-linear models

Neural networks capture complex feature interactions

Successfully implemented backpropagation from scratch

Explicit chain rule application

Mini-batch gradient descent

ReLU and Sigmoid activations

Established fair comparison methodology

Stratified 5-fold cross-validation

SMOTE for class imbalance

Multiple evaluation metrics (focus on recall)

Why Neural Networks Outperform Linear Models
text
Linear Model Boundary:  ─────────────────
Fraud Pattern:          Small Amount + Unusual Time + New Merchant
                        (requires AND combination = non-linear)

Neural Network Solution:
    Layer 1: Detect individual signals (amount, time, merchant)
    Layer 2: Combine signals (small AND unusual)
    Layer 3: Final decision (combination AND new merchant)
Future Improvements
Direction	Description	Expected Benefit
Recurrent Networks	Model transaction sequences over time	Capture temporal fraud patterns
Anomaly Detection	Autoencoder for unsupervised fraud detection	Detect novel fraud types
Real-time Deployment	Optimize inference latency (< 100ms)	Production readiness
Lessons Learned
Class imbalance is critical - Accuracy is meaningless; use precision-recall metrics

Gradient checking validates backpropagation - Numerical vs. analytical gradients should match

Framework vs. hand-coded - Both achieve similar results; frameworks add convenience

