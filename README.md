# Credit Card Fraud Detection: Custom Backpropagation vs. Keras MLP

**Team Member:** Lei Lei  
**Email:** lei.l2@northeastern.edu

---

## Introduction

Credit card fraud causes billions of dollars in annual losses worldwide. Financial institutions need accurate, real-time fraud detection systems to protect their customers and minimize financial risk. Traditional rule-based systems fail to capture complex, evolving fraud patterns, making machine learning approaches essential.

This project compares three machine learning models for fraud detection:

- **Logistic Regression** (baseline linear model)
- **Custom Multi-Layer Perceptron** (hand-coded backpropagation)
- **Keras MLP** (optimized framework implementation)

The primary goal is to demonstrate that deep neural networks, capable of learning non-linear feature interactions, significantly outperform linear models on this task. Additionally, the custom backpropagation implementation demonstrates a deep understanding of gradient flow through multiple network layers.

---

## Problem Statement

The Credit Card Fraud Detection dataset contains 284,807 transactions, of which only 492 (0.172%) are fraudulent. This presents three major challenges:

| Challenge | Description | Impact |
|-----------|-------------|--------|
| **Severe Class Imbalance** | Only 0.17% of transactions are fraud | Standard accuracy is misleading (99.8% by predicting all legitimate) |
| **Non-linear Fraud Patterns** | Fraud involves complex feature interactions (e.g., small amount × unusual time × new merchant) | Linear models cannot capture these patterns |
| **High Cost of False Negatives** | Missing a fraudulent transaction causes direct financial loss | Models must prioritize recall over raw accuracy |

**Key Research Question:** Can multi-layer neural networks, with hand-coded backpropagation, significantly outperform linear baselines on this highly imbalanced fraud detection task?

---

## Solution Design and Methodology

### Dataset Preprocessing
Raw Data (284,807 × 31) <br/>
│ <br/>
├── Scale 'Time' and 'Amount' features <br/>
├── Drop original unscaled columns <br/>
└── Result: 30 features (V1-V28 + scaled_time + scaled_amount) <br/>
SMOTE Oversampling
│ <br/>
├── Balance classes (284,315 fraud + 284,315 legitimate) <br/>
└── Result: 568,630 balanced samples <br/>

### Model Architecture

**Model 1: Logistic Regression (Baseline)**
- Linear decision boundary
- Serves as performance lower bound

**Model 2: Custom MLP (Hand-coded Backpropagation)**
Input (30) → Hidden1 (32, ReLU) → Hidden2 (16, ReLU) → Output (1, Sigmoid)

text
- Forward pass: manual matrix operations
- Backward pass: explicit chain rule implementation
- Mini-batch gradient descent (batch_size=64)

**Model 3: Keras MLP (Framework)**
Input (30) → Dense(64, ReLU) → Dropout(0.3) → Dense(32, ReLU) → Dropout(0.3) → Output(1, Sigmoid)

text
- Adam optimizer
- Dropout for regularization
- Early stopping (patience=10)

### Backpropagation Implementation (Custom MLP)

**Forward Pass:**
z[l] = W[l] · a[l-1] + b[l]
a[l] = g(z[l]) # ReLU for hidden, Sigmoid for output

text

**Backward Pass (Chain Rule):**
δ[L] = a[L] - y # Output layer error
δ[l] = (δ[l+1] · W[l+1]^T) ⊙ g'(z[l]) # Hidden layer error

∂L/∂W[l] = a[l-1]^T · δ[l] / m # Weight gradient (average)
∂L/∂b[l] = Σ δ[l] / m # Bias gradient (average)

text

### Evaluation Strategy

| Metric | Why Used | Target |
|--------|----------|--------|
| **Precision** | Minimize false alarms | > 0.99 |
| **Recall** | Catch fraudulent transactions | > 0.99 |
| **F1-Score** | Harmonic mean (balanced view) | > 0.99 |
| **AUC-PR** | Best for imbalanced data | > 0.99 |

**Validation:** Stratified 5-Fold Cross-Validation (preserves fraud ratio in each fold)

---

## Implementation and Execution

### Environment Setup

```bash
# Virtual environment (Python 3.12)
python3 -m venv fraud_detect
source fraud_detect/bin/activate
```

# Key packages

pip install numpy pandas scikit-learn tensorflow matplotlib seaborn imbalanced-learn

## Custom MLP Core Implementation
python
```
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
```

## Evaluation and Results

### 5-Fold Cross-Validation Results

| Model | Accuracy | Precision | Recall | F1-Score | AUC |
|-------|----------|-----------|--------|----------|-----|
| **Logistic Regression** | 0.9478 ± 0.0008 | 0.9743 ± 0.0008 | 0.9197 ± 0.0011 | 0.9462 ± 0.0008 | 0.9893 ± 0.0003 |
| **Custom MLP (Hand-coded)** | 0.9959 ± 0.0005 | 0.9927 ± 0.0007 | 0.9991 ± 0.0004 | 0.9959 ± 0.0005 | 0.9997 ± 0.0000 |
| **Keras MLP** | 0.9997 ± 0.0001 | 0.9994 ± 0.0001 | 1.0000 ± 0.0000 | 0.9997 ± 0.0001 | 0.9999 ± 0.0000 |

---

### Key Findings

**1. Linear models are insufficient for fraud detection**

| Metric | Logistic Regression | Custom MLP | Improvement |
|--------|---------------------|------------|-------------|
| Recall | 91.97% | 99.91% | **+7.94%** |
| F1-Score | 94.62% | 99.59% | **+4.97%** |

> Logistic Regression misses ~8% of fraudulent transactions. In financial terms: 8 missed frauds per 100 attempts = significant losses.

**2. Custom backpropagation works correctly**
- Hand-coded MLP achieves 99.6% F1-score
- Validates understanding of gradient flow through multiple layers

**3. Framework optimizations provide marginal improvement**

| Comparison | Custom MLP | Keras MLP | Gain |
|------------|------------|-----------|------|
| F1-Score | 99.59% | 99.97% | **+0.38%** |
| AUC | 99.97% | 99.99% | **+0.02%** |

---

### Performance vs. Resource Trade-off

**Q: Does Keras lead to increased time/memory cost? How do we balance performance and resources?**

**A: Here's our empirical comparison from training on a standard CPU (no GPU):**

| Aspect | Custom MLP (NumPy) | Keras MLP (TensorFlow) |
|--------|-------------------|------------------------|
| **Training Time (per fold)** | ~8-10 minutes | ~3-4 minutes |
| **Total Training Time (5-fold CV)** | ~45 minutes | ~18 minutes |
| **Memory Usage** | ~2-3 GB | ~1-2 GB |
| **Inference Latency (per transaction)** | ~5 ms | ~8 ms |
| **Model Size** | ~0.5 MB | ~1.2 MB |

**Key Observations:**

| Finding | Explanation |
|---------|-------------|
| **Keras trains faster** | Optimized backend (oneDNN, AVX2) and graph compilation |
| **Keras uses less memory** | Lazy evaluation and efficient memory management |
| **Inference slightly slower** | Framework overhead for single predictions |
| **Custom MLP has smaller footprint** | No additional library dependencies |

**Trade-off Analysis:**
Training Time Comparison (5-fold CV)
───────────────────────────────────────── <br/>
Keras MLP: ████████████████████████████████████ (97s per fold CV) <br/>
Custom MLP: ████████████████ (41s per fold CV)

Memory Usage
───────────────────────────────────────── <br/>
Custom MLP: ████████████████████ (2.5 GB) <br/>
Keras MLP: ██████████████ (1.5 GB)

Inference Latency (per transaction) <br/>
───────────────────────────────────────── <br/>
Custom MLP: ████ (5 ms) <br/>
Keras MLP: ██████ (8 ms) <br/>


| Scenario | Recommended Model | Reasoning |
|----------|-------------------|-----------|
| **Batch inference (thousands of transactions)** | Keras | Faster processing, better optimization |
| **Real-time single transaction** | Custom MLP | Lower per-prediction latency |
| **Resource-constrained environment** | Custom MLP | Smaller binary size, no TF dependency |
| **Production with high throughput** | Keras | Better CPU utilization, faster training |

**Conclusion on Trade-off:**

> "While Keras trains 2-3x faster and uses less memory during training, it has slightly higher per-inference latency due to framework overhead. For our fraud detection scenario where we process transactions in real-time, the 3ms difference is negligible compared to network latency. The 0.38% F1 improvement from Keras justifies its use. However, if deploying on embedded systems, the lightweight Custom MLP might be preferable."

**Training Time Breakdown (per fold, 50 epochs, 454K samples):**

### Future Improvements

| Direction | Description | Expected Benefit |
|-----------|-------------|------------------|
| **Recurrent Networks** | Model transaction sequences over time | Capture temporal fraud patterns |
| **Anomaly Detection** | Autoencoder for unsupervised fraud detection | Detect novel fraud types |
| **Real-time Deployment** | Optimize inference latency (< 100ms) | Production readiness |

### Lessons Learned

1. **Class imbalance is critical** - Accuracy is meaningless; use precision-recall metrics
2. **Gradient checking validates backpropagation** - Numerical vs. analytical gradients should match
3. **Framework vs. hand-coded** - Both achieve similar results; frameworks add convenience

---

## References

1. Kaggle Credit Card Fraud Detection Dataset: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

2. Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). "SMOTE: Synthetic Minority Over-sampling Technique." Journal of Artificial Intelligence Research, 16, 321-357.

3. Goodfellow, I., Bengio, Y., & Courville, A. (2016). "Deep Learning." MIT Press. (Chapter 6: Backpropagation)

4. TensorFlow Documentation: https://www.tensorflow.org/

5. Kingma, D. P., & Ba, J. (2015). "Adam: A Method for Stochastic Optimization." ICLR.