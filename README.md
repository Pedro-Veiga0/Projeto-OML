# Multiclass Logistic Classifier: OvO vs ECOC  
**University Project (Universidade do Minho)**

## 🎯 Overview
This project compares two multiclass classification strategies — **One-vs-One (OvO)** and **Error-Correcting Output Codes (ECOC)** — using **logistic regression** in both **primal** and **dual** forms (with polynomial kernels). The goal is to understand how these methods perform across synthetic and real datasets in terms of accuracy, computation time, and robustness.

All models were implemented **from scratch** using only `NumPy`, with a modular interface inspired by `scikit-learn`.

---

## 🧠 Implemented Models

### 🔹 Primal Logistic Classifier
- Mini-Batch Gradient Descent (MGmB)
- Cross-Entropy loss
- Tunable parameters: learning rate, batch size, number of iterations, convergence tolerance

### 🔸 Dual Logistic Classifier with Polynomial Kernel
- Kernelized optimization using the Gram matrix
- Polynomial kernel: K(x, x′) = (xᵀx′)^d
- Gradient descent on α coefficients instead of weights w

> Dual formulation enables non-linear decision boundaries without explicit feature transformation.

---

## 🔢 Classification Strategies

### ✅ One-vs-One (OvO)
- Trains C(k, 2) binary classifiers for k classes
- Predicts via **majority voting**
- Each model trained on a subset (2 classes)

### ✅ Error-Correcting Output Codes (ECOC)
- Encodes each class into a binary string
- Trains several binary classifiers based on ECOC matrix columns
- Prediction via **minimum Hamming distance** to estimated code

> ECOC offers error-tolerance, but with higher computational cost and memory usage.

---

## 🧪 Datasets

### 🔬 Synthetic Datasets
- `make_classification`, `make_gaussian_quantiles` from `sklearn`
- Custom "pizza-shaped" generator for visual class boundaries

### 🖼️ Real Dataset
- **Digits Dataset** (from `sklearn`)
  - 1797 grayscale 8x8 images (digits 0–9)
  - Reduced to 7 classes for ECOC compatibility

---

## 📊 Benchmark Results (Selected)

| Strategy | Form  | Kernel | Gradient | Accuracy (Test) | Train Time |
|----------|-------|--------|----------|-----------------|-------------|
| OvO      | Dual  | Poly 2 | MGmB     | **0.99**        | 6.2s       |
| ECOC     | Dual  | Poly 2 | MGmB     | 0.90            | 2.7s       |
| OvO      | Dual  | Linear | MGB      | **0.97**        | 2.6s       |
| ECOC     | Dual  | Linear | MGmB     | 0.94            | 9.1s       |
| OvO      | Primal| Linear | MGmB     | 0.72–0.87       | <1s        |
| ECOC     | Primal| Linear | MGB      | 0.59–0.67       | 1–2s       |

> Full benchmarking includes variations with 3, 5, and 7-class scenarios on both synthetic and real datasets.

---

## 🧠 Optimization Algorithms Compared
- **MGB**: Batch gradient
- **MGmB**: Mini-batch gradient
- **MGE**: Epoch-based mini-batch with shuffle

---

## 🔍 Insights

- OvO models generally outperform ECOC in both accuracy and efficiency
- Dual logistic classifiers significantly outperform primal versions in complex datasets
- ECOC is more robust to noisy or asymmetric datasets, especially when well-designed output codes are used
- Polynomial kernels introduce non-linearity, but can be unstable with poor parameter choices

---

## ⚙️ Usage

### 🔧 Setup

```bash
pip install numpy matplotlib scikit-learn
```

### ▶️ Running Models

```bash
# Train OvO classifier on digits dataset
python main.py --method ovo --form dual --kernel 1 --dataset digits

# Train ECOC classifier with polynomial kernel
python main.py --method ecoc --form dual --kernel 2 --dataset digits
```

> Full CLI supports configuration for learning rate, iterations, batch size, and verbose debugging.

---

## 🧰 Technologies Used
- Python 3
- NumPy (only library for model implementation)
- scikit-learn (for datasets)
- Matplotlib (for visualizations)