# AI-ML Recruitment 2026 — Task :Neural Network
> **Comprehensive Deep Learning Solution for Handwritten Digit Classification (MNIST)**  
> *Developed for Coding Ninjas / 10X Club SRM AI-ML Recruitment Drive*

[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge)]()
[![Task](https://img.shields.io/badge/Task-2%3A%20Neural%20Network-blue?style=for-the-badge)]()

---

## 1. Candidate Details
- **Candidate Name:** Harshita Khurana
- **GitHub Username:** hk301007
- **Target Organization:** Coding Ninjas / 10X Club SRM
- **Assigned Task Selected:** Task 2 (Neural Network)

---

## 2. Tasks Completed
- [x] **Task 2: Neural Network** (Completed with advanced ablation studies and error analysis)
  - [x] **Part A – Dataset Understanding**: 70k samples, 28x28 grayscale, 10 classes, semantic labels.
  - [x] **Part B – Data Preprocessing**: Min-Max scaling, Z-score standardization, flattening, train/val/test splits.
  - [x] **Part C – Build the Neural Network**: Modular PyTorch MLP with layer-by-layer parameter derivation.
  - [x] **Part D – Activation Functions**: Mathematical breakdown of ReLU, Dying ReLU nuance, and Softmax output.
  - [x] **Part E – Training Dynamics**: Per-epoch loss/accuracy recording, dual-axis learning curves.
  - [x] **Part F – Comprehensive Evaluation**: Test accuracy (~98%), Seaborn confusion matrix, classification report, qualitative misclassification analysis.
  - [x] **Part G – Scientific Experimentation**: 4-model ablation study (Depth, Activation, Optimizer)
---

## 3. Problem Statement
The objective of this project is to design, implement, train, and evaluate a feedforward neural network (Multi-Layer Perceptron) to classify handwritten digits from **0 to 9** from the classic **MNIST** database. 

Beyond simply achieving high accuracy, the project aims to:
1. Deconstruct the inner workings of artificial neural networks (forward pass, backpropagation, weight updates).
2. Empirically investigate how changes to activation functions, architectural depth, and optimization algorithms impact model convergence, training stability, and generalization.
3. Diagnose and analyze error modes through confusion matrix decomposition and misclassification inspection.

---

## 4. Approach & Methodology

```
+---------------------------------------------------------------------------------------------------+
|                                      PIPELINE ARCHITECTURE                                        |
+---------------------------------------------------------------------------------------------------+
|  [MNIST 28x28 Raw]                                                                                |
|          |                                                                                        |
|          v                                                                                        |
|  [Preprocessing: Normalization (mean=0.1307, std=0.3081) + Flatten (784-D)]                       |
|          |                                                                                        |
|          v                                                                                        |
|  [Data Splitting: Train (54,000) | Validation (6,000) | Test (10,000)]                            |
|          |                                                                                        |
|          +--------------------------------------------------------------------+                   |
|          |                                                                    |                   |
|          v                                                                    v                   |
|  [Baseline Model]                                                    [Ablation Suite]             |
|  Linear(784 -> 128) -> ReLU -> Linear(128 -> 10)                     1. Deeper (784->256->128->10)|
|          |                                                           2. Sigmoid Activation        |
|          v                                                           3. SGD with Momentum (0.9)   |
|  [Training: Cross-Entropy Loss + Adam (lr=1e-3)]                              |                   |
|          |                                                                    v                   |
|          v                                                           [Comparative Benchmark]      |
|  [Evaluation: Accuracy, Confusion Matrix, Precision/Recall/F1, Error Gallery]                      |
+---------------------------------------------------------------------------------------------------+
```

### Key Methodological Decisions:
1. **Object-Oriented PyTorch Implementation:** Modular subclassing of `nn.Module` ensuring reproducibility, device agility (CPU/CUDA), and clean parameter decoupling.
2. **Proper Data Hygiene (No Data Leakage):** The dataset was strictly partitioned into 54,000 training, 6,000 validation, and 10,000 test samples. Global normalization parameters $(\mu=0.1307, \sigma=0.3081)$ reflect the dataset distribution without peeking at test labels.
3. **Numerically Stable Cross-Entropy:** PyTorch's `nn.CrossEntropyLoss` computes the Log-Sum-Exp trick directly on raw output logits, eliminating float underflow/overflow associated with calculating raw probabilities before logarithms.
4. **Diagnostic Error Analysis:** In addition to macro metrics, we systematically extracted and rendered the most confident misclassified digits to analyze handwritten stroke ambiguities.

---

## 5. Technologies Used
- **Core Deep Learning Framework:** PyTorch 2.x (`torch`, `torch.nn`, `torch.optim`, `torchvision`)
- **Scientific Computing & Data Manipulation:** NumPy, SciPy, Pandas
- **Model Evaluation & Metrics:** Scikit-Learn (`confusion_matrix`, `classification_report`, `accuracy_score`)
- **Data Visualization & Plotting:** Matplotlib, Seaborn
- **Interactive Development:** Jupyter Notebook (`.ipynb`)
- **Language & Runtime:** Python 3.14 (fully compatible with Python 3.10+)

---

## 6. Results & Experimental Outcomes

### A. Baseline Model Performance (10,000 Unseen Test Images)
- **Test Accuracy:** **97.70%**
- **Macro Precision:** **97.68%**
- **Macro Recall:** **97.69%**
- **Macro F1-Score:** **97.68%**
- **Total Misclassifications:** 230 out of 10,000 test samples (Error Rate: **2.30%**)

#### Per-Class Performance Breakdown:
| Digit Class | Precision | Recall | F1-Score | Support |
|:---:|:---:|:---:|:---:|:---:|
| **0** | 0.9808 | 0.9908 | 0.9858 | 980 |
| **1** | 0.9885 | 0.9885 | 0.9885 | 1,135 |
| **2** | 0.9741 | 0.9835 | 0.9788 | 1,032 |
| **3** | 0.9723 | 0.9743 | 0.9733 | 1,010 |
| **4** | 0.9784 | 0.9664 | 0.9723 | 982 |
| **5** | 0.9743 | 0.9787 | 0.9765 | 892 |
| **6** | 0.9800 | 0.9739 | 0.9770 | 958 |
| **7** | 0.9891 | 0.9679 | 0.9784 | 1,028 |
| **8** | 0.9585 | 0.9713 | 0.9648 | 974 |
| **9** | 0.9723 | 0.9732 | 0.9728 | 1,009 |
| **Macro Average** | **0.9768** | **0.9769** | **0.9768** | **10,000** |

---

### B. Part G: Multi-Model Ablation Study & Comparison
We evaluated four architectural and optimization variations under identical train/validation splits for 10 epochs:

| Experiment Model | Hidden Layers / Units | Activation | Optimizer | Parameters | Val Acc (%) | Test Acc (%) | Train Time |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Baseline MLP** | 1 layer (128) | **ReLU** | Adam (1e-3) | 101,770 | **97.73%** | **97.70%** | ~143s |
| **Deeper MLP** | 2 layers (256 $\to$ 128) | **ReLU** | Adam (1e-3) | 235,146 | **97.62%** | **97.92%** | ~155s |
| **Sigmoid MLP** | 1 layer (128) | **Sigmoid** | Adam (1e-3) | 101,770 | **97.30%** | **97.60%** | ~138s |
| **Baseline + SGD** | 1 layer (128) | **ReLU** | SGD-M (1e-2) | 101,770 | **97.77%** | **97.82%** | ~263s |

---

### C. Visual Assets Generated
All generated high-resolution assets are saved in the `assets/` directory:
- `assets/sample_digits.png`: 4x4 visual grid of MNIST digits with labels.
- `assets/baseline_learning_curves.png`: Training vs Validation loss and accuracy curves across 10 epochs.
- `assets/confusion_matrix.png`: Annotated 10x10 Seaborn heatmap with true vs predicted counts.
- `assets/misclassified_digits.png`: Top misclassified test examples showing true label, predicted label, and confidence score.
- `assets/experiment_comparison.png`: Trajectory overlay and final accuracy bar chart across all 4 ablation models.

---

## 7. Key Learnings
1. **The Crucial Role of Activation Function Derivatives:**
   Comparing ReLU with Sigmoid empirically proved the impact of gradient saturation. Even in a shallow single-hidden-layer network, Sigmoid's maximum gradient of $0.25$ noticeably hindered backpropagation efficiency, whereas ReLU's constant gradient of $1.0$ for $z > 0$ enabled rapid, steady convergence.
2. **The Power of Adaptive Optimizers vs Momentum SGD:**
   While SGD with Momentum can reach comparable accuracy, it is highly sensitive to the chosen learning rate. Adam's per-parameter adaptive second-moment scaling ($v_t$) made training remarkably robust and accelerated convergence within the first 3 epochs.
3. **Interpreting Confusion Matrices for Qualitative Error Diagnostics:**
   Examining the confusion matrix revealed that classification errors are not uniformly distributed; errors strongly cluster around morphological similarities (e.g., digit **4 vs 9**, digit **3 vs 5**, and digit **7 vs 2**). Inspecting misclassified samples showed that many errors correspond to genuine human handwriting ambiguity (e.g., broken loops or heavy slants).

---

## 8. Challenges & Solutions

### Challenge 1: Logit Overflow & Numerical Instability in Softmax
- **The Issue:** Naively computing Softmax $\sigma(z)_i = \frac{e^{z_i}}{\sum e^{z_j}}$ followed by Categorical Cross-Entropy $-\sum y_i \log(\hat{y}_i)$ can result in floating-point overflow ($e^{z_i} \to \infty$) for large logits or catastrophic underflow ($\log(0) \to -\infty$).
- **The Solution:** Kept the model's output layer strictly as raw unnormalized linear logits, and utilized PyTorch's unified `nn.CrossEntropyLoss`. Under the hood, this uses the mathematically equivalent **Log-Sum-Exp** identity:
  $$\log \sum_{j} e^{z_j} = c + \log \sum_j e^{z_j - c} \quad \text{where } c = \max_j(z_j)$$
  This subtracts the maximum logit before exponentiation, guaranteeing numerical stability.

### Challenge 2: Risk of Data Leakage & Overfitting
- **The Issue:** Evaluating hyperparameter choices directly on the 10,000-sample test set risks information leakage, where design choices overfit the test partition.
- **The Solution:** Enforced a strict three-way split using `torch.utils.data.random_split`: 54,000 samples for training, 6,000 for validation (epoch-level checkpointing and hyperparameter validation), and reserved the 10,000 test set images solely for final, unbiased evaluation.

---

## 9. Project Directory Structure
```
AIML-Recruitment-2026-NeuralNetwork/
├── README.md                      # Comprehensive recruitment report (this file)
├── requirements.txt               # Reproducible environment dependencies
├── .gitignore                     # Git ignore rules for checkpoints and data
├── run_pipeline.py                # Master CLI script executing all experiments & plots
├── build_notebook.py              # Generates the self-contained Jupyter notebook
├── notebooks/
│   └── mnist_neural_network.ipynb # Rich interactive notebook with all theory & outputs
├── src/
│   ├── __init__.py
│   ├── dataset.py                 # Data loading, normalization, train/val splits
│   ├── models.py                  # BaselineMLP, DeeperMLP, SigmoidMLP, DropoutMLP
│   ├── train.py                   # PyTorch training & validation loops
│   ├── evaluate.py                # Confusion matrix, metrics, error extractor
│   └── visualize.py               # Publication-grade plotting routines
└── assets/
    ├── sample_digits.png          # MNIST sample inspection grid
    ├── baseline_learning_curves.png # Loss and accuracy dynamics
    ├── confusion_matrix.png       # Test set confusion matrix heatmap
    ├── misclassified_digits.png   # Gallery of misclassified images
    └── experiment_comparison.png  # Ablation study comparison plots
```

---
