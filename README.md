# AI-ML Recruitment 2026 — Task 2: Neural Network
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
- **Recruitment Track:** AI-ML Recruitment — Second Years (2026)
- **Assigned Task Selected:** Task 2 (Neural Network)

---

## 2. Tasks Completed
- [x] **Task 1: Air Quality Forecasting** (Not selected — specialized in Deep Learning / Neural Networks)
- [x] **Task 2: Neural Network** (Completed with advanced ablation studies and error analysis)
  - [x] **Part A – Dataset Understanding**: 70k samples, 28x28 grayscale, 10 classes, semantic labels.
  - [x] **Part B – Data Preprocessing**: Min-Max scaling, Z-score standardization, flattening, train/val/test splits.
  - [x] **Part C – Build the Neural Network**: Modular PyTorch MLP with layer-by-layer parameter derivation.
  - [x] **Part D – Activation Functions**: Mathematical breakdown of ReLU, Dying ReLU nuance, and Softmax output.
  - [x] **Part E – Training Dynamics**: Per-epoch loss/accuracy recording, dual-axis learning curves.
  - [x] **Part F – Comprehensive Evaluation**: Test accuracy (~98%), Seaborn confusion matrix, classification report, qualitative misclassification analysis.
  - [x] **Part G – Scientific Experimentation**: 4-model ablation study (Depth, Activation, Optimizer) answering all 3 mandatory analysis questions.
  - [x] **Bonus**: Viva / Club Interview Preparation Cheatsheet.

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

#### Mandatory Part G Questions Answered:
1. **What did you change?**
   - *Architecture Depth:* Added a second hidden layer ($784 \to 256 \to 128 \to 10$), expanding parameters from 101k to 235k (+131%).
   - *Activation Function:* Replaced non-saturating ReLU with smooth Sigmoid ($\sigma(z) = \frac{1}{1 + e^{-z}}$) while keeping learning rates and optimizers identical.
   - *Optimization Algorithm:* Replaced adaptive moment estimation (Adam) with classical Stochastic Gradient Descent with Nesterov Momentum ($\mu = 0.9, \text{lr} = 0.01$).
2. **What changed in the results?**
   - The **Deeper MLP** achieved the highest accuracy (**97.92%**), reducing test error from 2.30% to 2.08%.
   - The **Sigmoid MLP** suffered a notable accuracy drop down to **97.60%** and exhibited sluggish loss reduction in early epochs compared to ReLU.
   - **SGD with Momentum** achieved competitive accuracy (**97.82%**) but required a 10x higher learning rate ($0.01$) to escape initial flat loss regions.
3. **Why do you think the change affected the model's performance?**
   - *Deeper Hierarchy:* Hierarchical composition allows the first layer to learn edge/stroke orientations, while the second layer combines these primitives into localized digit loops and junctions.
   - *Vanishing Gradients in Sigmoid:* Sigmoid derivatives peak at only $0.25$ and shrink exponentially to $0$ as $|z|$ grows. Gradients flowing back to the input layer were attenuated, suppressing effective weight updates.
   - *Adaptive Moments in Adam:* Adam scales individual parameter updates inversely with gradient magnitudes, effortlessly navigating anisotropic curvatures without manual learning-rate schedules.

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

## 10. How to Run & Reproduce

### Step 1: Clone Repository & Install Dependencies
```bash
git clone https://github.com/<YourUsername>/AIML-Recruitment-2026-<YourName>.git
cd AIML-Recruitment-2026-<YourName>
pip install -r requirements.txt
```

### Step 2: Run the Full End-to-End Pipeline (CLI)
To train all models, generate comparison charts, and save checkpoints:
```bash
python run_pipeline.py
```

### Step 3: Run / View the Interactive Jupyter Notebook
Open the pre-computed notebook in JupyterLab, VS Code, or Google Colab:
```bash
jupyter notebook notebooks/mnist_neural_network.ipynb
```

---

## 11. Bonus: Club Interview Preparation Cheatsheet

<details>
<summary><b>Click to expand 10 High-Yield Viva / Technical Interview Questions & Answers</b></summary>

<br>

1. **Q: Why don't we use Mean Squared Error (MSE) for multi-class classification?**  
   *A:* MSE assumes errors follow a Gaussian distribution. When paired with Softmax/Sigmoid, large prediction errors cause the gradient to approach zero (saturation plateau), slowing down learning. Cross-Entropy is derived from Maximum Likelihood Estimation; its gradient $\nabla_{z_i} \mathcal{L} = p_i - y_i$ is linear with respect to the error, providing strong corrective signals.

2. **Q: What is the "Dying ReLU" problem, and how do you resolve it?**  
   *A:* If a neuron's weights are updated such that its pre-activation $z = Wx+b \le 0$ across all training samples, its output and derivative are permanently 0. It will never update again. Solutions include using **Leaky ReLU** ($f(x) = \max(\alpha x, x)$ with $\alpha \approx 0.01$), **Parametric ReLU (PReLU)**, or **He (Kaiming) weight initialization**.

3. **Q: Why do we initialize weights with random values instead of zeros?**  
   *A:* Initializing weights to zero causes all hidden neurons in a layer to compute the exact same activation and receive the exact same gradient update during backpropagation (**symmetry problem**). The network would behave as if each layer had only 1 neuron. Random initialization breaks this symmetry.

4. **Q: Why is Adam optimizer usually preferred over vanilla SGD?**  
   *A:* Adam combines two key ideas: Momentum (first moment, exponential moving average of past gradients to smooth oscillations) and RMSProp (second moment, scaling learning rates inversely by root mean square of gradients). This allows it to make rapid progress on flat surfaces while avoiding divergence on steep ravines.

5. **Q: What is the Universal Approximation Theorem?**  
   *A:* Proved by Cybenko (1989) and Hornik (1991), it states that a feedforward network with a single hidden layer containing a finite number of neurons and non-linear activation functions can approximate any continuous function on compact subsets of $\mathbb{R}^n$ to arbitrary precision.

6. **Q: What is the difference between Batch Gradient Descent, Mini-Batch GD, and SGD?**  
   *A:* Batch GD computes gradients over the entire dataset (stable but slow, doesn't fit in RAM). Stochastic GD computes gradients per single sample (fast, but very noisy trajectory). Mini-Batch GD (e.g. batch size 64) strikes the optimal balance: vectorized hardware efficiency and smooth stochastic gradients that help escape local minima.

7. **Q: What is the difference between $L_1$ and $L_2$ regularization?**  
   *A:* $L_1$ regularization adds penalty $\lambda \sum |w_i|$, driving less important weights to exactly zero, producing sparse models (feature selection). $L_2$ regularization (weight decay) adds penalty $\frac{1}{2} \lambda \sum w_i^2$, shrinking weights towards zero without setting them exactly to zero, preventing dominant weights.

8. **Q: How does Dropout work during training vs testing?**  
   *A:* During training, Dropout randomly deactivates each neuron with probability $p$, preventing co-adaptation of features. During testing/inference, Dropout is disabled (`model.eval()`), and neuron outputs are scaled by $(1-p)$ (or inverted dropout scales by $\frac{1}{1-p}$ during training) to preserve expected activation values.

9. **Q: What are the fundamental limitations of an MLP on computer vision tasks?**  
   *A:* MLPs flatten images into 1D vectors, destroying 2D spatial locality and neighborhood relationships. They lack **translation invariance** (a digit shifted by 3 pixels appears as completely different features). Convolutional Neural Networks (CNNs) solve this via weight sharing and local receptive fields.

10. **Q: Why do we normalize MNIST with mean 0.1307 and std 0.3081?**  
    *A:* 0.1307 and 0.3081 are the exact pre-calculated global mean and standard deviation of all pixel intensities across the 60,000 MNIST training images. Standardizing by these statistics centers the data cloud at origin $(0, 0)$ with unit variance, ensuring spherical loss contours.

</details>

---

## 12. Submission Verification Checklist
- [x] Repository follows naming convention: `AIML-Recruitment-2026-<YourName>`
- [x] Public GitHub repository with clean `.gitignore`
- [x] Executable code and pre-rendered Jupyter Notebook
- [x] All 8 required README sections included and detailed
- [x] All visual figures present and saved in `assets/`
- [x] 100% reproducible with fixed random seed 42
