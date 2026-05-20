

# 🧠 Robust Attention State Prediction from EEG Signals Using Meta-Learning and Uncertainty-Aware Deep Learning



<br/>

<table>
<tr>
<td align="center"><b>🎯 97.4%</b><br/><sub>Classification Accuracy</sub></td>
<td align="center"><b>🔬 9</b><br/><sub>Subjects</sub></td>
<td align="center"><b>📊 418</b><br/><sub>Epochs</sub></td>
<td align="center"><b>🧪 4</b><br/><sub>Paradigms</sub></td>
<td align="center"><b>📉 p < 10⁻⁶</b><br/><sub>Uncertainty Separation</sub></td>
</tr>
</table>

<br/>

> *A two-stage meta-learning framework for EEG-based attention state classification with integrated predictive uncertainty quantification — validated across subjects and experimental paradigms using Leave-One-Subject-Out cross-validation.*

<br/>

</div>

---


## 🔍 Overview

EEG-based attention monitoring systems are increasingly critical for safety applications such as driver monitoring and human-machine interaction. Current approaches face three fundamental challenges — this work addresses all three:

<br/>

| ⚠️ Challenge | ✅ This Work's Solution |
|:---|:---|
| Single-paradigm training limits real-world generalizability | Multi-paradigm framework combining oddball and naturalistic video tasks |
| Predefined attention labels introduce circular reasoning | Data-driven state discovery via unsupervised clustering on ground-truth spectral features |
| No uncertainty quantification on predictions | Model-specific uncertainty methods with statistically verified calibration (p < 10⁻⁶) |

<br/>

EEG data was collected from **9 subjects** across **4 experimental paradigms**. Three neurophysiologically distinct attention states were discovered through unsupervised clustering and subsequently classified using a meta-learning pipeline that combines LightGBM spectral predictions with contextual paradigm information.

---

## 📁 Repository Contents

```
📦msc-thesis-eeg-attention
 ├── 📓 Thesis_code.ipynb     ← Full implementation pipeline
 └── 📄 Thesis_Final.pdf      ← Complete written thesis
```

| File | Description |
|:---|:---|
| `Thesis_code.ipynb` | End-to-end implementation: EEG preprocessing, feature extraction, unsupervised clustering, LightGBM spectral specialist, three meta-learners, and model-specific uncertainty quantification |
| `Thesis_Final.pdf` | Full written thesis including literature review, methodology, results, and discussion |

---

## 🏗️ Framework Architecture

The pipeline operates across three sequential stages:

```
╔══════════════════════════════════════════════════════════════════╗
║                        RAW EEG SIGNAL                           ║
║              16 channels  ·  125 Hz  ·  9 subjects              ║
╚═════════════════════════════╦════════════════════════════════════╝
                              ║
                              ▼
╔══════════════════════════════════════════════════════════════════╗
║               STAGE 1  ·  PREPROCESSING & FEATURES             ║
║                                                                  ║
║  Butterworth filter (0.5–45 Hz)  →  Z-score normalization       ║
║  Epoching (1s windows)           →  Quality assurance           ║
║                                                                  ║
║  Feature Extraction  →  59-dimensional spectral vector          ║
║  ├── Global band powers (5)                                      ║
║  ├── Relative band powers (5)                                    ║
║  ├── ROI band powers across 7 cortical regions (35)              ║
║  ├── Alpha/beta & theta/beta ratios (2)                          ║
║  └── Hemispheric asymmetry + aliases (12)                        ║
╚════════════╦════════════════════════════════╦═════════════════════╝
             ║                                ║
             ▼                                ▼
╔════════════════════════╗     ╔══════════════════════════════════╗
║  ATTENTION STATE       ║     ║  SPECTRAL SPECIALIST             ║
║  DISCOVERY             ║     ║  LightGBM Multi-Output Regressor ║
║                        ║     ║                                  ║
║  k-means clustering    ║     ║  Input:  59-dim feature vector   ║
║  on 7-dim target       ║     ║  Output: 7-dim target vector     ║
║  spectral features     ║     ║                                  ║
║                        ║     ║  R² > 0.94  for 5 band powers   ║
║  3 states found:       ║     ║  R² = 0.75  for α/β ratio       ║
║  · Baseline Attention  ║     ║  Mean MAE = 0.122 (LOSO)        ║
║  · Low Arousal         ║     ╚══════════════╦═══════════════════╝
║  · High Attention      ║                    ║
╚════════════╦═══════════╝                    ║ + 9 contextual
             ║  cluster labels                ║   one-hot features
             ╚═══════════════╦════════════════╝
                             ║
                             ▼
╔══════════════════════════════════════════════════════════════════╗
║            STAGE 2  ·  META-LEARNERS  (16-dim input)            ║
║                                                                  ║
║   Logistic Regression     XGBoost          Transformer          ║
║   Accuracy: 97.4%         Accuracy: 96.4%  Accuracy: 95.7%      ║
║   F1-Macro: 0.972         F1-Macro: 0.963  F1-Macro: 0.956      ║
╚══════════════════════════════╦═══════════════════════════════════╝
                               ║
                               ▼
╔══════════════════════════════════════════════════════════════════╗
║           STAGE 3  ·  UNCERTAINTY QUANTIFICATION                ║
║                                                                  ║
║  Platt Scaling           Ensemble Entropy    MC Dropout          ║
║  (Logistic Regression)   (XGBoost)           (Transformer)       ║
║                                                                  ║
║  Incorrect predictions exhibit 2–4× higher uncertainty          ║
║  across all models  ·  p < 10⁻⁶  ·  Cohen's d > 1.7            ║
╚══════════════════════════════════════════════════════════════════╝
```

---

## 🧩 Discovered Attention States

Three neurophysiologically distinct states identified through k-means clustering on ground-truth EEG spectral features:

> **Clustering quality:** Calinski-Harabasz Index = **122.89** &nbsp;·&nbsp; Silhouette Coefficient = **0.266**

<br/>

<table>
<thead>
<tr>
<th align="center">State</th>
<th>Label</th>
<th align="center">Alpha/Beta Ratio</th>
<th align="center">Dominant Band</th>
<th>Neurophysiological Profile</th>
<th align="center">Data Share</th>
</tr>
</thead>
<tbody>
<tr>
<td align="center">🔵 <b>0</b></td>
<td><b>Baseline Attention</b></td>
<td align="center">1.91 ± 1.34</td>
<td align="center">Theta (θ)</td>
<td>Moderate engagement, routine processing</td>
<td align="center">30.6%</td>
</tr>
<tr>
<td align="center">⚪ <b>1</b></td>
<td><b>Low Arousal</b></td>
<td align="center">1.06 ± 0.83</td>
<td align="center">Delta (δ)</td>
<td>Reduced alertness, drowsiness</td>
<td align="center">28.9%</td>
</tr>
<tr>
<td align="center">🔴 <b>2</b></td>
<td><b>High Attention</b></td>
<td align="center">0.52 ± 0.31</td>
<td align="center">Beta (β)</td>
<td>Focused engagement, active cognitive processing</td>
<td align="center">40.4%</td>
</tr>
</tbody>
</table>

<br/>

> States are ordered by alpha/beta ratio (high → low). High α/β indicates relaxation; low α/β indicates focused attention — consistent with established EEG neuroscience literature.

---

## 📊 Key Results

### Spectral Specialist — LightGBM (LOSO Cross-Validation)

| Prediction Target | R² Score | MAE | Quality |
|:---|:---:|:---:|:---:|
| Delta band power | 0.989 | 0.009 | 🟢 Excellent |
| Theta band power | 0.903 | 0.005 | 🟢 Excellent |
| Alpha band power | 0.974 | 0.006 | 🟢 Excellent |
| Beta band power | 0.969 | 0.006 | 🟢 Excellent |
| Gamma band power | 0.945 | 0.003 | 🟢 Excellent |
| Alpha/Beta ratio | 0.750 | 0.146 | 🟡 Good |
| Theta/Beta ratio | 0.247 | 0.681 | 🟠 Moderate |

> 8/9 subjects achieved R² > 0.83 &nbsp;·&nbsp; Overall mean R² = **0.837** &nbsp;·&nbsp; Overall mean MAE = **0.122**

---

### Meta-Learner Classification (LOSO Cross-Validation)

| Model | Accuracy | Balanced Accuracy | F1-Macro |
|:---|:---:|:---:|:---:|
| 🥇 Logistic Regression | **97.4%** | **97.4%** | **0.972** |
| 🥈 XGBoost | 96.4% | 96.4% | 0.963 |
| 🥉 Transformer | 95.7% | 95.8% | 0.956 |

> Near-identical accuracy and balanced accuracy for all models confirm no majority-class bias. The superior performance of Logistic Regression indicates the discovered attention states are approximately **linearly separable** in the 16-dimensional meta-feature space.

---

### Uncertainty Quantification — Correct vs. Incorrect Predictions

| Model | Method | Correct | Incorrect | Difference | p-value | Cohen's d |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| Logistic Regression | Platt Scaling + Entropy | 0.336 ± 0.265 | 0.777 ± 0.134 | +0.441 | 1.27 × 10⁻⁶ | 2.096 |
| XGBoost | Ensemble Entropy | 0.124 ± 0.212 | 0.512 ± 0.210 | +0.388 | 7.43 × 10⁻⁹ | 1.842 |
| Transformer | MC Dropout (50 passes) | 0.046 ± 0.124 | 0.372 ± 0.236 | +0.326 | 6.66 × 10⁻⁸ | 1.732 |

> All Cohen's d values exceed 0.8 (large effect threshold). Incorrect predictions show **2–4× higher uncertainty** across all three models, confirming reliable calibration for safety-critical deployment.

---

## 🛠️ Technical Stack

| Category | Tools / Methods |
|:---|:---|
| **Language** | Python 3 |
| **EEG Signal Processing** | Welch's PSD estimation, Butterworth band-pass filter (0.5–45 Hz), z-score normalization, epoch segmentation (1s windows) |
| **Machine Learning** | LightGBM, XGBoost, scikit-learn (Logistic Regression, MultiOutputRegressor, GridSearchCV) |
| **Deep Learning** | PyTorch — custom Transformer encoder (2 layers, 4 attention heads, 128-dim FFN, MC Dropout) |
| **Uncertainty Quantification** | Platt scaling (LR), ensemble entropy (XGBoost), Monte Carlo Dropout — 50 forward passes (Transformer) |
| **Clustering** | k-means with Calinski-Harabasz Index and Silhouette Coefficient validation |
| **Statistical Testing** | Mann-Whitney U test, independent t-test, one-way ANOVA, Shapiro-Wilk, Levene's test, Cohen's d |
| **Cross-Validation** | Leave-One-Subject-Out (LOSO) — 9 folds |
| **Visualization** | Matplotlib, Seaborn |

---

## 🔬 Experimental Setup

| Parameter | Detail |
|:---|:---|
| Participants | 9 subjects, ages 20–40 |
| EEG Hardware | OpenBCI-based system, 16 active electrodes |
| Electrode Placement | International 10-20 system · Reference electrode: Fz |
| Sampling Rate | 125 Hz |
| Regions Covered | Frontal, central, parietal, temporal, occipital |
| Total Epochs | 418 event-locked epochs |
| Epoch Length | 1 second (125 samples per channel) |
| ERP Window | −200 ms pre-stimulus to +800 ms post-stimulus |

<br/>

**Experimental Paradigms:**

| Paradigm | Events (n) | Purpose |
|:---|:---:|:---|
| 🟡 Oddball Task | Target: 76 · Non-target: 103 | P300 evocation, controlled attention marker |
| 🔴 Car Crash Video | 134 | High-arousal, externally directed attention |
| 🟢 Animal Surprise Video | 84 | Moderate naturalistic attentional engagement |
| 🟠 Surprise Events Video | 21 | Unexpected stimuli, attentional spike induction |

---

## 💡 Research Contributions

**1. Data-driven attention labeling**
Unsupervised clustering on ground-truth EEG spectral features was used to discover attention states from the data itself, avoiding the circular reasoning inherent in predefined labeling schemes (e.g., assuming target trials are "high attention" by definition).

**2. Cross-paradigm generalization**
Oddball and naturalistic video paradigms were combined into a single unified training framework. LOSO cross-validation confirmed the framework generalizes across both subjects and experimental contexts without overfitting to any single paradigm.

**3. Safety-critical uncertainty quantification**
Model-specific uncertainty estimation was applied across all three meta-learners and statistically validated using the Mann-Whitney U test. All models reliably distinguish confident from uncertain predictions (all p < 10⁻⁶, Cohen's d > 1.7), making the framework suitable for high-stakes deployment such as driver distraction monitoring and pilot workload detection.

---

## 📌 Note on Data

The EEG dataset used in this work was collected through in-house experimentation at SNAP GmbH, Bochum, and is not publicly available. The code is shared for methodological transparency and reproducibility. The full pipeline can be adapted to any EEG dataset with comparable recording setups and paradigm structures.

---

## 📖 Thesis Document

The full written thesis is available here: [`Thesis_Final.pdf`](./Thesis_Final.pdf)

This work was completed in February 2026 as part of the MSc in Data Science at the University of Potsdam, in collaboration with SNAP GmbH, Bochum.

---

## 📬 Contact

**Ahmed Sameed**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat-square&logo=linkedin&logoColor=white)](https://linkedin.com/in/ahmed-sameed11/)

---
