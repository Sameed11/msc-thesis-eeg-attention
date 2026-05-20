# Robust Attention State Prediction from EEG Signals Using Meta-Learning and Uncertainty-Aware Deep Learning

## Overview

This repository contains the full implementation and written thesis for my MSc research on EEG-based attention state prediction. The work proposes a two-stage, uncertainty-aware meta-learning framework that addresses three fundamental limitations in current brain-computer interface (BCI) research: single-paradigm training, predefined attention labeling, and the absence of predictive uncertainty quantification.

EEG data was collected from nine subjects across four experimental paradigms (oddball task, car crash video, animal surprise video, and surprise events). Attention states were first discovered in a data-driven manner through unsupervised clustering, then classified using a meta-learning pipeline that combines spectral predictions with contextual information. The framework achieved 97.4% classification accuracy under Leave-One-Subject-Out (LOSO) cross-validation, with statistically significant uncertainty quantification between correct and incorrect predictions (p < 10^-6).

---

## Repository Contents

| File | Description |
|---|---|
| `Thesis_code.ipynb` | Full implementation: preprocessing, feature extraction, clustering, spectral specialist, meta-learners, and uncertainty quantification |
| `Thesis_Final.pdf` | Complete written thesis document |

---

## Framework Architecture

The predictive framework consists of three sequential stages:

**Stage 1: Attention State Discovery**
Raw EEG signals are preprocessed through a structured pipeline (band-pass filtering, epoching, normalization, quality assurance). Power spectral density is estimated using Welch's method across five frequency bands: delta (0.5-4 Hz), theta (4-7 Hz), alpha (8-12 Hz), beta (13-30 Hz), and gamma (30-40 Hz). A seven-dimensional target vector of relative band powers and attention-relevant ratios (alpha/beta, theta/beta) is constructed and used as input to k-means clustering, which discovers three neurophysiologically distinct attention states without relying on predefined labels.

**Stage 2: Meta-Learning Classification**
A LightGBM spectral specialist first learns to predict the seven-dimensional neurophysiological feature vector from a 59-dimensional input comprising global band powers, relative band powers, ROI band powers across seven cortical regions, hemispheric asymmetry, and attention-relevant ratios. The specialist's predictions are then combined with nine one-hot encoded contextual features (experimental paradigm and condition) to form a 16-dimensional meta-feature vector. Three meta-learners classify each epoch into one of three attention states: Logistic Regression, XGBoost, and a Transformer.

**Stage 3: Uncertainty Quantification**
Model-specific uncertainty is quantified for each meta-learner: calibrated predictive entropy (Platt scaling) for Logistic Regression, ensemble-based entropy for XGBoost, and Monte Carlo Dropout across 50 forward passes for the Transformer. The Mann-Whitney U test verifies that incorrect predictions exhibit significantly higher uncertainty than correct predictions.

---

## Discovered Attention States

Three neurophysiologically distinct attention states were identified through unsupervised k-means clustering (CHI = 122.89, Silhouette = 0.266):

| State | Alpha/Beta Ratio | Profile | % of Data |
|---|---|---|---|
| Baseline Attention | 1.91 | Theta-dominant, routine processing | 30.6% |
| Low Arousal | 1.06 | Delta-dominant, drowsiness | 28.9% |
| High Attention | 0.52 | Beta-dominant, focused engagement | 40.4% |

---

## Key Results

**Spectral Specialist (LightGBM)**
- Mean R² > 0.94 for five EEG band powers (delta, theta, alpha, beta, gamma)
- Mean MAE < 0.01 for band power predictions
- Alpha-beta ratio: R² = 0.750, MAE = 0.146
- 8 out of 9 subjects achieved R² > 0.83 under LOSO cross-validation

**Meta-Learner Classification (LOSO Cross-Validation)**

| Model | Accuracy | Balanced Accuracy | F1-Macro |
|---|---|---|---|
| Logistic Regression | 97.4% | 97.4% | 0.972 |
| XGBoost | 96.4% | 96.4% | 0.963 |
| Transformer | 95.7% | 95.8% | 0.956 |

**Uncertainty Quantification**

All three models reliably separated correct from incorrect predictions:

| Model | Correct Uncertainty | Incorrect Uncertainty | p-value | Cohen's d |
|---|---|---|---|---|
| Logistic Regression | 0.336 | 0.777 | 1.27 x 10^-6 | 2.096 |
| XGBoost | 0.124 | 0.512 | 7.43 x 10^-9 | 1.842 |
| Transformer | 0.046 | 0.372 | 6.66 x 10^-8 | 1.732 |

Incorrect predictions exhibited 2-4x higher uncertainty than correct predictions across all models, indicating strong calibration for safety-critical deployment.

---

## Technical Stack

- **Language:** Python
- **EEG Processing:** Welch's PSD estimation, Butterworth band-pass filtering, z-score normalization, epoch segmentation
- **Machine Learning:** LightGBM, XGBoost, scikit-learn (Logistic Regression, MultiOutputRegressor)
- **Deep Learning:** PyTorch (custom Transformer encoder with multi-head self-attention)
- **Uncertainty Quantification:** Platt scaling, ensemble entropy, Monte Carlo Dropout
- **Statistical Testing:** Mann-Whitney U test, independent t-test, one-way ANOVA, Shapiro-Wilk, Levene's test, Cohen's d
- **Clustering:** k-means with Calinski-Harabasz and Silhouette validation
- **Validation:** Leave-One-Subject-Out (LOSO) cross-validation
- **Visualization:** Matplotlib, Seaborn

---

## Experimental Setup

- **Participants:** 9 subjects (ages 20-40)
- **EEG Hardware:** OpenBCI-based system, 16 active electrodes, International 10-20 placement, 125 Hz sampling rate, reference electrode Fz
- **Paradigms:** Oddball task (P300 evocation), car crash video, animal surprise video, surprise events video
- **Epochs:** 418 event-locked epochs across all paradigms and subjects
- **Cross-Validation:** LOSO (9 folds, one subject held out per fold)

---

## Research Contributions

1. **Data-driven attention state labeling:** Instead of relying on predefined attention labels, unsupervised clustering on ground-truth spectral features identified three neurophysiologically meaningful attention states, avoiding circular reasoning in the labeling scheme.

2. **Cross-paradigm generalization:** Combined oddball and naturalistic video paradigms into a unified training framework, validated using LOSO cross-validation across all paradigm types.

3. **Integrated uncertainty quantification:** Applied model-specific uncertainty methods to all three meta-learners and demonstrated statistically significant separation between confident and uncertain predictions, making the framework suitable for safety-critical applications such as driver monitoring and human-machine interaction.

---

## Citation

If you find this work useful, please cite:

```
Ahmed Sameed. Robust Attention State Prediction from EEG Signals Using
Meta-Learning and Uncertainty-Aware Deep Learning. MSc Thesis,
University of Potsdam & SNAP GmbH, 2026.
```

---

## Note on Data

The EEG dataset used in this work was collected through in-house experimentation at SNAP GmbH, Bochum, and is not publicly available. The code is shared for reproducibility and methodological transparency. The full pipeline can be adapted to any EEG dataset with comparable preprocessing and feature extraction steps.

---

## Contact

**Ahmed Sameed**
LinkedIn: [linkedin.com/in/ahmed-sameed11](https://linkedin.com/in/ahmed-sameed11/)
