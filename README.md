# Reevaluating Deep Learning Baselines in EEG-Based Emotion Recognition

Code, results and figures for the revised manuscript:

> **Reevaluating Deep Learning Baselines in EEG-Based Emotion Recognition: Quantifying Window-Level Leakage under Trial-Level Cross-Validation**
> (revision v2, Cognitive Neurodynamics)

## Contents

```
code/     analysis, figure and manuscript-build scripts
results/  all numeric outputs (JSON/CSV) produced by the analysis script
figures/  all result figures and MATLAB topographic maps (PNG, 300 dpi)
```

| File | Purpose |
|---|---|
| `code/revision_v2_analyses.py` | **Single seeded script** that recomputes every number reported in the paper from the cached DE features (RF, static CNN, CNN-LSTM, leakage controls, overlap scan, band ablation, frontal asymmetry, LOSO, permutation test) |
| `code/extract_deap_advanced_features.m` | Original MATLAB feature extraction (DE features from the DEAP preprocessed release) |
| `code/train_hybrid_emotion_model.py` | Original CNN-LSTM training script |
| `code/train_rf_trial_level.py` | Original Random Forest training script |
| `code/train_window_level_leakage.py` | Original window-level (leaky) training script |
| `code/regenerate_topoplots_v2.m` | MATLAB script regenerating the topographic maps (Figures 9, 11–16) |
| `code/make_figures_v2.py` | Python script regenerating Figures 5–8 and 10 |
| `code/build_manuscript_v2.py` + `docx_redline.py` | Build the revision DOCX with all numbers read from `results/*.json` |
| `results/results_rf.json`, `results_cnn.json` | Full per-model, per-protocol results |
| `results/per_subject_*.json/csv` | Per-subject accuracies, balanced accuracies, majority-class baselines and label distributions |
| `results/numbers_v2.json` | Consolidated numbers used verbatim in the manuscript and the response letter |

## Data

The features used for every analysis are derived from the **DEAP dataset**
(Koelstra et al., 2012), which is distributed under a licence that does not
allow redistribution. Users must request DEAP directly from the authors. The
feature cache (`de_features_all.mat`, 39,680 windows × 56 differential-entropy
features, 14 channels × 4 bands, band-major ordering) can be regenerated with
`code/extract_deap_advanced_features.m` and is also provided as journal
supplementary material.

## Reproducing the results

```bash
python code/revision_v2_analyses.py rf     # RF + asymmetry + LOSO + ablation
python code/revision_v2_analyses.py cnn    # CNN-LSTM (3 seeds) + static CNN + grid-order control
python code/revision_v2_analyses.py perm   # label-permutation test
python code/make_figures_v2.py             # Figures 5-8, 10
matlab -batch regenerate_topoplots_v2      # Figures 9, 11-16
```

All analyses use fixed seeds (RF 42; CNN 42/43/44) and were run with
Python 3.12, PyTorch 2.x, scikit-learn, SciPy; topographic maps with MATLAB R2025a.

## Contact

Bin Li (corresponding author) — benliinz@gmail.com
