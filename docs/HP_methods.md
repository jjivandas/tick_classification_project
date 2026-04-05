# SVM Hyperparameter Tuning Methods

## Context

We train SVM classifiers on frozen BioCLIP embeddings (512-dim) for tick species classification. The pipeline uses a `StandardScaler → SVC` pipeline evaluated via Monte Carlo k-shot experiments (100 random train/test splits per shot level K=1,3,5,10,25,35).

Default SVM parameters: `C=1.0, gamma=scale, kernel=rbf`.

---

## Method 1: GridSearchCV with Frequency Voting (v1, deprecated)

**Run directory:** `results/svm_hp/runs/260405_003930/`

### Approach

1. **Search phase (20 MC runs at K=35 only):**
   - For each run, split data into 35 train specimens/species + rest as test
   - Run `GridSearchCV` with 5-fold stratified CV on the training set
   - Search grid: C=[0.01, 0.1, 1, 10, 100], gamma=[scale, auto, 0.001, 0.01, 0.1], kernel=[rbf, linear]
   - Record which (C, gamma, kernel) combo achieves best CV score

2. **Selection:** Pick the parameter combination that was selected most frequently across the 20 search runs (frequency voting)

3. **Evaluation:** Re-run full 100 MC sweep at all K values using the single selected parameter set

### Results

- Most frequently selected params were very close to defaults (C=1, gamma=scale, rbf)
- Improvements were marginal: +0.2% to +3.9% macro accuracy depending on K

### Limitations

- **Frequency voting is weak:** Ignores magnitude of improvement — a combo that wins by 0.1% counts the same as one that wins by 5%
- **Single K search:** Only searches at K=35 where there's plenty of training data. Best params at K=35 may not be best at K=3
- **One fixed config for all K:** Forces a single parameter set across all shot levels, even though optimal params likely vary by K
- **Overhead:** GridSearchCV × 20 MC runs is expensive for what amounts to confirming the defaults are good

---

## Method 2: Direct Config Comparison (v2, current)

**Run directory:** `results/svm_hp/`

### Approach

1. **Define candidate configurations manually:**
   Each config is a (C, gamma, kernel) tuple chosen to cover different regions of the hyperparameter space:

   | Config | C | gamma | kernel | Rationale |
   |--------|---|-------|--------|-----------|
   | A_default | 1.0 | scale | rbf | Baseline — current default params |
   | B_smooth | 10 | 0.001 | rbf | Higher C, very small gamma — broader, smoother decision regions |
   | C_tight | 100 | scale | rbf | Much higher C — less regularization, tighter fit to training data |
   | D_linear | 10 | scale | linear | Linear kernel — tests whether nonlinear boundaries are even needed |
   | E_regularized | 0.1 | auto | rbf | Lower C — more regularization, auto gamma = 1/n_features |

2. **Full sweep for each config:**
   - Run all 100 MC splits at every K value (1, 3, 5, 10, 25, 35)
   - Same seeds across configs so splits are identical — direct apples-to-apples comparison
   - Record per-run, per-species predictions for each config

3. **Comparison:**
   - Macro accuracy curves for all configs overlaid on one plot
   - Per-config combined (per-species + macro) plots
   - Confusion matrices at all K values for each config
   - Side-by-side confusion matrix: best config vs default at max K
   - CSV with all metrics for easy analysis

### Why this approach

- **Transparent:** You see exactly what each config does at every K, no black-box selection
- **Fair comparison:** Same random splits across all configs (same seeds)
- **Per-K visibility:** A config that's best at K=35 may be worst at K=3 — this approach reveals that
- **Simple:** No nested CV overhead, just run and compare
- **Extensible:** Easy to add more configs later without re-running existing ones

### Output structure

```
results/svm_hp/
├── A_default/
│   ├── predictions.csv
│   ├── analysis/
│   │   ├── shot_summary.csv
│   │   └── per_species_accuracy.csv
│   └── plots/
│       ├── macro_accuracy.png
│       ├── per_species_accuracy.png
│       ├── combined_species_macro.png
│       └── confusion_mean_rownorm_K{01..35}.png
├── B_smooth/
│   └── (same structure)
├── C_tight/
│   └── (same structure)
├── D_linear/
│   └── (same structure)
├── E_regularized/
│   └── (same structure)
└── comparison/
    ├── analysis/
    │   └── all_configs_summary.csv
    └── plots/
        ├── all_configs_macro_overlay.png
        ├── best_vs_default_macro.png
        └── best_vs_default_confusion_K35.png
```
