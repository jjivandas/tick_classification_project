# Tick Species Classification Using BioCLIP 2: Project History & Results Catalog

> This document serves as a comprehensive reference for thesis preparation, oral defense, and grant applications. It covers the full methodology, project timeline, species exclusion rationale, results, and a catalog of all generated figures with file paths.

---

## 1. Terminology & Methodology

### Correct Scientific Terminology

| Common (Incorrect) Term | Correct Term | Explanation |
|--------------------------|--------------|-------------|
| "Fine-tuning the SVM" | **Few-shot transfer learning with an SVM classification head** | BioCLIP weights are frozen; only the SVM is trained on extracted embeddings |
| "Fine-tuning BioCLIP" | **Feature extraction with a frozen foundation model** | BioCLIP is used as-is, no gradient updates |
| "Accuracy" (unqualified) | **Macro accuracy (balanced accuracy)** | Mean per-class recall; robust to class imbalance |
| "Tuning SVM parameters" | **Hyperparameter optimization** | Searching over C, gamma, kernel — not updating model weights |
| "Training runs" | **Monte Carlo cross-validation** | 100 random stratified train/test splits per shot count |

### What BioCLIP Contributes

[BioCLIP 2](https://imageomics.github.io/bioclip-2/) is a vision foundation model pretrained on the TreeOfLife-200M dataset (200 million biological images spanning the tree of life). It provides:

1. **Domain-relevant feature representations:** Unlike general-purpose vision models (e.g., ImageNet-pretrained ResNets), BioCLIP has seen millions of biological specimen images during pretraining, encoding morphological features relevant to taxonomic classification.
2. **Zero-shot classification capability:** BioCLIP 2 can classify species it has never been explicitly trained on by matching image embeddings to text embeddings of species names. This provides a strong baseline (59.3% macro accuracy on our 5-species task) without any training data.
3. **High-quality embeddings for downstream classifiers:** The 512-dimensional embeddings BioCLIP 2 produces capture fine-grained morphological differences between tick species, enabling a simple SVM to achieve 92.4% macro accuracy with only 35 labeled specimens per species.

### How Macro Accuracy Is Computed

Macro accuracy (balanced accuracy) is the **unweighted mean of per-class recall**:

```
Macro Accuracy = (1/C) * sum(recall_i for i in 1..C)
```

where C = number of species classes. This metric:
- Gives equal weight to each species regardless of sample size
- Prevents the majority class (Dermacentor variabilis, N=318) from dominating the metric
- Is computed via `sklearn.metrics.balanced_accuracy_score`

---

## 2. Pipeline Description

### Stage 1: Data Cleaning

**Source:** Raw CSV metadata from collaborators (multiple snapshots: Aug 2025, Sep 2025, Oct 2025, Feb 2026, Mar 2026)

**Steps:**
1. Load CSV, drop rows with missing Sample ID or species label
2. Deduplicate by Sample ID (keep first occurrence)
3. Normalize species labels (fix typos, standardize capitalization)
4. Remove generic "Ixodes" specimens (genus-level only, no species ID) — 103 removed
5. Remove Ixodes cookei duplicates (Sample IDs ending in 'a') — 22 removed
6. Annotate pathogen results (T-prefix = not tested, blank = negative)
7. Filter species with fewer than 50 specimens — 9 species removed
8. Cross-reference with image directory, requiring both dorsal and ventral views per specimen

**Output:** `data/processed/final_data_cleaned.json` — 721 specimens, 1,442 images, 5 species

**Implementation:** `scripts/data_cleanup.py`, `notebooks/01_data_exploration.ipynb`

### Stage 2: BioCLIP Zero-Shot Inference

**Steps:**
1. Load tick images (dorsal + ventral views)
2. Extract 512-dimensional embeddings using BioCLIP 2 (`hf-hub:imageomics/bioclip-2` via `open_clip` backend)
3. Cache embeddings as `.npy` files in `data/processed/emb_cache/`
4. Run zero-shot classification using `bioclip.CustomLabelsClassifier` with species names as text prompts
5. Evaluate per-view (dorsal vs. ventral) and per-specimen accuracy

**Output:** `results/bioclip_zeroshot/runs/<timestamp>/bioclip_results.npz`

**Baseline result:** 59.3% macro accuracy (5 species, zero training data)

**Implementation:** `notebooks/02_bioclip_inference.ipynb`

### Stage 3: Few-Shot SVM Classification

**Steps:**
1. Load cached BioCLIP embeddings from `.npy` files
2. Average dorsal + ventral embeddings per specimen (producing one 512-dim vector per specimen)
3. For each shot count K in {1, 3, 5, 10, 25, 35}:
   - Repeat 100 times (Monte Carlo):
     - Stratified random sample of K specimens per species for training
     - Remaining specimens form the test set
     - Train `sklearn.pipeline.Pipeline([StandardScaler(), SVC(kernel='rbf', class_weight='balanced', probability=True)])`
     - Record per-specimen predictions, confidence scores, macro accuracy, overall accuracy
4. Aggregate metrics across 100 runs: mean, std, 95% CI

**Output:** Per-run directory under `results/svm/runs/<timestamp>/` containing:
- `predictions.csv` — all per-specimen predictions across all runs
- `analysis/shot_summary.csv` — aggregated metrics per shot count
- `analysis/per_species_accuracy.csv` — per-species performance
- `analysis/confusion_summary.csv` — misclassification patterns
- `plots/` — confusion matrices, learning curves, accuracy plots

**Implementation:** `notebooks/03_finetuning_svm_new.ipynb`

### Stage 4: Embedding Visualization

**Steps:**
1. Load cached BioCLIP embeddings (raw, pre-SVM — these visualizations show BioCLIP's representation space, not the SVM decision boundary)
2. Average dorsal + ventral per specimen
3. Run t-SNE (perplexity=30, random_state=42) and UMAP (n_neighbors=15, min_dist=0.1)
4. Generate interactive HTML viewer colored by species, sex, pathogen status

**Output:** `results/tsne/embedding_explorer.html`, `results/tsne/tsne_coords.csv`, `results/tsne/umap_coords.csv`

**Implementation:** `scripts/embedding_explorer.py`

---

## 3. Species Exclusion Rationale

### Why Species Were Excluded

The classification dataset was restricted to species with **50 or more unique specimens**. This threshold was chosen to account for the few-shot evaluation protocol:

- With K=35 training specimens per species, a minimum of ~50 total specimens is needed to have a meaningful test set (at least 15 test specimens per class)
- Species with very few specimens produce highly variable accuracy estimates across Monte Carlo runs, making results statistically unreliable
- The threshold balances dataset size against statistical power

### Filtering Steps

| Filter | Specimens Removed | Reason |
|--------|-------------------|--------|
| Generic "Ixodes" (genus-only label) | 103 | No species-level identification; unusable for species classification |
| Ixodes cookei duplicates (ID ending in 'a') | 22 | Duplicate images of existing specimens identified in data audit |
| Species with < 50 specimens | 91 (across 9 species) | Insufficient sample size for reliable few-shot evaluation |

### Species Removed (with specimen counts)

| Species | Unique Specimens | Reason for Exclusion |
|---------|-----------------|----------------------|
| Ixodes cookei | 30 | Below threshold; additionally had 22 duplicate images removed |
| Amblyomma maculatum | 26 | Below threshold |
| Ixodes dentatus | 25 | Below threshold |
| Ixodes kingi | 6 | Below threshold |
| Dermacentor andersoni | 1 | Below threshold |
| Ixodes banksi | 1 | Below threshold |
| Ixodes texanus | 1 | Below threshold |
| Ixodes brunneus | 1 | Below threshold |
| Ixodes muris | 1 | Below threshold |

### Final Classification Dataset (5 species)

| Species | Specimens | Images (dorsal + ventral) | % of Dataset |
|---------|-----------|--------------------------|--------------|
| Dermacentor variabilis | 318 | 636 | 44.1% |
| Ixodes scapularis | 174 | 348 | 24.1% |
| Amblyomma americanum | 124 | 248 | 17.2% |
| Haemaphysalis leporispalustris | 53 | 106 | 7.4% |
| Haemaphysalis longicornis | 52 | 104 | 7.2% |
| **Total** | **721** | **1,442** | **100%** |

---

## 4. Project Timeline & Key Milestones

### Phase 1: Initial Setup & Zero-Shot Baseline (July 2025)

- **Jul 18, 2025:** Built initial data pipeline (CSV loading, image cross-referencing, JSON dataset creation). Ran BioCLIP zero-shot inference on tick images with 9 species including generic "Ixodes" genus labels.
- **Jul 27, 2025:** Created confusion matrices for deeper per-class analysis. Identified patterns of misclassification between morphologically similar species.

**Data:** August 2025 CSV snapshot, 9 species

### Phase 2: Data Curation (September 2025)

- **Sep 1, 2025:** Removed 103 generic "Ixodes" specimens that lacked species-level identification. Created parallel cleaned dataset with 8 species (599 specimens). This was a critical data quality step — genus-only labels are meaningless for species-level classification.

**Data:** September 2025 CSV, 8 species (599 specimens)

### Phase 3: Few-Shot SVM Introduction (September-October 2025)

- **Sep 19, 2025:** First SVM few-shot experiment. Trained SVM on frozen BioCLIP embeddings with K=5 shots per species, 100 Monte Carlo runs. Established the transfer learning pipeline.
- **Oct 12, 2025:** Expanded to multi-shot sweep with K in {1, 3, 5, 10}. Implemented macro accuracy learning curves and species-level learning curves. Results showed satisfying monotonic improvement with increasing K.
- **Oct 30, 2025:** Major code refactor for reusability. Extended to K=25. Added full visualization suite (confusion matrices, per-species recall bars, error galleries).

**Data:** October 2025 CSV, 8 species

### Phase 4: Legacy Results (October-November 2025)

These results represent the 8-species, pre-cleanup dataset:

| Shots (K) | Macro Accuracy (mean) | Overall Accuracy (mean) | 95% CI (macro) |
|-----------|-----------------------|------------------------|----------------|
| 1  | 62.9% | 74.4% | +/- 1.8% |
| 3  | 74.0% | 81.9% | +/- 1.1% |
| 5  | 78.5% | 84.1% | +/- 0.8% |
| 10 | 83.9% | 86.4% | +/- 0.6% |
| 25 | 88.9% | 90.2% | +/- 0.5% |

### Phase 5: Infrastructure & New Data (February 2026)

- **Feb 6-10, 2026:** Rebuilt run-based architecture with timestamped directories. Migrated to centralized path configuration (`config/paths.json`). Created new production notebook (`03_finetuning_svm_new.ipynb`). Tested with K=20 shots.

**Data:** October 2025 CSV still in use, 8 species

### Phase 6: Data Expansion & Breakthrough (February-March 2026)

- **Feb 27, 2026:** Integrated new data from collaborators (February 2026 CSV). The expanded dataset, combined with a switch from 8 species to 5 species (to account for the lower number of specimens per species in the excluded classes), improved results significantly.
- **Mar 24-25, 2026:** Achieved **92.4% macro accuracy** at K=35 on the 5-species dataset. This represents a 3.5 percentage point improvement over the legacy 8-species best (88.9% at K=25).

### Phase 7: Data Cleanup & Visualization (March 2026)

- **Mar 27, 2026:** Implemented systematic data cleanup script (`scripts/data_cleanup.py`) with full audit trail. Built t-SNE and UMAP embedding visualizations from raw BioCLIP embeddings. Created interactive HTML embedding explorer.

**Data:** March 13, 2026 CSV (most recent), 5 species, 721 specimens

---

## 5. Current Best Results

### Run ID: 260324_164203 (March 24, 2026)

**Dataset:** 5 species, 721 specimens, 1,442 images (dorsal + ventral)
**Method:** Few-shot SVM on frozen BioCLIP 2 embeddings (`pybioclip` v1.3.3, model `hf-hub:imageomics/bioclip-2`), 100 Monte Carlo runs per K
**BioCLIP zero-shot baseline:** 59.3% macro accuracy

#### Macro Accuracy vs. Shots (K)

| Shots (K) | Macro Accuracy | Overall Accuracy | 95% CI (macro) | 95% CI (overall) |
|-----------|---------------|-----------------|----------------|-------------------|
| 1  | 57.9% | 65.4% | +/- 1.6% | +/- 2.6% |
| 3  | 72.4% | 77.6% | +/- 1.0% | +/- 1.1% |
| 5  | 78.7% | 82.2% | +/- 0.8% | +/- 0.6% |
| 10 | 85.6% | 87.3% | +/- 0.6% | +/- 0.4% |
| 25 | 90.8% | 90.7% | +/- 0.4% | +/- 0.3% |
| **35** | **92.4%** | **91.6%** | **+/- 0.4%** | **+/- 0.3%** |

#### Per-Species Accuracy at K=35

| Species | Macro Accuracy | Std Dev | 95% CI |
|---------|---------------|---------|--------|
| Ixodes scapularis | 96.5% | 2.2% | +/- 0.4% |
| Haemaphysalis leporispalustris | 96.3% | 4.6% | +/- 0.9% |
| Amblyomma americanum | 90.1% | 4.5% | +/- 0.9% |
| Haemaphysalis longicornis | 89.6% | 8.0% | +/- 1.6% |
| Dermacentor variabilis | 89.5% | 2.3% | +/- 0.4% |

#### Per-Species Accuracy at K=1 (one-shot)

| Species | Macro Accuracy | Std Dev |
|---------|---------------|---------|
| Dermacentor variabilis | 77.6% | 25.6% |
| Ixodes scapularis | 72.5% | 23.5% |
| Haemaphysalis leporispalustris | 57.7% | 20.4% |
| Haemaphysalis longicornis | 46.6% | 18.9% |
| Amblyomma americanum | 35.0% | 16.3% |

### Comparison: Legacy (8 species) vs. Current (5 species)

| Shots (K) | Legacy Macro Acc (8 spp) | Current Macro Acc (5 spp) | Difference |
|-----------|-------------------------|--------------------------|------------|
| 1  | 62.9% | 57.9% | -5.0% |
| 3  | 74.0% | 72.4% | -1.6% |
| 5  | 78.5% | 78.7% | +0.2% |
| 10 | 83.9% | 85.6% | +1.7% |
| 25 | 88.9% | 90.8%* | +1.9% |
| 35 | N/A   | 92.4% | — |

*Note: The K values differ between runs (legacy used K=25, current uses K=25 and K=35). The switch from 8 to 5 species was made to account for the lower number of specimens per species in the excluded classes, improving statistical reliability of the evaluation. Lower K results appear slightly lower for 5 species because the metric is more sensitive with fewer classes.*

---

## 6. Image Catalog

All paths are relative to the project root:
`/Users/jayjivandas/Research_Imageomics/BioClip/tick_classification_project/`

### 6.1 Data Distribution Plots

| # | Description | Path |
|---|-------------|------|
| 1 | Species distribution bar chart (5 classification species, N>=50) | `data/processed/species_distribution.png` |
| 2 | Metadata distribution (sex, life stage, condition, pathogen) | `data/processed/metadata_distribution.png` |

### 6.2 Legacy Results (8 species, Oct 2025 data)

| # | Description | Path |
|---|-------------|------|
| 3 | Macro accuracy vs. shots learning curve (K=1,3,5,10,25) with 95% CI band | `results/legacy_svm_bioclip/macro_accuracy_curve.png` |
| 4 | Species-level learning curves (all species, all K) | `results/legacy_svm_bioclip/species_learning_curves_page01.png` |
| 5 | Row-normalized confusion matrix, K=1 | `results/legacy_svm_bioclip/confusion_mean_rownorm_K01.png` |
| 6 | Row-normalized confusion matrix, K=3 | `results/legacy_svm_bioclip/confusion_mean_rownorm_K03.png` |
| 7 | Row-normalized confusion matrix, K=5 | `results/legacy_svm_bioclip/confusion_mean_rownorm_K05.png` |
| 8 | Row-normalized confusion matrix, K=10 | `results/legacy_svm_bioclip/confusion_mean_rownorm_K10.png` |
| 9 | Row-normalized confusion matrix, K=25 | `results/legacy_svm_bioclip/confusion_mean_rownorm_K25.png` |
| 10 | Per-species recall bars at K=10 | `results/legacy_svm_bioclip/per_species_bars_K10.png` |
| 11 | Per-species recall (horizontal) at K=10 | `results/legacy_svm_bioclip/simple_per_species_recall_K10.png` |
| 12 | Per-species recall (vertical) at K=10 | `results/legacy_svm_bioclip/simple_per_species_recall_vertical_K10.png` |
| 13 | True count vs. true positive count per species | `results/legacy_svm_bioclip/true_vs_tp_counts.png` |

### 6.3 Current Best Run (5 species, Mar 2026 data — Run 260324_164203)

| # | Description | Path |
|---|-------------|------|
| 14 | Macro accuracy vs. shots learning curve (K=1,3,5,10,25,35) | `results/svm/runs/260324_164203/plots/macro_accuracy_curve.png` |
| 15 | Species-level learning curves | `results/svm/runs/260324_164203/plots/species_learning_curves_page01.png` |
| 16 | Row-normalized confusion matrix, K=1 | `results/svm/runs/260324_164203/plots/confusion_mean_rownorm_K01.png` |
| 17 | Row-normalized confusion matrix, K=3 | `results/svm/runs/260324_164203/plots/confusion_mean_rownorm_K03.png` |
| 18 | Row-normalized confusion matrix, K=5 | `results/svm/runs/260324_164203/plots/confusion_mean_rownorm_K05.png` |
| 19 | Row-normalized confusion matrix, K=10 | `results/svm/runs/260324_164203/plots/confusion_mean_rownorm_K10.png` |
| 20 | Row-normalized confusion matrix, K=25 | `results/svm/runs/260324_164203/plots/confusion_mean_rownorm_K25.png` |
| 21 | Row-normalized confusion matrix, K=35 | `results/svm/runs/260324_164203/plots/confusion_mean_rownorm_K35.png` |
| 22 | Combined BioCLIP zero-shot vs. SVM K=35 confusion matrices | `results/svm/runs/260324_164203/plots/combined_confusion_bioclip_vs_svm35.png` |

### 6.4 Intermediate Milestone Runs

#### Run 260228_142210 (Feb 2026, transition to new data)

| # | Description | Path |
|---|-------------|------|
| 23 | Macro accuracy curve (K=1,3,5,10,20) | `results/svm/runs/260228_142210/plots/macro_accuracy_curve.png` |
| 24 | Species learning curves | `results/svm/runs/260228_142210/plots/species_learning_curves_page01.png` |
| 25 | Confusion matrix K=20 | `results/svm/runs/260228_142210/plots/confusion_mean_rownorm_K20.png` |

#### Run 260302_184234 (Mar 2, 2026, pre-breakthrough)

| # | Description | Path |
|---|-------------|------|
| 26 | Macro accuracy curve | `results/svm/runs/260302_184234/plots/macro_accuracy_curve.png` |
| 27 | Combined BioCLIP vs. SVM K=20 confusion matrices | `results/svm/runs/260302_184234/plots/combined_confusion_bioclip_vs_svm20.png` |
| 28 | Species learning curves | `results/svm/runs/260302_184234/plots/species_learning_curves_page01.png` |

### 6.5 Embedding Visualizations

| # | Description | Path |
|---|-------------|------|
| 29 | Interactive t-SNE + UMAP embedding explorer (open in browser) | `results/tsne/embedding_explorer.html` |

**Note:** The t-SNE and UMAP visualizations are computed on **raw BioCLIP embeddings** (before SVM training). They show how BioCLIP's pretrained feature space naturally clusters tick species.

### 6.6 Early Run Plots (Feb 2026, initial architecture)

| # | Description | Path |
|---|-------------|------|
| 30 | Macro accuracy curve (first new-architecture run) | `results/svm/runs/260206_045322/plots/macro_accuracy_curve.png` |
| 31 | Species learning curves | `results/svm/runs/260206_045322/plots/species_learning_curves_page01.png` |

---

## 7. Future Work

### Immediate Next Steps

1. **SVM Hyperparameter Optimization:** Systematic grid or random search over SVM hyperparameters (C, gamma, kernel type). Currently using default `SVC(kernel='rbf', class_weight='balanced')`. Expected improvement: 1-2 percentage points in macro accuracy.

2. **Cross-validation strategy:** Consider nested cross-validation for more robust hyperparameter selection.

### Longer-Term Directions

3. **BioCLIP Fine-Tuning (actual weight updates):** Unlike the current approach (frozen feature extraction), fine-tuning would update BioCLIP's weights on tick images. This could improve feature quality for morphologically similar species but requires careful regularization to avoid overfitting on small datasets.

4. **Dataset Expansion:** As more specimens are collected and identified, species currently below the 50-specimen threshold (e.g., Ixodes cookei at 30, Amblyomma maculatum at 26) may become viable for inclusion.

5. **Multi-view fusion strategies:** Currently averaging dorsal + ventral embeddings. Alternative approaches: attention-based fusion, concatenation, or view-specific classifiers.

---

## 8. Key Files Reference

| File | Purpose |
|------|---------|
| `notebooks/01_data_exploration.ipynb` | Data loading, cleaning, visualization |
| `notebooks/02_bioclip_inference.ipynb` | BioCLIP zero-shot evaluation + embedding extraction |
| `notebooks/03_finetuning_svm_new.ipynb` | Few-shot SVM training and evaluation |
| `scripts/data_cleanup.py` | Systematic data curation pipeline |
| `scripts/embedding_explorer.py` | t-SNE + UMAP visualization |
| `config/paths.json` | Centralized path configuration |
| `data/processed/final_data_cleaned.json` | Cleaned dataset (721 specimens, 5 species) |
| `data/processed/cleanup_summary.json` | Full audit trail of data cleaning steps |
| `data/processed/class_names_cleaned.json` | List of 5 classification species |
| `data/processed/emb_cache/*.npy` | Cached BioCLIP embeddings (2,748 files) |
