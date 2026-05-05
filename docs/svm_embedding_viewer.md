# SVM Embedding Viewer

Interactive 2-D viewer of the tick dataset under the **fine-tuned SVM**.
For each shot count `k`, every specimen gets a representation that reflects
how the classifier sees it; t-SNE and UMAP project that representation to 2-D.

## Why this exists

The original `scripts/embedding_explorer.py` projects raw BioCLIP embeddings.
That answers "what does BioCLIP think these ticks look like?" but not
"how does our fine-tuned SVM reorganize that space as we add more support shots?"

This viewer answers the second question: shot toggle on top, switch between
t-SNE and UMAP, pick any metadata field as primary or secondary label.
Default colors lock to species, matching the macro-accuracy plot.

## Pipeline

`scripts/svm_embedding_explorer.py` runs in five steps:

1. **Specimen index.** Load `data/processed/final_data_cleaned.json`, keep
   specimens with both dorsal and ventral views, drop species with fewer than
   50 specimens. Result: 5 species, 721 specimens.
2. **Specimen embeddings.** Read the BioCLIP cache from
   `data/processed/emb_cache/`, average dorsal + ventral per specimen.
   No re-embedding.
3. **Best HP auto-pick.** Read `shot_summary.csv` for each config in
   `results/svm_hp/runs/260405_225314/{G_finer,H_wider,I_tiny_gamma,J_logreg}/`,
   pick the one with the highest 35-shot mean macro accuracy. (G_finer wins:
   `C=10, gamma=0.0005, kernel=rbf`, 93.34% at k=35.)
4. **Per-shot proba aggregation.** For each `k ∈ {1, 10, 25, 35}`, run
   100 Monte-Carlo splits using the *same seed function as notebook 04*:
   ```
   seed_for(k, run_id) = int(sha256(f"42_{k}_{run_id}").hexdigest()[:8], 16) % 2**31
   ```
   so the train/test splits are byte-identical to those in
   `results/svm_hp/runs/260405_225314/<best>/predictions.csv`. For each
   split: train `Pipeline(StandardScaler, SVC(probability=True))` on the
   support set, then `clf.predict_proba(X_all)` on every specimen — not just
   the held-out ones — so every specimen has a probability vector at every k.
   Average across the 100 runs.
5. **2-D projections.** For each `k ∈ {0, 1, 10, 25, 35}` apply t-SNE
   (`perplexity=30, init="pca"`) and UMAP (`n_neighbors=15, min_dist=0.1`)
   with `random_state=42`. `k=0` projects raw 768-D BioCLIP; `k>0` projects
   the 5-D mean proba.

## Output layout

```
results/tsne/runs/<YYMMDD_HHMMSS>/
├── coords.csv                 # long-form: sample_id, metadata, shots, method, x, y
├── proba_mean_k1.npy          # (721, 5) mean proba at k=1
├── proba_mean_k10.npy
├── proba_mean_k25.npy
├── proba_mean_k35.npy
├── class_order_k*.json        # which class index maps to which species
├── manifest.json              # HP, seeds, sanity checks, timestamps
└── viewer.html                # self-contained interactive viewer
results/tsne/runs/latest_run.txt
```

## Reproducibility checks

The script prints two checks per shot, also recorded in `manifest.json`:

| Check | What it confirms |
|---|---|
| Row-sum max deviation from 1.0 | `predict_proba` rows really sum to 1 (sanity) |
| Agreement vs `predictions.csv` | The new run's argmax-on-test agrees with the stored `species_pred` from `260405_225314/<best>/predictions.csv` for the same `(shots, run_id, sample_id)` triple |

Notes on the agreement number:

- The existing CSV stored `clf.predict()` (the SVM decision function's
  argmax). The new pipeline's argmax of `predict_proba` is what gets
  visualized. These two are **not** guaranteed to agree when `probability=True`
  because Platt scaling is fit by internal cross-validation
  ([sklearn note](https://scikit-learn.org/stable/modules/svm.html#scores-probabilities)).
- At `k ≥ 10`, agreement is ≥97% — confirming the seed scheme and HP
  exactly match notebook 04. The 1–3% gap is the expected predict-vs-proba-argmax
  divergence.
- At `k=1`, agreement collapses (≈0%) because the Platt CV is fitted on
  5 training samples (one per class) and is highly unstable. The proba-based
  embedding at `k=1` is still meaningful as "what the calibrated SVM thinks
  this point is" — it just no longer matches the raw decision function.

The end-of-run "argmax at k=35 matches true species" is the analogue of the
macro-accuracy result and should be ~93–96%.

## Color palette (locked)

When the primary label is `species`, the viewer uses these exact hex values
regardless of selector ordering, matching notebook 03's macro-accuracy plot:

| Species | Hex | Color |
|---|---|---|
| Amblyomma americanum | `#1f77b4` | blue |
| Dermacentor variabilis | `#ff7f0e` | orange |
| Haemaphysalis leporispalustris | `#2ca02c` | green |
| Haemaphysalis longicornis | `#d62728` | red |
| Ixodes scapularis | `#9467bd` | purple |

For any other primary label (sex, life_stage, etc.) the viewer falls back
to a default categorical palette. The secondary label always modulates
shade within each primary value via an HSL transform (saturation +
lightness gradient over the secondary's sort index).

## Viewer controls

- **Shots:** `0`, `1`, `10`, `25`, `35`. `0` is raw BioCLIP, the rest are
  100-MC mean SVM proba.
- **Method:** `t-SNE` / `UMAP`.
- **Primary label:** any specimen-level metadata field — drives the hue.
- **Secondary label:** any specimen-level metadata field — drives the shade
  within each primary group.
- Hover shows: sample_id, species, sex, life_stage, attached, pathogen,
  pathogen_result, tick_condition, plus the current primary and secondary
  values.

`view` (dorsal/ventral) is intentionally not in the selector — points are
specimen-level (dorsal+ventral averaged), matching the SVM training
pipeline. A separate per-view study would re-run the pipeline at the image
level.

## How to run

From the repo root:

```bash
.venv/bin/python scripts/svm_embedding_explorer.py
```

Runs on the OSC login node — no GPU node, no scheduler. SVC + sklearn t-SNE
+ umap-learn are CPU-only and BioCLIP embeddings are cached, so the workload
is ~5 minutes on a small CPU. To open the viewer, copy the produced
`viewer.html` to a machine with a browser, or open it in a Jupyter file
view (single self-contained HTML, ~few MB).

## How to read the plot

- **k=0** (raw BioCLIP) shows the unsupervised structure — clusters reflect
  general visual similarity, but with messy boundaries between species that
  look alike.
- **k=1, 10, 25, 35** show the same specimens reorganized by the SVM's
  averaged probability. As k grows, expect:
  - clusters tighten and pull toward species centroids,
  - between-cluster distance grows,
  - hard cases (e.g. *Haemaphysalis longicornis* vs *H. leporispalustris*)
    that overlap at low k separate at higher k.
- Toggling primary to `sex` or `life_stage` over the same projection
  surfaces whether residual confusions correlate with non-species
  attributes — useful for diagnosing systematic errors.

## Critical files

- `scripts/svm_embedding_explorer.py` — pipeline + HTML writer.
- `scripts/embedding_explorer.py` — original raw-BioCLIP viewer (untouched).
- `scripts/reference_script/bioclip_tsne_cli.py` — pattern for the
  primary/secondary HSL coloring and the controls layout.
- `notebooks/03_finetuning_svm_new.ipynb`,
  `notebooks/04_hyperparameter_search.ipynb` — source of truth for the
  SVM training pipeline (StandardScaler + SVC, balanced weights,
  `probability=True`) and the seed function.
- `results/svm_hp/runs/260405_225314/` — HP sweep used to pick the winning
  config.
- `data/processed/final_data_cleaned.json`,
  `data/processed/emb_cache/` — inputs, both pre-existing.
