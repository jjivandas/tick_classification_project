"""
Embedding Explorer — t-SNE & UMAP Visualization
=================================================
Loads BioCLIP embeddings for the cleaned dataset, runs t-SNE and UMAP,
and generates an interactive HTML viewer colored by metadata.

Usage:
    python scripts/embedding_explorer.py

Output:
    results/tsne/embedding_explorer.html  — interactive viewer (open in browser)
    results/tsne/tsne_coords.csv          — t-SNE coordinates + metadata
    results/tsne/umap_coords.csv          — UMAP coordinates + metadata
"""

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder

# ── Setup paths ──────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
from utils.paths import load_paths

paths = load_paths()
EMB_CACHE = paths["emb_cache_dir"]
PROCESSED_DIR = paths["processed_dir"]
RESULTS_DIR = Path(paths["results_dir"]) / "tsne"


def _cache_fp(img_path: str) -> Path:
    """Same hashing as notebook 03 — sha256[:24]."""
    h = hashlib.sha256(img_path.encode("utf-8")).hexdigest()[:24]
    return EMB_CACHE / f"{h}.npy"


def load_data():
    """Load cleaned dataset and gather specimen-level embeddings + metadata."""
    data_path = PROCESSED_DIR / "final_data_cleaned.json"
    with open(data_path) as f:
        records = json.load(f)
    print(f"Loaded {len(records)} image records from {data_path.name}")

    # Group by specimen (average dorsal + ventral embeddings)
    specimens = defaultdict(lambda: {"embeddings": [], "meta": None})
    missing = 0
    for rec in records:
        sid = rec["sample_id"]
        cache_file = _cache_fp(rec["image_path"])
        if cache_file.exists():
            emb = np.load(cache_file)
            specimens[sid]["embeddings"].append(emb)
            if specimens[sid]["meta"] is None:
                specimens[sid]["meta"] = {
                    "sample_id": sid,
                    "species": rec["true_label"],
                    "sex": rec["sex"] or "Unknown",
                    "life_stage": rec["life_stage"] or "Unknown",
                    "attached": rec["attached"] or "Unknown",
                    "pathogen": rec["pathogen"] if rec["pathogen"] != "none" else "None",
                    "pathogen_result": rec["pathogen_result"],
                    "tick_condition": rec["tick_condition"] or "Unknown",
                }
        else:
            missing += 1

    if missing:
        print(f"Warning: {missing} image embeddings not found in cache")

    # Average embeddings per specimen
    X = []
    meta_rows = []
    for sid, data in specimens.items():
        if data["embeddings"] and data["meta"]:
            avg_emb = np.mean(data["embeddings"], axis=0)
            X.append(avg_emb)
            meta_rows.append(data["meta"])

    X = np.array(X)
    meta_df = pd.DataFrame(meta_rows)
    print(f"Built {len(X)} specimen embeddings (dim={X.shape[1]})")
    return X, meta_df


def run_tsne(X, perplexity=30, random_state=42):
    """Run t-SNE dimensionality reduction."""
    print(f"Running t-SNE (perplexity={perplexity})...")
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state,
                max_iter=1000, learning_rate="auto", init="pca")
    coords = tsne.fit_transform(X)
    print("  t-SNE complete.")
    return coords


def run_umap(X, n_neighbors=15, min_dist=0.1, random_state=42):
    """Run UMAP dimensionality reduction."""
    import umap

    print(f"Running UMAP (n_neighbors={n_neighbors}, min_dist={min_dist})...")
    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist,
                        random_state=random_state)
    coords = reducer.fit_transform(X)
    print("  UMAP complete.")
    return coords


def build_interactive_html(meta_df, tsne_coords, umap_coords, output_path):
    """Build a self-contained interactive HTML viewer with Plotly."""
    try:
        import plotly
    except ImportError:
        print("Installing plotly...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "plotly", "-q"])
        import plotly

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # Add coordinates to dataframe
    df = meta_df.copy()
    df["tsne_x"] = tsne_coords[:, 0]
    df["tsne_y"] = tsne_coords[:, 1]
    df["umap_x"] = umap_coords[:, 0]
    df["umap_y"] = umap_coords[:, 1]

    # Metadata fields to color by
    color_fields = ["species", "sex", "life_stage", "attached",
                    "pathogen_result", "tick_condition"]

    # Nice display names
    field_labels = {
        "species": "Species",
        "sex": "Sex",
        "life_stage": "Life Stage",
        "attached": "Attached",
        "pathogen_result": "Pathogen Result",
        "tick_condition": "Tick Condition",
    }

    # Color palettes for each field
    color_palettes = {
        "species": {
            "Amblyomma americanum": "#e41a1c",
            "Dermacentor variabilis": "#377eb8",
            "Haemaphysalis leporispalustris": "#4daf4a",
            "Haemaphysalis longicornis": "#984ea3",
            "Ixodes scapularis": "#ff7f00",
        },
        "sex": {
            "Female": "#e41a1c",
            "Male": "#377eb8",
            "Unknown": "#999999",
        },
        "life_stage": {
            "Adult": "#e41a1c",
            "Nymph": "#377eb8",
            "Larva": "#4daf4a",
            "Unknown": "#999999",
        },
        "attached": {
            "Yes": "#e41a1c",
            "No": "#377eb8",
            "Unknown": "#999999",
        },
        "pathogen_result": {
            "Positive": "#e41a1c",
            "negative": "#4daf4a",
            "not_tested": "#999999",
        },
        "tick_condition": {
            "Fed": "#e41a1c",
            "Unfed": "#377eb8",
            "Engorged": "#ff7f00",
            "Undetermined": "#999999",
            "Unknown": "#cccccc",
        },
    }

    # Build all traces (hidden by default, toggled by buttons)
    fig = go.Figure()

    trace_indices = {}  # (method, field) -> list of trace indices
    trace_idx = 0

    for method in ["tsne", "umap"]:
        x_col = f"{method}_x"
        y_col = f"{method}_y"

        for field in color_fields:
            categories = sorted(df[field].unique())
            palette = color_palettes.get(field, {})
            indices = []

            for cat in categories:
                mask = df[field] == cat
                subset = df[mask]
                color = palette.get(cat, "#999999")

                # Build hover text
                hover_text = []
                for _, row in subset.iterrows():
                    text = (
                        f"<b>{row['species']}</b><br>"
                        f"Sample: {row['sample_id']}<br>"
                        f"Sex: {row['sex']}<br>"
                        f"Life Stage: {row['life_stage']}<br>"
                        f"Condition: {row['tick_condition']}<br>"
                        f"Pathogen: {row['pathogen_result']}<br>"
                        f"Attached: {row['attached']}"
                    )
                    hover_text.append(text)

                fig.add_trace(go.Scatter(
                    x=subset[x_col],
                    y=subset[y_col],
                    mode="markers",
                    name=str(cat),
                    text=hover_text,
                    hoverinfo="text",
                    marker=dict(
                        size=8,
                        color=color,
                        opacity=0.75,
                        line=dict(width=0.5, color="white"),
                    ),
                    visible=False,
                ))
                indices.append(trace_idx)
                trace_idx += 1

            trace_indices[(method, field)] = indices

    # Default view: t-SNE colored by species
    for idx in trace_indices[("tsne", "species")]:
        fig.data[idx].visible = True

    # Build dropdown buttons
    method_buttons = []
    for method in ["tsne", "umap"]:
        method_label = "t-SNE" if method == "tsne" else "UMAP"
        for field in color_fields:
            visibility = [False] * trace_idx
            for idx in trace_indices[(method, field)]:
                visibility[idx] = True

            method_buttons.append(dict(
                label=f"{method_label} — {field_labels[field]}",
                method="update",
                args=[
                    {"visible": visibility},
                    {"title": f"Tick Embeddings: {method_label} colored by {field_labels[field]}",
                     "xaxis.title": f"{method_label} 1",
                     "yaxis.title": f"{method_label} 2"},
                ],
            ))

    fig.update_layout(
        title="Tick Embeddings: t-SNE colored by Species",
        xaxis_title="t-SNE 1",
        yaxis_title="t-SNE 2",
        template="plotly_white",
        width=1100,
        height=750,
        legend=dict(
            title="Legend",
            font=dict(size=13),
            borderwidth=1,
            itemsizing="constant",
        ),
        updatemenus=[
            dict(
                type="dropdown",
                direction="down",
                x=0.02,
                y=0.98,
                xanchor="left",
                yanchor="top",
                buttons=method_buttons,
                font=dict(size=13),
                bgcolor="white",
                borderwidth=1,
                pad=dict(t=10, b=10),
            ),
        ],
        annotations=[
            dict(
                text="View:",
                x=0.02, y=1.02,
                xref="paper", yref="paper",
                xanchor="left",
                showarrow=False,
                font=dict(size=14, color="black"),
            )
        ],
        margin=dict(t=80),
    )

    # Write HTML
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path), include_plotlyjs=True)
    print(f"Saved interactive viewer: {output_path}")


def main():
    print("=" * 60)
    print("  EMBEDDING EXPLORER")
    print("=" * 60)

    # Load data
    X, meta_df = load_data()

    # Run both methods
    tsne_coords = run_tsne(X)
    umap_coords = run_umap(X)

    # Save coordinate CSVs
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    tsne_df = meta_df.copy()
    tsne_df["dim1"] = tsne_coords[:, 0]
    tsne_df["dim2"] = tsne_coords[:, 1]
    tsne_df.to_csv(RESULTS_DIR / "tsne_coords.csv", index=False)
    print(f"Saved {RESULTS_DIR / 'tsne_coords.csv'}")

    umap_df = meta_df.copy()
    umap_df["dim1"] = umap_coords[:, 0]
    umap_df["dim2"] = umap_coords[:, 1]
    umap_df.to_csv(RESULTS_DIR / "umap_coords.csv", index=False)
    print(f"Saved {RESULTS_DIR / 'umap_coords.csv'}")

    # Build interactive HTML
    html_path = RESULTS_DIR / "embedding_explorer.html"
    build_interactive_html(meta_df, tsne_coords, umap_coords, html_path)

    print("\n" + "=" * 60)
    print("  DONE")
    print("=" * 60)
    print(f"Open {html_path} in a browser to explore.")
    print("Use the dropdown (top-left) to switch between t-SNE/UMAP")
    print("and color by species, sex, life stage, condition, etc.")
    print("Hover over any point to see specimen details.")
    print("=" * 60)


if __name__ == "__main__":
    main()
