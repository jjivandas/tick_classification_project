#!/usr/bin/env python3
"""Create BioCLIP embeddings and interactive t-SNE visualizations for labeled images."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from glob import glob
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


SYSTEM_COLUMNS = {"image_path", "image_name", "image_id", "_csv_match_status", "_embedding_index"}
KNOWN_LABEL_FIXES = {
    "Dermacentor variablis": "Dermacentor variabilis",
}


def _require_bioclip():
    try:
        from bioclip import TreeOfLifeClassifier  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "Could not import BioCLIP. Install `pybioclip` in your environment first."
        ) from exc
    return TreeOfLifeClassifier


def _extract_image_id(
    image_path: Path, id_regex: str | None, strip_trailing_segment: bool
) -> str:
    stem = image_path.stem
    if strip_trailing_segment:
        stem = re.sub(r"-\d{1,2}$", "", stem)
    if not id_regex:
        return stem

    match = re.search(id_regex, stem)
    if not match:
        return stem
    if "id" in match.groupdict():
        return match.group("id")
    if match.groups():
        return match.group(1)
    return match.group(0)


def _normalize_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Iterate by position so duplicate column names are handled safely.
    for idx, _ in enumerate(df.columns):
        series = df.iloc[:, idx]
        if pd.api.types.is_string_dtype(series) or series.dtype == object:
            normalized = series.fillna("").astype(str).str.strip()
            df.iloc[:, idx] = normalized.replace(KNOWN_LABEL_FIXES)
    return df


def _resolve_images(image_glob: str) -> list[Path]:
    paths = sorted(Path(p) for p in glob(image_glob, recursive=True))
    image_paths = [p for p in paths if p.is_file()]
    if not image_paths:
        raise SystemExit(f"No images matched pattern: {image_glob}")
    return image_paths


def _select_default_label_columns(df: pd.DataFrame) -> list[str]:
    label_cols = []
    for col in df.columns:
        if col in SYSTEM_COLUMNS:
            continue
        if col in {"tsne_x", "tsne_y", "pca_x", "pca_y", "umap_x", "umap_y"}:
            continue
        if str(col).startswith("Unnamed:"):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
        label_cols.append(col)
    if not label_cols:
        for col in df.columns:
            if col not in SYSTEM_COLUMNS:
                if col in {"tsne_x", "tsne_y", "pca_x", "pca_y", "umap_x", "umap_y"}:
                    continue
                if str(col).startswith("Unnamed:"):
                    continue
                label_cols.append(col)
    return label_cols


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen = set()
    out = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _build_image_table(
    image_paths: Iterable[Path], id_regex: str | None, strip_trailing_segment: bool
) -> pd.DataFrame:
    rows = []
    iterable = image_paths
    if tqdm is not None:
        iterable = tqdm(image_paths, desc="Indexing images", unit="img")
    for p in iterable:
        rows.append(
            {
                "image_path": str(p.resolve()),
                "image_name": p.name,
                "image_id": _extract_image_id(p, id_regex, strip_trailing_segment),
            }
        )
    return pd.DataFrame(rows)


def _prepare_labels(labels_csv: Path, id_column: str) -> tuple[pd.DataFrame, int]:
    labels_df = pd.read_csv(labels_csv, dtype=str)
    labels_df = _normalize_text_columns(labels_df)
    if id_column not in labels_df.columns:
        raise SystemExit(f"Column `{id_column}` was not found in labels CSV: {labels_csv}")

    dup_mask = labels_df[id_column].duplicated(keep="first")
    dup_count = int(dup_mask.sum())
    if dup_count:
        labels_df = labels_df[~dup_mask].copy()
    return labels_df, dup_count


def run_embed(args: argparse.Namespace) -> None:
    TreeOfLifeClassifier = _require_bioclip()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = _resolve_images(args.image_glob)
    image_df = _build_image_table(
        image_paths, args.image_id_regex, args.strip_trailing_segment
    )

    labels_df, dup_count = _prepare_labels(Path(args.labels_csv), args.id_column)
    merged_df = image_df.merge(labels_df, how="left", left_on="image_id", right_on=args.id_column)
    merged_df["_csv_match_status"] = np.where(merged_df[args.id_column].isna(), "unmatched", "matched")
    merged_df["_embedding_index"] = np.arange(len(merged_df), dtype=np.int64)

    unmatched_count = int((merged_df["_csv_match_status"] == "unmatched").sum())

    print(
        f"Embedding {len(merged_df)} images with device={args.device} "
        f"(matched={len(merged_df) - unmatched_count}, unmatched={unmatched_count}).",
        flush=True,
    )
    if dup_count:
        print(
            f"Warning: dropped {dup_count} duplicate CSV rows based on `{args.id_column}` (kept first).",
            flush=True,
        )

    classifier = TreeOfLifeClassifier(
        device=args.device,
        model_str=args.model,
        pretrained_str=args.pretrained,
    )

    all_embeddings: list[np.ndarray] = []
    batch_size = max(1, args.batch_size)
    batch_starts = range(0, len(merged_df), batch_size)
    if tqdm is not None:
        batch_starts = tqdm(batch_starts, desc="Embedding batches", unit="batch")
        image_bar = tqdm(total=len(merged_df), desc="Embedded images", unit="img")
    else:
        image_bar = None

    for start in batch_starts:
        stop = min(start + batch_size, len(merged_df))
        batch_paths = merged_df.iloc[start:stop]["image_path"].tolist()
        images = [classifier.ensure_rgb_image(p) for p in batch_paths]
        feats = classifier.create_image_features(images, normalize=args.normalize)
        all_embeddings.append(feats.detach().cpu().numpy().astype(np.float32))
        if image_bar is not None:
            image_bar.update(len(batch_paths))
        else:
            print(f"Embedded {stop}/{len(merged_df)} images", flush=True)

    if image_bar is not None:
        image_bar.close()

    embeddings = np.concatenate(all_embeddings, axis=0)
    np.save(output_dir / "embeddings.npy", embeddings)
    merged_df.to_csv(output_dir / "metadata.csv", index=False)

    manifest = {
        "created_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "image_glob": args.image_glob,
        "labels_csv": str(Path(args.labels_csv).resolve()),
        "id_column": args.id_column,
        "image_id_regex": args.image_id_regex,
        "strip_trailing_segment": bool(args.strip_trailing_segment),
        "model": args.model,
        "pretrained": args.pretrained,
        "device": args.device,
        "normalize_embeddings": bool(args.normalize),
        "num_images": int(embeddings.shape[0]),
        "embedding_dim": int(embeddings.shape[1]),
        "num_csv_duplicates_dropped": dup_count,
        "num_unmatched_images": unmatched_count,
        "files": {
            "embeddings_npy": "embeddings.npy",
            "metadata_csv": "metadata.csv",
        },
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Wrote embeddings to: {output_dir}", flush=True)
    print(" - embeddings.npy", flush=True)
    print(" - metadata.csv", flush=True)
    print(" - manifest.json", flush=True)


def _get_plotly_js() -> str:
    try:
        from plotly.offline.offline import get_plotlyjs  # type: ignore
    except Exception as exc:
        raise SystemExit(
            "Could not import plotly. Install `plotly` to use the visualize command."
        ) from exc
    return get_plotlyjs()


def _run_tsne(embeddings: np.ndarray, perplexity: float, random_state: int) -> np.ndarray:
    try:
        from sklearn.manifold import TSNE  # type: ignore
    except Exception as exc:
        raise SystemExit(
            "Could not import scikit-learn. Install `scikit-learn` to run t-SNE."
        ) from exc

    if embeddings.shape[0] < 3:
        raise SystemExit("Need at least 3 samples to compute a t-SNE projection.")

    max_perplexity = max(1.0, float(embeddings.shape[0] - 1))
    chosen_perplexity = min(perplexity, max_perplexity)

    tsne = TSNE(
        n_components=2,
        perplexity=chosen_perplexity,
        random_state=random_state,
        init="pca",
        learning_rate="auto",
    )
    return tsne.fit_transform(embeddings)


def _run_pca(embeddings: np.ndarray, random_state: int) -> np.ndarray:
    try:
        from sklearn.decomposition import PCA  # type: ignore
    except Exception as exc:
        raise SystemExit(
            "Could not import scikit-learn. Install `scikit-learn` to run PCA."
        ) from exc
    pca = PCA(n_components=2, random_state=random_state)
    return pca.fit_transform(embeddings)


def _run_umap(
    embeddings: np.ndarray,
    random_state: int,
    n_neighbors: int,
    min_dist: float,
    metric: str,
) -> np.ndarray:
    try:
        import umap.umap_ as umap  # type: ignore
    except Exception as exc:
        raise SystemExit(
            "Could not import umap-learn. Install `umap-learn` to run UMAP."
        ) from exc
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
        n_jobs=1,
    )
    return reducer.fit_transform(embeddings)


def _run_projection(args: argparse.Namespace, embeddings: np.ndarray) -> np.ndarray:
    method = args.method.lower()
    if method == "tsne":
        return _run_tsne(embeddings, perplexity=args.perplexity, random_state=args.seed)
    if method == "pca":
        return _run_pca(embeddings, random_state=args.seed)
    if method == "umap":
        return _run_umap(
            embeddings,
            random_state=args.seed,
            n_neighbors=args.umap_n_neighbors,
            min_dist=args.umap_min_dist,
            metric=args.umap_metric,
        )
    raise SystemExit(f"Unknown method: {args.method}")


def _write_interactive_html(
    out_html: Path,
    records: list[dict[str, object]],
    label_columns: list[str],
    primary_default: str,
    secondary_default: str,
    projection_name: str,
) -> None:
    plotly_js = _get_plotly_js()
    records_json = json.dumps(records, ensure_ascii=False)
    label_columns_json = json.dumps(label_columns, ensure_ascii=False)
    primary_default_json = json.dumps(primary_default, ensure_ascii=False)
    secondary_default_json = json.dumps(secondary_default, ensure_ascii=False)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>BioCLIP t-SNE</title>
  <style>
    body {{
      font-family: "Segoe UI", Tahoma, sans-serif;
      margin: 0;
      padding: 1rem;
      background: #f8fafc;
      color: #0f172a;
    }}
    .wrap {{
      max-width: 1400px;
      margin: 0 auto;
      background: #ffffff;
      border: 1px solid #e2e8f0;
      border-radius: 12px;
      box-shadow: 0 8px 30px rgba(2, 6, 23, 0.08);
      padding: 1rem 1rem 0.5rem 1rem;
    }}
    .controls {{
      display: flex;
      flex-wrap: wrap;
      gap: 1rem;
      align-items: center;
      margin-bottom: 0.5rem;
    }}
    .controls label {{
      font-size: 0.95rem;
      color: #334155;
    }}
    .controls select {{
      margin-left: 0.5rem;
      border: 1px solid #cbd5e1;
      border-radius: 8px;
      padding: 0.25rem 0.5rem;
      background: #ffffff;
      font-size: 0.95rem;
    }}
    #plot {{
      width: 100%;
      height: 82vh;
      min-height: 560px;
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="controls">
      <label>Primary label
        <select id="primary-select"></select>
      </label>
      <label>Secondary label
        <select id="secondary-select"></select>
      </label>
    </div>
    <div id="plot"></div>
  </div>
  <script>{plotly_js}</script>
  <script>
    const records = {records_json};
    const labelColumns = {label_columns_json};
    const primaryDefault = {primary_default_json};
    const secondaryDefault = {secondary_default_json};
    const palette = [
      "#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#e377c2",
      "#7f7f7f", "#bcbd22", "#17becf", "#4c78a8", "#f58518", "#54a24b", "#e45756",
      "#72b7b2", "#b279a2", "#ff9da6", "#9d755d", "#bab0ab"
    ];
    // Strong secondary contrast while preserving primary hue.

    function safeValue(row, col) {{
      const v = row[col];
      if (v === null || v === undefined) return "(missing)";
      const s = String(v).trim();
      return s.length ? s : "(missing)";
    }}

    function uniqueSorted(values) {{
      return Array.from(new Set(values)).sort((a, b) => a.localeCompare(b));
    }}

    function clamp(v, lo, hi) {{
      return Math.max(lo, Math.min(hi, v));
    }}

    function hexToRgb(hex) {{
      const h = hex.replace("#", "");
      const n = parseInt(h, 16);
      return {{ r: (n >> 16) & 255, g: (n >> 8) & 255, b: n & 255 }};
    }}

    function rgbToHsl(r, g, b) {{
      let rn = r / 255, gn = g / 255, bn = b / 255;
      const max = Math.max(rn, gn, bn), min = Math.min(rn, gn, bn);
      let h = 0, s = 0;
      const l = (max + min) / 2;
      if (max !== min) {{
        const d = max - min;
        s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
        switch (max) {{
          case rn: h = (gn - bn) / d + (gn < bn ? 6 : 0); break;
          case gn: h = (bn - rn) / d + 2; break;
          default: h = (rn - gn) / d + 4; break;
        }}
        h /= 6;
      }}
      return {{ h: h * 360, s: s * 100, l: l * 100 }};
    }}

    function styleColor(baseHex, style) {{
      const rgb = hexToRgb(baseHex);
      const hsl = rgbToHsl(rgb.r, rgb.g, rgb.b);
      const s = clamp(hsl.s * style.satMul, 8, 100);
      const l = clamp(hsl.l + style.lightDelta, 10, 90);
      const a = clamp(style.alpha, 0.28, 1.0);
      return `hsla(${{hsl.h.toFixed(1)}}, ${{s.toFixed(1)}}%, ${{l.toFixed(1)}}%, ${{a.toFixed(3)}})`;
    }}

    function secondaryStyle(index, total) {{
      // Spread classes across the full style range for maximal visual separation.
      const denom = Math.max(1, total - 1);
      const t = index / denom; // 0..1
      const satMul = 1.40 - (0.70 * t);   // 1.40 -> 0.70
      const lightDelta = -22 + (44 * t);  // -22 -> +22
      const alpha = 0.98 - (0.18 * t);    // 0.98 -> 0.80
      return {{
        satMul: satMul,
        lightDelta: lightDelta,
        alpha: alpha
      }};
    }}

    function buildTraces(primaryCol, secondaryCol) {{
      const primaryValues = uniqueSorted(records.map((r) => safeValue(r, primaryCol)));
      const secondaryValues = uniqueSorted(records.map((r) => safeValue(r, secondaryCol)));
      const colorMap = Object.fromEntries(primaryValues.map((v, i) => [v, palette[i % palette.length]]));
      const secondaryIndexMap = Object.fromEntries(secondaryValues.map((v, i) => [v, i]));

      const groups = new Map();
      for (const row of records) {{
        const p = safeValue(row, primaryCol);
        const s = safeValue(row, secondaryCol);
        const key = `${{p}}||${{s}}`;
        if (!groups.has(key)) {{
          groups.set(key, {{ p, s, x: [], y: [], customdata: [] }});
        }}
        const g = groups.get(key);
        g.x.push(row.tsne_x);
        g.y.push(row.tsne_y);
        g.customdata.push([row.image_id || "", row.image_name || "", row.image_path || "", p, s]);
      }}

      const traces = [];
      const orderedGroups = Array.from(groups.values()).sort((a, b) => {{
        const pCmp = a.p.localeCompare(b.p);
        if (pCmp !== 0) return pCmp;
        return a.s.localeCompare(b.s);
      }});
      for (const g of orderedGroups) {{
        const secIdx = secondaryIndexMap[g.s] ?? 0;
        const style = secondaryStyle(secIdx, secondaryValues.length);
        const fillColor = styleColor(colorMap[g.p], style);
        traces.push({{
          type: "scattergl",
          mode: "markers",
          name: `${{g.p}} | ${{g.s}}`,
          x: g.x,
          y: g.y,
          customdata: g.customdata,
          marker: {{
            size: 9,
            color: fillColor,
            symbol: "circle",
            line: {{ width: 0 }}
          }},
          hovertemplate:
            "Image ID: %{{customdata[0]}}<br>" +
            "Image: %{{customdata[1]}}<br>" +
            "Path: %{{customdata[2]}}<br>" +
            `${{primaryCol}}: %{{customdata[3]}}<br>` +
            `${{secondaryCol}}: %{{customdata[4]}}<extra></extra>`
        }});
      }}
      return traces;
    }}

    function renderPlot() {{
      const primaryCol = document.getElementById("primary-select").value;
      const secondaryCol = document.getElementById("secondary-select").value;
      const traces = buildTraces(primaryCol, secondaryCol);
          const layout = {{
        title: {{
          text: `BioCLIP Embeddings {projection_name.upper()}<br><sup>Primary: ${{primaryCol}} | Secondary: ${{secondaryCol}}</sup>`
        }},
        legend: {{
          title: {{ text: "Primary | Secondary" }},
          orientation: "v"
        }},
        xaxis: {{ title: "{projection_name.upper()} 1", zeroline: false, gridcolor: "rgba(148,163,184,0.2)" }},
        yaxis: {{ title: "{projection_name.upper()} 2", zeroline: false, gridcolor: "rgba(148,163,184,0.2)" }},
        paper_bgcolor: "#ffffff",
        plot_bgcolor: "#ffffff",
        margin: {{ l: 60, r: 20, t: 70, b: 55 }}
      }};
      Plotly.react("plot", traces, layout, {{ responsive: true }});
    }}

    function setUpSelectors() {{
      const pSel = document.getElementById("primary-select");
      const sSel = document.getElementById("secondary-select");
      for (const c of labelColumns) {{
        const optP = document.createElement("option");
        optP.value = c;
        optP.textContent = c;
        pSel.appendChild(optP);
        const optS = document.createElement("option");
        optS.value = c;
        optS.textContent = c;
        sSel.appendChild(optS);
      }}
      pSel.value = labelColumns.includes(primaryDefault) ? primaryDefault : labelColumns[0];
      sSel.value = labelColumns.includes(secondaryDefault) ? secondaryDefault : labelColumns[0];
      pSel.addEventListener("change", renderPlot);
      sSel.addEventListener("change", renderPlot);
    }}

    setUpSelectors();
    renderPlot();
  </script>
</body>
</html>
"""

    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html, encoding="utf-8")


def run_visualize(args: argparse.Namespace) -> None:
    stage_bar = tqdm(total=4, desc="Visualize", unit="step") if tqdm is not None else None

    input_dir = Path(args.input_dir).resolve()
    embeddings_path = input_dir / "embeddings.npy"
    metadata_path = input_dir / "metadata.csv"
    if not embeddings_path.exists():
        raise SystemExit(f"Missing embeddings file: {embeddings_path}")
    if not metadata_path.exists():
        raise SystemExit(f"Missing metadata file: {metadata_path}")

    embeddings = np.load(embeddings_path)
    meta_df = pd.read_csv(metadata_path)
    if stage_bar is not None:
        stage_bar.update(1)
        stage_bar.set_postfix_str("Loaded embeddings + metadata")
    if len(meta_df) != embeddings.shape[0]:
        raise SystemExit(
            f"Row mismatch: metadata has {len(meta_df)} rows but embeddings has {embeddings.shape[0]} rows."
        )

    if args.labels_csv:
        if "image_name" not in meta_df.columns:
            raise SystemExit(
                "metadata.csv is missing `image_name`; cannot remap labels in visualize."
            )
        labels_df, dup_count = _prepare_labels(Path(args.labels_csv), args.id_column)
        recalculated_ids = meta_df["image_name"].astype(str).map(
            lambda n: _extract_image_id(
                Path(n), args.image_id_regex, args.strip_trailing_segment
            )
        )
        # Replace prior label columns with a fresh join from the provided labels CSV.
        meta_df = meta_df.drop(
            columns=[c for c in labels_df.columns if c in meta_df.columns], errors="ignore"
        )
        meta_df["image_id"] = recalculated_ids
        meta_df = meta_df.merge(labels_df, how="left", left_on="image_id", right_on=args.id_column)
        meta_df["_csv_match_status"] = np.where(meta_df[args.id_column].isna(), "unmatched", "matched")
        matched_count = int((meta_df["_csv_match_status"] == "matched").sum())
        unmatched_count = int((meta_df["_csv_match_status"] == "unmatched").sum())
        print(
            f"Rejoined labels in visualize (matched={matched_count}, unmatched={unmatched_count}).",
            flush=True,
        )
        if dup_count:
            print(
                f"Warning: dropped {dup_count} duplicate CSV rows based on `{args.id_column}` (kept first).",
                flush=True,
            )

    tsne_xy = _run_projection(args, embeddings)
    if stage_bar is not None:
        stage_bar.update(1)
        stage_bar.set_postfix_str(f"Computed {args.method.upper()}")
    vis_df = meta_df.copy()
    vis_df["tsne_x"] = tsne_xy[:, 0]
    vis_df["tsne_y"] = tsne_xy[:, 1]

    if "_csv_match_status" in vis_df.columns:
        matched_count = int((vis_df["_csv_match_status"] == "matched").sum())
        if matched_count == 0:
            print(
                "Warning: 0 matched labels found in metadata. "
                "All label columns will collapse to missing values.",
                flush=True,
            )

    label_columns = args.label_columns or _select_default_label_columns(vis_df)
    if not label_columns:
        raise SystemExit("No label columns are available for plotting.")
    label_columns = _dedupe_preserve_order(label_columns)

    unknown_cols = [c for c in label_columns if c not in vis_df.columns]
    if unknown_cols:
        raise SystemExit(f"Unknown label columns: {unknown_cols}")

    low_variance = []
    for col in label_columns:
        series = vis_df[col]
        non_missing = series.dropna().astype(str).str.strip()
        non_missing = non_missing[non_missing != ""]
        if non_missing.nunique() <= 1:
            low_variance.append(col)
    if low_variance:
        print(
            "Warning: these label columns have <=1 non-missing unique value "
            f"and may not visibly separate points: {low_variance}",
            flush=True,
        )

    primary_col = args.primary_label or label_columns[0]
    secondary_col = args.secondary_label or label_columns[min(1, len(label_columns) - 1)]
    if primary_col not in label_columns:
        raise SystemExit(f"Primary label column `{primary_col}` is not in --label-columns.")
    if secondary_col not in label_columns:
        raise SystemExit(f"Secondary label column `{secondary_col}` is not in --label-columns.")

    keep_cols = ["image_id", "image_name", "image_path", "tsne_x", "tsne_y"] + label_columns
    keep_cols = _dedupe_preserve_order([c for c in keep_cols if c in vis_df.columns])
    records_df = vis_df[keep_cols].copy()
    records_df = _normalize_text_columns(records_df)
    records = records_df.to_dict(orient="records")

    output_html = Path(args.output_html).resolve()
    _write_interactive_html(
        out_html=output_html,
        records=records,
        label_columns=label_columns,
        primary_default=primary_col,
        secondary_default=secondary_col,
        projection_name=args.method,
    )
    if stage_bar is not None:
        stage_bar.update(1)
        stage_bar.set_postfix_str("Wrote HTML")

    tsne_csv_path = output_html.with_suffix(".tsne.csv")
    vis_df.to_csv(tsne_csv_path, index=False)
    if stage_bar is not None:
        stage_bar.update(1)
        stage_bar.set_postfix_str("Wrote CSV")
        stage_bar.close()
    print(f"Wrote interactive plot: {output_html}", flush=True)
    print(f"Wrote t-SNE coordinates: {tsne_csv_path}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bioclip-tsne",
        description="Generate BioCLIP embeddings and interactive labeled t-SNE plots.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    embed = subparsers.add_parser("embed", help="Create BioCLIP embeddings and metadata artifacts.")
    embed.add_argument(
        "--image-glob",
        required=True,
        help="Glob pattern for image files (example: '/path/to/idi/*.jpg').",
    )
    embed.add_argument("--labels-csv", required=True, help="Path to labels CSV file.")
    embed.add_argument(
        "--id-column",
        default="Sample ID",
        help="CSV column used to join labels to images (default: 'Sample ID').",
    )
    embed.add_argument(
        "--image-id-regex",
        default=None,
        help=(
            "Optional regex to extract image_id from filename stem. "
            "Uses named group 'id' if present, else first capture group, else full match."
        ),
    )
    embed.add_argument(
        "--strip-trailing-segment",
        action="store_true",
        help=(
            "If set, remove a trailing '-<1-2 digits>' segment from filename stem before "
            "joining to CSV (example: ZOE-0013-01-2 -> ZOE-0013-01)."
        ),
    )
    embed.add_argument(
        "--output-dir",
        default="./artifacts/bioclip_embeddings",
        help="Output directory for embeddings and metadata.",
    )
    embed.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Compute device for BioCLIP embeddings.",
    )
    embed.add_argument(
        "--model",
        default="hf-hub:imageomics/bioclip-2",
        help="BioCLIP model string passed to TreeOfLifeClassifier.",
    )
    embed.add_argument(
        "--pretrained",
        default=None,
        help="Optional pretrained tag/checkpoint compatible with the selected model.",
    )
    embed.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Number of images per embedding batch.",
    )
    embed.add_argument(
        "--normalize",
        action="store_true",
        help="If set, L2-normalize embeddings before saving.",
    )

    vis = subparsers.add_parser("visualize", help="Create interactive t-SNE HTML from saved embeddings.")
    vis.add_argument(
        "--input-dir",
        required=True,
        help="Directory created by the `embed` command (must contain embeddings.npy and metadata.csv).",
    )
    vis.add_argument(
        "--output-html",
        required=True,
        help="Path to write interactive HTML visualization.",
    )
    vis.add_argument(
        "--label-columns",
        nargs="+",
        default=None,
        help="Subset of metadata columns to make selectable in the visualization UI.",
    )
    vis.add_argument(
        "--labels-csv",
        default=None,
        help=(
            "Optional labels CSV to remap labels at visualize-time without recomputing embeddings."
        ),
    )
    vis.add_argument(
        "--id-column",
        default="Sample ID",
        help="CSV column used to join labels when --labels-csv is provided.",
    )
    vis.add_argument(
        "--image-id-regex",
        default=None,
        help=(
            "Optional regex to extract image_id from image filename stem when --labels-csv is used."
        ),
    )
    vis.add_argument(
        "--strip-trailing-segment",
        action="store_true",
        help=(
            "When remapping labels (--labels-csv), remove a trailing '-<1-2 digits>' "
            "segment before joining."
        ),
    )
    vis.add_argument(
        "--primary-label",
        default=None,
        help="Default primary label column for color coding.",
    )
    vis.add_argument(
        "--secondary-label",
        default=None,
        help="Default secondary label column for marker symbol.",
    )
    vis.add_argument(
        "--method",
        choices=["tsne", "pca", "umap"],
        default="tsne",
        help="Projection method for 2D visualization (default: tsne).",
    )
    vis.add_argument(
        "--perplexity",
        type=float,
        default=30.0,
        help="t-SNE perplexity (automatically clipped to valid range). Used only when --method tsne.",
    )
    vis.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for projection method.",
    )
    vis.add_argument(
        "--umap-n-neighbors",
        type=int,
        default=15,
        help="UMAP n_neighbors. Used only when --method umap.",
    )
    vis.add_argument(
        "--umap-min-dist",
        type=float,
        default=0.1,
        help="UMAP min_dist. Used only when --method umap.",
    )
    vis.add_argument(
        "--umap-metric",
        default="euclidean",
        help="UMAP metric. Used only when --method umap.",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "embed":
        run_embed(args)
    elif args.command == "visualize":
        run_visualize(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
