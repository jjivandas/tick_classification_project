"""
SVM-Aware Embedding Explorer
============================

Builds an interactive t-SNE/UMAP viewer over multiple shot counts:
  - k = 0  -> raw BioCLIP embeddings (512-D)
  - k > 0  -> per-specimen mean SVC.predict_proba (5-D) averaged over 100
              Monte-Carlo runs, using the SAME seed scheme as notebook 04.

The script never re-embeds images; it reads the BioCLIP cache from
data/processed/emb_cache. Hyperparameters are auto-picked: the config in
results/svm_hp/runs/260405_225314 with the highest 35-shot mean_macro is
selected, and its parameters are applied to the new run.

Output: results/tsne/runs/<YYMMDD_HHMMSS>/{coords.csv, proba_mean_k*.npy,
manifest.json, viewer.html}.
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
from utils.paths import load_paths  # noqa: E402

paths = load_paths()
EMB_CACHE = paths["emb_cache_dir"]
PROCESSED_DIR = paths["processed_dir"]
RESULTS_DIR = paths["results_dir"]
HP_RUN_DIR = RESULTS_DIR / "svm_hp" / "runs" / "260405_225314"
TSNE_RUNS_DIR = RESULTS_DIR / "tsne" / "runs"

SHOTS_FOR_VIEWER = [0, 1, 10, 25, 35]
SHOTS_FOR_PROBA = [1, 10, 25, 35]
N_RUNS = 100
BASE_SEED = 42
MIN_TOTAL = 50

SPECIES_PALETTE = {
    "Amblyomma americanum": "#1f77b4",
    "Dermacentor variabilis": "#ff7f0e",
    "Haemaphysalis leporispalustris": "#2ca02c",
    "Haemaphysalis longicornis": "#fc0202",
    "Ixodes scapularis": "#8213ea",
}

METADATA_FIELDS = [
    "species",
    "sex",
    "life_stage",
    "attached",
    "pathogen",
    "pathogen_result",
    "tick_condition",
]


def seed_for(k: int, run_id: int) -> int:
    """Hash-based seed identical to notebook 04 (line 141)."""
    tag = f"{BASE_SEED}_{k}_{run_id}"
    return int(hashlib.sha256(tag.encode()).hexdigest()[:8], 16) % (2**31)


def cache_fp(img_path: str) -> Path:
    h = hashlib.sha256(img_path.encode("utf-8")).hexdigest()[:24]
    return EMB_CACHE / f"{h}.npy"


def load_specimens():
    """Build by_species[species][sample_id] = {'dorsal':..., 'ventral':...}.

    Mirrors notebook 04 Block 1: keep specimens with both views, drop species
    with < MIN_TOTAL specimens.
    """
    data_path = PROCESSED_DIR / "final_data_cleaned.json"
    class_names_path = PROCESSED_DIR / "class_names_cleaned.json"
    with open(class_names_path) as f:
        species_set = set(json.load(f))
    with open(data_path) as f:
        records = json.load(f)

    by_species: dict[str, dict[str, dict[str, str]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    meta_by_sid: dict[str, dict] = {}
    for r in records:
        sp = r["true_label"]
        if sp not in species_set:
            continue
        sid = r["sample_id"]
        view = str(r["view"]).strip().lower()
        by_species[sp][sid][view] = r["image_path"]
        if sid not in meta_by_sid:
            meta_by_sid[sid] = {
                "sample_id": sid,
                "species": sp,
                "sex": _norm(r.get("sex")),
                "life_stage": _norm(r.get("life_stage")),
                "attached": _norm(r.get("attached")),
                "pathogen": _norm(r.get("pathogen"), none_label="None"),
                "pathogen_result": _norm(r.get("pathogen_result")),
                "tick_condition": _norm(r.get("tick_condition")),
            }

    for sp in list(by_species.keys()):
        for sid in list(by_species[sp].keys()):
            views = by_species[sp][sid]
            if not ("dorsal" in views and "ventral" in views):
                del by_species[sp][sid]
        if not by_species[sp]:
            del by_species[sp]

    for sp in list(by_species.keys()):
        if len(by_species[sp]) < MIN_TOTAL:
            del by_species[sp]

    included_species = sorted(by_species.keys())
    return by_species, included_species, meta_by_sid


def _norm(v, none_label: str = "Unknown") -> str:
    if v is None:
        return none_label
    if isinstance(v, float) and np.isnan(v):
        return none_label
    s = str(v).strip()
    if not s or s.lower() == "nan":
        return none_label
    if s.lower() == "none":
        return none_label
    return s


def specimen_vec(rec: dict) -> np.ndarray:
    """Average dorsal + ventral cached BioCLIP embeddings."""
    z_d = np.load(cache_fp(rec["dorsal"]))
    z_v = np.load(cache_fp(rec["ventral"]))
    return 0.5 * (z_d + z_v)


def build_specimen_table(by_species, included_species, meta_by_sid):
    """Stack one row per specimen: X[N, 512] + meta_df (N rows)."""
    sids: list[str] = []
    species_of: list[str] = []
    X: list[np.ndarray] = []
    for sp in included_species:
        for sid in sorted(by_species[sp].keys()):
            X.append(specimen_vec(by_species[sp][sid]))
            sids.append(sid)
            species_of.append(sp)
    X_arr = np.stack(X).astype(np.float32)
    meta_rows = [meta_by_sid[sid] for sid in sids]
    meta_df = pd.DataFrame(meta_rows)
    return X_arr, meta_df, sids, species_of


def split_once(by_species, species_list, k: int, seed: int):
    """Same as notebook 04: shuffle specimen ids per species, k -> train, rest -> test."""
    rng = random.Random(seed)
    train_pairs, test_pairs = [], []
    for sp in species_list:
        sids = list(by_species[sp].keys())
        rng.shuffle(sids)
        train_pairs.extend([(sp, sid) for sid in sids[:k]])
        test_pairs.extend([(sp, sid) for sid in sids[k:]])
    return train_pairs, test_pairs


def pick_best_hp() -> tuple[str, dict]:
    """Pick HP config with highest 35-shot mean_macro from latest HP run."""
    candidates = ["G_finer", "H_wider", "I_tiny_gamma", "J_logreg"]
    config_params = {
        "G_finer": {"C": 10, "gamma": 0.0005, "kernel": "rbf"},
        "H_wider": {"C": 10, "gamma": 0.005, "kernel": "rbf"},
        "I_tiny_gamma": {"C": 10, "gamma": 0.0001, "kernel": "rbf"},
        "J_logreg": {"type": "logreg"},
    }
    best, best_score = None, -1.0
    for name in candidates:
        summary = HP_RUN_DIR / name / "analysis" / "shot_summary.csv"
        if not summary.exists():
            continue
        df = pd.read_csv(summary)
        row = df[df["shots"] == 35]
        if row.empty:
            continue
        score = float(row["mean_macro"].iloc[0])
        if score > best_score:
            best_score = score
            best = name
    if best is None:
        raise SystemExit(f"No HP configs found in {HP_RUN_DIR}")
    return best, config_params[best]


def make_classifier(params: dict, seed: int) -> Pipeline:
    if params.get("type") == "logreg":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=1000, class_weight="balanced",
                penalty="l2", solver="saga", random_state=seed)),
        ])
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(
            C=params["C"], gamma=params["gamma"], kernel=params["kernel"],
            class_weight="balanced", probability=True, random_state=seed)),
    ])


def compute_proba_mean(
    by_species, included_species, sids: list[str], species_of: list[str],
    X: np.ndarray, hp_name: str, hp_params: dict, k: int,
):
    """Average predict_proba over N_RUNS Monte-Carlo splits at given k.

    Also accumulates per-(k, run_id, sample_id) argmax predictions to verify
    against the existing predictions.csv for the held-out test specimens only.

    Returns (proba_mean[N, n_classes], class_order, agreement_records).
    """
    sid_to_row = {sid: i for i, sid in enumerate(sids)}
    sp_to_label = species_of  # parallel to sids

    proba_sum: np.ndarray | None = None
    class_order: list[str] | None = None
    pred_rows: list[dict] = []

    try:
        from tqdm.auto import tqdm
        run_iter = tqdm(range(N_RUNS), desc=f"  k={k}", leave=False)
    except Exception:  # pragma: no cover
        run_iter = range(N_RUNS)

    for run_id in run_iter:
        seed = seed_for(k, run_id)
        train_pairs, test_pairs = split_once(by_species, included_species, k=k, seed=seed)

        train_idx = [sid_to_row[sid] for _, sid in train_pairs]
        test_idx = [sid_to_row[sid] for _, sid in test_pairs]
        Xtr = X[train_idx]
        ytr = np.array([sp for sp, _ in train_pairs])

        clf = make_classifier(hp_params, seed)
        clf.fit(Xtr, ytr)

        if class_order is None:
            class_order = list(clf.classes_)
            proba_sum = np.zeros((X.shape[0], len(class_order)), dtype=np.float64)

        proba_all = clf.predict_proba(X)  # (N, n_classes)
        proba_sum += proba_all

        # record argmax-on-test for cross-checking against predictions.csv
        proba_test = proba_all[test_idx]
        yhat_test = np.array(class_order)[proba_test.argmax(axis=1)]
        for (sp_true, sid), pred in zip(test_pairs, yhat_test):
            pred_rows.append({
                "shots": k, "run_id": run_id, "sample_id": sid,
                "species_true": sp_true, "species_pred": str(pred),
            })

    proba_mean = proba_sum / N_RUNS
    return proba_mean, class_order, pred_rows


def verify_against_predictions_csv(
    pred_rows: list[dict], hp_name: str, k: int,
) -> tuple[int, int]:
    """Return (matches, total) for argmax-on-test vs the stored predictions.csv."""
    csv_path = HP_RUN_DIR / hp_name / "predictions.csv"
    if not csv_path.exists():
        return 0, 0
    new_df = pd.DataFrame(pred_rows)
    old_df = pd.read_csv(csv_path)
    old_df = old_df[old_df["shots"] == k][
        ["shots", "run_id", "sample_id", "species_pred"]
    ]
    merged = new_df.merge(
        old_df, on=["shots", "run_id", "sample_id"], suffixes=("_new", "_old"),
    )
    if merged.empty:
        return 0, 0
    matches = int((merged["species_pred_new"] == merged["species_pred_old"]).sum())
    return matches, len(merged)


def project_2d(input_arr: np.ndarray, *, perplexity: float = 30.0,
               n_neighbors: int = 15, min_dist: float = 0.1, seed: int = 42,
               label: str = ""):
    import time
    n = input_arr.shape[0]
    perp = float(min(perplexity, max(1.0, n - 1)))

    print(f"    [{label}] t-SNE start (n={n}, dim={input_arr.shape[1]})", flush=True)
    t0 = time.time()
    tsne = TSNE(n_components=2, perplexity=perp, random_state=seed,
                init="pca", learning_rate="auto")
    tsne_xy = tsne.fit_transform(input_arr)
    print(f"    [{label}] t-SNE done in {time.time() - t0:.1f}s", flush=True)

    import umap.umap_ as umap_mod
    print(f"    [{label}] UMAP start (n={n}, dim={input_arr.shape[1]})", flush=True)
    t0 = time.time()
    reducer = umap_mod.UMAP(
        n_components=2, n_neighbors=min(n_neighbors, max(2, n - 1)),
        min_dist=min_dist, random_state=seed, n_jobs=1,
    )
    umap_xy = reducer.fit_transform(input_arr)
    print(f"    [{label}] UMAP done in {time.time() - t0:.1f}s", flush=True)
    return tsne_xy, umap_xy


def build_long_coords(meta_df: pd.DataFrame,
                      coords_per_shot: dict[int, dict[str, np.ndarray]]):
    parts = []
    for k, methods in coords_per_shot.items():
        for method, xy in methods.items():
            chunk = meta_df.copy()
            chunk["shots"] = int(k)
            chunk["method"] = method
            chunk["x"] = xy[:, 0].astype(float)
            chunk["y"] = xy[:, 1].astype(float)
            parts.append(chunk)
    return pd.concat(parts, axis=0, ignore_index=True)


def write_viewer_html(coords_df: pd.DataFrame, out_path: Path,
                      hp_name: str, hp_params: dict, n_specimens: int):
    """Self-contained interactive HTML."""
    import plotly.offline.offline as plotly_offline
    plotly_js = plotly_offline.get_plotlyjs()

    records = coords_df.to_dict(orient="records")
    payload = {
        "records": records,
        "metadata_fields": METADATA_FIELDS,
        "shots": SHOTS_FOR_VIEWER,
        "methods": ["tsne", "umap"],
        "species_palette": SPECIES_PALETTE,
        "default_primary": "species",
        "default_secondary": "sex",
        "default_shots": 35,
        "default_method": "tsne",
        "hp": {"name": hp_name, "params": _stringify_hp(hp_params)},
        "n_specimens": int(n_specimens),
    }
    payload_json = json.dumps(payload, ensure_ascii=False)

    html = _HTML_TEMPLATE.replace("__PLOTLY_JS__", plotly_js).replace(
        "__PAYLOAD_JSON__", payload_json,
    )
    out_path.write_text(html, encoding="utf-8")


def _stringify_hp(params: dict) -> dict:
    return {k: (str(v) if not isinstance(v, (int, float, bool, str)) else v)
            for k, v in params.items()}


_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>SVM Embedding Explorer</title>
<style>
  :root { color-scheme: light; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", sans-serif;
    margin: 0; padding: 1.25rem; background: #f4f6fa; color: #0f172a;
  }
  .wrap {
    max-width: 1480px; margin: 0 auto; background: #ffffff;
    border: 1px solid #e2e8f0; border-radius: 12px;
    box-shadow: 0 6px 24px rgba(15, 23, 42, 0.07); padding: 1rem 1.25rem 0.5rem;
  }
  header h1 { font-size: 1.05rem; margin: 0 0 0.25rem; font-weight: 600; }
  header .subtitle { font-size: 0.85rem; color: #64748b; margin-bottom: 0.85rem; }
  .controls { display: flex; flex-wrap: wrap; gap: 0.85rem 1.4rem; align-items: center;
              padding: 0.5rem 0 0.85rem; border-bottom: 1px solid #eef2f7; }
  .controls > label { font-size: 0.9rem; color: #334155; display: flex; align-items: center; gap: 0.45rem; }
  .controls select { border: 1px solid #cbd5e1; border-radius: 8px; padding: 0.3rem 0.55rem;
                     background: #ffffff; font-size: 0.9rem; color: #0f172a; }
  .shot-group { display: inline-flex; gap: 0.25rem; }
  .shot-group button, .method-group button {
    border: 1px solid #cbd5e1; background: #ffffff; padding: 0.32rem 0.7rem;
    font-size: 0.88rem; border-radius: 8px; cursor: pointer; color: #0f172a;
  }
  .shot-group button.active, .method-group button.active {
    background: #1f2937; color: #ffffff; border-color: #1f2937;
  }
  #plot { width: 100%; height: 78vh; min-height: 540px; }
  footer { font-size: 0.75rem; color: #94a3b8; padding: 0.5rem 0 0; }
</style>
</head>
<body>
  <div class="wrap">
    <header>
      <h1>SVM Embedding Explorer</h1>
      <div class="subtitle" id="subtitle"></div>
    </header>
    <div class="controls">
      <label>Shots
        <span class="shot-group" id="shot-group"></span>
      </label>
      <label>Method
        <span class="method-group" id="method-group"></span>
      </label>
      <label>Primary label
        <select id="primary-select"></select>
      </label>
      <label>Secondary label
        <select id="secondary-select"></select>
      </label>
    </div>
    <div id="plot"></div>
    <footer id="meta-footer"></footer>
  </div>
  <script>__PLOTLY_JS__</script>
  <script>
  const PAYLOAD = __PAYLOAD_JSON__;

  const fallbackPalette = [
    "#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b","#e377c2",
    "#7f7f7f","#bcbd22","#17becf"
  ];

  const state = {
    shots: PAYLOAD.default_shots,
    method: PAYLOAD.default_method,
    primary: PAYLOAD.default_primary,
    secondary: PAYLOAD.default_secondary,
  };

  function safeValue(row, col) {
    const v = row[col];
    if (v === null || v === undefined) return "(missing)";
    const s = String(v).trim();
    return s.length ? s : "(missing)";
  }

  function uniqueSorted(values) {
    return Array.from(new Set(values)).sort((a, b) => a.localeCompare(b));
  }

  function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

  function hexToRgb(hex) {
    const h = hex.replace("#", "");
    const n = parseInt(h, 16);
    return { r: (n >> 16) & 255, g: (n >> 8) & 255, b: n & 255 };
  }

  function rgbToHsl(r, g, b) {
    let rn = r / 255, gn = g / 255, bn = b / 255;
    const max = Math.max(rn, gn, bn), min = Math.min(rn, gn, bn);
    let h = 0, s = 0;
    const l = (max + min) / 2;
    if (max !== min) {
      const d = max - min;
      s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
      switch (max) {
        case rn: h = (gn - bn) / d + (gn < bn ? 6 : 0); break;
        case gn: h = (bn - rn) / d + 2; break;
        default: h = (rn - gn) / d + 4; break;
      }
      h /= 6;
    }
    return { h: h * 360, s: s * 100, l: l * 100 };
  }

  function styleColor(baseHex, style) {
    const rgb = hexToRgb(baseHex);
    const hsl = rgbToHsl(rgb.r, rgb.g, rgb.b);
    const s = clamp(hsl.s * style.satMul, 8, 100);
    const l = clamp(hsl.l + style.lightDelta, 10, 90);
    const a = clamp(style.alpha, 0.28, 1.0);
    return `hsla(${hsl.h.toFixed(1)}, ${s.toFixed(1)}%, ${l.toFixed(1)}%, ${a.toFixed(3)})`;
  }

  function secondaryStyle(index, total) {
    const denom = Math.max(1, total - 1);
    const t = index / denom;
    return {
      satMul: 1.4 - 0.7 * t,
      lightDelta: -22 + 44 * t,
      alpha: 0.98 - 0.18 * t,
    };
  }

  function primaryColorMap(primaryCol, primaryValues) {
    if (primaryCol === "species") {
      const map = {};
      primaryValues.forEach((v, i) => {
        map[v] = PAYLOAD.species_palette[v] || fallbackPalette[i % fallbackPalette.length];
      });
      return map;
    }
    const map = {};
    primaryValues.forEach((v, i) => { map[v] = fallbackPalette[i % fallbackPalette.length]; });
    return map;
  }

  function buildTraces(filteredRows, primaryCol, secondaryCol) {
    const primaryValues = uniqueSorted(filteredRows.map((r) => safeValue(r, primaryCol)));
    const secondaryValues = uniqueSorted(filteredRows.map((r) => safeValue(r, secondaryCol)));
    const colorMap = primaryColorMap(primaryCol, primaryValues);
    const secondaryIndexMap = Object.fromEntries(secondaryValues.map((v, i) => [v, i]));

    const groups = new Map();
    for (const row of filteredRows) {
      const p = safeValue(row, primaryCol);
      const s = safeValue(row, secondaryCol);
      const key = `${p}||${s}`;
      if (!groups.has(key)) groups.set(key, { p, s, x: [], y: [], customdata: [] });
      const g = groups.get(key);
      g.x.push(row.x);
      g.y.push(row.y);
      g.customdata.push([
        row.sample_id || "", row.species || "", row.sex || "", row.life_stage || "",
        row.attached || "", row.pathogen || "", row.pathogen_result || "",
        row.tick_condition || "", p, s,
      ]);
    }

    const ordered = Array.from(groups.values()).sort((a, b) => {
      const c = a.p.localeCompare(b.p);
      return c !== 0 ? c : a.s.localeCompare(b.s);
    });

    const traces = [];
    for (const g of ordered) {
      const idx = secondaryIndexMap[g.s] ?? 0;
      const fill = styleColor(colorMap[g.p], secondaryStyle(idx, secondaryValues.length));
      traces.push({
        type: "scattergl", mode: "markers",
        name: `${g.p} | ${g.s}`,
        x: g.x, y: g.y, customdata: g.customdata,
        marker: { size: 8, color: fill, symbol: "circle", line: { width: 0 } },
        hovertemplate:
          "Sample: %{customdata[0]}<br>" +
          "Species: %{customdata[1]}<br>" +
          "Sex: %{customdata[2]}<br>" +
          "Life stage: %{customdata[3]}<br>" +
          "Attached: %{customdata[4]}<br>" +
          "Pathogen: %{customdata[5]}<br>" +
          "Pathogen result: %{customdata[6]}<br>" +
          "Condition: %{customdata[7]}<br>" +
          `<b>${primaryCol}</b>: %{customdata[8]}<br>` +
          `<b>${secondaryCol}</b>: %{customdata[9]}<extra></extra>`,
      });
    }
    return traces;
  }

  function shotLabel(k) { return k === 0 ? "0 (raw BioCLIP)" : `${k}`; }
  function methodLabel(m) { return m === "tsne" ? "t-SNE" : "UMAP"; }

  function render() {
    const rows = PAYLOAD.records.filter(
      (r) => r.shots === state.shots && r.method === state.method
    );
    const traces = buildTraces(rows, state.primary, state.secondary);
    const inputDesc = state.shots === 0
      ? "raw BioCLIP (512-D)"
      : `mean SVM predict_proba (5-D, ${state.shots}-shot, 100 MC runs)`;
    const layout = {
      title: {
        text: `${methodLabel(state.method)} of ${inputDesc}` +
              `<br><sup>Primary: ${state.primary}  ·  Secondary: ${state.secondary}  ·  N=${rows.length} specimens</sup>`,
        font: { size: 14 },
      },
      legend: { title: { text: "Primary | Secondary" }, orientation: "v" },
      xaxis: { title: `${methodLabel(state.method)} 1`, zeroline: false, gridcolor: "rgba(148,163,184,0.18)" },
      yaxis: { title: `${methodLabel(state.method)} 2`, zeroline: false, gridcolor: "rgba(148,163,184,0.18)" },
      paper_bgcolor: "#ffffff", plot_bgcolor: "#ffffff",
      margin: { l: 60, r: 24, t: 70, b: 56 },
    };
    Plotly.react("plot", traces, layout, { responsive: true });
    document.getElementById("subtitle").textContent =
      `HP: ${PAYLOAD.hp.name}  ·  ${formatHp(PAYLOAD.hp.params)}  ·  ${PAYLOAD.n_specimens} specimens, dorsal+ventral averaged`;
  }

  function formatHp(p) {
    return Object.entries(p).map(([k, v]) => `${k}=${v}`).join(", ");
  }

  function setupShotGroup() {
    const root = document.getElementById("shot-group");
    PAYLOAD.shots.forEach((k) => {
      const b = document.createElement("button");
      b.type = "button";
      b.textContent = shotLabel(k);
      if (k === state.shots) b.classList.add("active");
      b.addEventListener("click", () => {
        state.shots = k;
        root.querySelectorAll("button").forEach((x) => x.classList.remove("active"));
        b.classList.add("active");
        render();
      });
      root.appendChild(b);
    });
  }

  function setupMethodGroup() {
    const root = document.getElementById("method-group");
    PAYLOAD.methods.forEach((m) => {
      const b = document.createElement("button");
      b.type = "button";
      b.textContent = methodLabel(m);
      if (m === state.method) b.classList.add("active");
      b.addEventListener("click", () => {
        state.method = m;
        root.querySelectorAll("button").forEach((x) => x.classList.remove("active"));
        b.classList.add("active");
        render();
      });
      root.appendChild(b);
    });
  }

  function setupSelects() {
    const prim = document.getElementById("primary-select");
    const sec = document.getElementById("secondary-select");
    PAYLOAD.metadata_fields.forEach((f) => {
      const op = document.createElement("option"); op.value = f; op.textContent = f; prim.appendChild(op);
      const os = document.createElement("option"); os.value = f; os.textContent = f; sec.appendChild(os);
    });
    prim.value = state.primary;
    sec.value = state.secondary;
    prim.addEventListener("change", () => { state.primary = prim.value; render(); });
    sec.addEventListener("change", () => { state.secondary = sec.value; render(); });
  }

  setupShotGroup();
  setupMethodGroup();
  setupSelects();
  render();
  </script>
</body>
</html>
"""


def main():
    sys.stdout.reconfigure(line_buffering=True)
    print("=" * 64)
    print("  SVM Embedding Explorer")
    print("=" * 64)

    by_species, included_species, meta_by_sid = load_specimens()
    n_specimens = sum(len(s) for s in by_species.values())
    print(f"Included species: {len(included_species)} | total specimens: {n_specimens}")

    print("Loading cached BioCLIP embeddings + averaging dorsal/ventral...")
    X, meta_df, sids, species_of = build_specimen_table(
        by_species, included_species, meta_by_sid,
    )
    print(f"  X shape: {X.shape}  meta_df: {meta_df.shape}")

    hp_name, hp_params = pick_best_hp()
    print(f"Picked HP config: {hp_name} ({hp_params})")

    proba_means: dict[int, np.ndarray] = {}
    class_orders: dict[int, list[str]] = {}
    repro_summary: list[dict] = []

    for k in SHOTS_FOR_PROBA:
        print(f"Running {N_RUNS} MC runs at k={k}...")
        proba_mean, class_order, pred_rows = compute_proba_mean(
            by_species, included_species, sids, species_of,
            X, hp_name, hp_params, k,
        )
        proba_means[k] = proba_mean
        class_orders[k] = class_order

        row_sums = proba_mean.sum(axis=1)
        sum_max_dev = float(np.abs(row_sums - 1.0).max())

        matches, total = verify_against_predictions_csv(pred_rows, hp_name, k)
        agreement = (matches / total) if total else None
        repro_summary.append({
            "shots": k, "matches": matches, "total": total,
            "agreement": agreement, "row_sum_max_dev": sum_max_dev,
        })
        print(f"  proba row-sum max deviation from 1.0: {sum_max_dev:.2e}")
        if total:
            print(f"  agreement vs predictions.csv [{hp_name}]: "
                  f"{matches}/{total} = {agreement * 100:.2f}%")

    print("Projecting all shots with t-SNE + UMAP...", flush=True)
    coords_per_shot: dict[int, dict[str, np.ndarray]] = {}
    for k in SHOTS_FOR_VIEWER:
        if k == 0:
            input_arr = X
        else:
            input_arr = proba_means[k]
        print(f"  -> k={k} input dim={input_arr.shape[1]}", flush=True)
        tsne_xy, umap_xy = project_2d(input_arr, label=f"k={k}")
        coords_per_shot[k] = {"tsne": tsne_xy, "umap": umap_xy}

    coords_df = build_long_coords(meta_df, coords_per_shot)

    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    out_dir = TSNE_RUNS_DIR / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    coords_df.to_csv(out_dir / "coords.csv", index=False)
    for k, p in proba_means.items():
        np.save(out_dir / f"proba_mean_k{k}.npy", p)
        with open(out_dir / f"class_order_k{k}.json", "w") as f:
            json.dump(class_orders[k], f)

    if class_orders:
        canonical_order = class_orders[max(class_orders)]
        argmax_at_max_k = np.array(canonical_order)[
            proba_means[max(class_orders)].argmax(axis=1)
        ]
        species_recovery = float(
            (argmax_at_max_k == meta_df["species"].to_numpy()).mean()
        )
    else:
        species_recovery = None

    manifest = {
        "timestamp": timestamp,
        "shots_for_viewer": SHOTS_FOR_VIEWER,
        "shots_for_proba": SHOTS_FOR_PROBA,
        "n_runs": N_RUNS,
        "base_seed": BASE_SEED,
        "seed_scheme": "sha256(f'{BASE_SEED}_{k}_{run_id}').hexdigest()[:8] mod 2^31  (matches notebook 04 line 141)",
        "hp_config": hp_name,
        "hp_params": _stringify_hp(hp_params),
        "hp_source_run": str(HP_RUN_DIR.relative_to(REPO_ROOT)),
        "n_specimens": int(X.shape[0]),
        "embedding_dim_raw": int(X.shape[1]),
        "min_total_per_species": MIN_TOTAL,
        "metadata_fields": METADATA_FIELDS,
        "species_palette": SPECIES_PALETTE,
        "reproducibility_check": repro_summary,
        "species_recovery_argmax_at_max_k": species_recovery,
        "outputs": {
            "coords_csv": "coords.csv",
            "proba_npy": [f"proba_mean_k{k}.npy" for k in SHOTS_FOR_PROBA],
            "class_order_json": [f"class_order_k{k}.json" for k in SHOTS_FOR_PROBA],
            "viewer_html": "viewer.html",
        },
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print("Writing viewer.html...")
    write_viewer_html(coords_df, out_dir / "viewer.html",
                      hp_name, hp_params, n_specimens=X.shape[0])

    latest_path = TSNE_RUNS_DIR / "latest_run.txt"
    latest_path.write_text(str(out_dir))

    print("=" * 64)
    print(f"  Done. Outputs at: {out_dir}")
    if species_recovery is not None:
        print(f"  argmax at k={max(class_orders)} matches true species in "
              f"{species_recovery * 100:.1f}% of specimens")
    print("=" * 64)


if __name__ == "__main__":
    main()
