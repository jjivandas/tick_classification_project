"""Compare both / dorsal / ventral runs of notebooks 02 & 03.

Reads the latest tagged run dir for each VIEW_MODE and produces:
  results/comparison/<timestamp>/
    macro_curves.png       - SVM macro acc (3 curves) + BioCLIP baselines
    overall_curves.png     - SVM overall acc (3 curves)
    svm_summary.csv        - long-form (mode, shots, mean_macro, ci95, ...)
    bioclip_summary.csv    - (mode, macro_acc, n_species)
    README.md              - short text summary

For "both" mode the run dir may have no _both suffix (legacy runs predate the
VIEW_MODE refactor and are algebraically identical). Falls back to the latest
untagged run when no _both suffix exists.
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from utils.paths import load_paths  # noqa: E402

paths = load_paths()
SVM_RUNS = Path(paths["svm_results_dir"]) / "runs"
BC_RUNS = Path(paths["bioclip_results_dir"]) / "runs"
OUT_ROOT = Path(paths["results_dir"]) / "comparison"

MODES = ["both", "dorsal", "ventral"]
MODE_COLORS = {"both": "#000000", "dorsal": "#1f77b4", "ventral": "#d62728"}


def latest_run(runs_dir: Path, mode: str) -> Path | None:
    """Latest run dir for a mode. For 'both', falls back to untagged dirs."""
    if not runs_dir.exists():
        return None
    suffix = f"_{mode}"
    tagged = sorted(p for p in runs_dir.iterdir()
                    if p.is_dir() and p.name.endswith(suffix))
    if tagged:
        return tagged[-1]
    if mode == "both":
        untagged = sorted(p for p in runs_dir.iterdir()
                          if p.is_dir() and not any(p.name.endswith(f"_{m}") for m in MODES))
        if untagged:
            return untagged[-1]
    return None


def load_svm_shot_summary(run_dir: Path) -> pd.DataFrame | None:
    """Prefer shot_summary.csv (richer); fall back to aggregated_metrics."""
    p = run_dir / "analysis" / "shot_summary.csv"
    if p.exists():
        df = pd.read_csv(p)
        return df
    p = run_dir / "analysis" / "aggregated_metrics.csv"
    if p.exists():
        df = pd.read_csv(p)
        agg = (df.groupby("shots", as_index=False)
                 .agg(mean_macro=("macro_acc", "mean"),
                      std_macro=("macro_acc", "std"),
                      n_runs_macro=("macro_acc", "size"),
                      mean_overall=("overall_acc", "mean"),
                      std_overall=("overall_acc", "std")))
        agg["ci95_macro"] = 1.96 * agg["std_macro"] / np.sqrt(agg["n_runs_macro"])
        agg["ci95_overall"] = 1.96 * agg["std_overall"] / np.sqrt(agg["n_runs_macro"])
        return agg
    return None


def load_bc_macro(run_dir: Path) -> tuple[float, list[str]] | None:
    p = run_dir / "bioclip_results.npz"
    if not p.exists():
        return None
    d = np.load(p, allow_pickle=True)
    return float(d["macro_acc"][0]), list(d["large_species"])


def bc_confusion(run_dir: Path, class_order: list[str]) -> np.ndarray | None:
    """Row-normalized BioCLIP zero-shot confusion aligned to class_order."""
    p = run_dir / "bioclip_results.npz"
    if not p.exists():
        return None
    from sklearn.metrics import confusion_matrix
    d = np.load(p, allow_pickle=True)
    y_true = list(d["y_true"])
    y_pred = list(d["y_pred"])
    cm = confusion_matrix(y_true, y_pred, labels=class_order).astype(float)
    row_sums = cm.sum(axis=1, keepdims=True)
    return np.where(row_sums > 0, cm / row_sums, 0)


def svm_mean_confusion(run_dir: Path, K: int, class_order: list[str]) -> np.ndarray | None:
    """Mirror of notebook 03 Block 9A: row-normalize each run's confusion at
    shots=K, then average over runs."""
    p = run_dir / "predictions.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    dK = df[df["shots"] == K]
    if dK.empty:
        return None
    n = len(class_order)
    acc = np.zeros((n, n), dtype=float)
    counts = np.zeros(n, dtype=int)
    for _, dKr in dK.groupby("run_id"):
        ct = pd.crosstab(pd.Series(dKr["species_true"], name="t"),
                         pd.Series(dKr["species_pred"], name="p"))
        ct = ct.reindex(index=class_order, columns=class_order, fill_value=0).to_numpy().astype(float)
        rs = ct.sum(axis=1, keepdims=True)
        nz = (rs[:, 0] > 0)
        m_norm = np.zeros_like(ct)
        m_norm[nz] = ct[nz] / rs[nz]
        acc[nz] += m_norm[nz]
        counts[nz] += 1
    for i in range(n):
        if counts[i] > 0:
            acc[i] /= counts[i]
    return acc


def abbrev_italic(name: str) -> str:
    """e.g. 'Dermacentor variabilis' -> r'$\\it{D.\\ variabilis}$'"""
    parts = name.split()
    if len(parts) < 2:
        return name
    return r'$\it{' + parts[0][0] + r'.\ ' + ' '.join(parts[1:]) + r'}$'


def plot_confusion_3up(mats: dict[str, np.ndarray | None],
                       class_order: list[str],
                       title_suffix: str,
                       macro_by_mode: dict[str, float],
                       counts_by_mode: dict[str, dict[str, int]] | None,
                       out_path: Path):
    """Draw 3 side-by-side row-normalized confusion matrices in MODES order."""
    import matplotlib.pyplot as plt
    n = len(class_order)
    fig, axes = plt.subplots(1, 3, figsize=(min(6 * 3, 16), max(5.5, 0.7 * n + 2)),
                             sharey=True)
    x_labels = [abbrev_italic(s) for s in class_order]
    last_im = None
    for ax_idx, (ax, mode) in enumerate(zip(axes, MODES)):
        cm = mats.get(mode)
        macro = macro_by_mode.get(mode)
        if cm is None:
            ax.set_title(f"{mode} — missing")
            ax.set_xticks([]); ax.set_yticks([])
            continue
        im = ax.imshow(cm, aspect="auto", cmap="Blues", vmin=0, vmax=1)
        last_im = im
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=10)
        ax.set_xlabel("Predicted")
        # Only the leftmost panel keeps y-tick labels (with N counts if available)
        if ax_idx == 0:
            if counts_by_mode is not None and mode in counts_by_mode:
                y_labels = [f"{abbrev_italic(s)}  (N={counts_by_mode[mode].get(s, 0)})"
                            for s in class_order]
            else:
                y_labels = x_labels
            ax.set_yticklabels(y_labels, fontsize=10)
            ax.set_ylabel("True")
        else:
            ax.tick_params(axis="y", labelleft=False)
        # Cell values
        for i in range(n):
            for j in range(n):
                v = cm[i, j]
                if v > 0.01:
                    ax.text(j, i, f"{v*100:.0f}", ha="center", va="center",
                            color="white" if v > 0.5 else "black", fontsize=9)
        macro_txt = f"  (macro {macro*100:.1f}%)" if macro is not None else ""
        ax.set_title(f"{mode}{macro_txt}", fontsize=12)
    fig.suptitle(title_suffix, fontsize=13, y=1.02)
    if last_im is not None:
        fig.colorbar(last_im, ax=axes, fraction=0.02, pad=0.02)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    out_dir = OUT_ROOT / datetime.now().strftime("%y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    svm_rows = []
    bc_rows = []
    resolved = {}

    for mode in MODES:
        svm_dir = latest_run(SVM_RUNS, mode)
        bc_dir = latest_run(BC_RUNS, mode)
        resolved[mode] = {"svm": svm_dir, "bioclip": bc_dir}

        if svm_dir is not None:
            df = load_svm_shot_summary(svm_dir)
            if df is not None:
                df = df.copy()
                df["mode"] = mode
                svm_rows.append(df)
        if bc_dir is not None:
            res = load_bc_macro(bc_dir)
            if res is not None:
                macro, species = res
                bc_rows.append({"mode": mode, "macro_acc": macro,
                                "n_species": len(species),
                                "run_dir": str(bc_dir)})

    print(f"Resolved run dirs:")
    for m, r in resolved.items():
        print(f"  {m:8s} SVM={r['svm'].name if r['svm'] else '<missing>'}  "
              f"BC={r['bioclip'].name if r['bioclip'] else '<missing>'}")

    if not svm_rows and not bc_rows:
        print("Nothing to compare yet. Run notebooks 02/03 in each mode first.")
        return

    # ---- write tables -------------------------------------------------------
    if svm_rows:
        svm_df = pd.concat(svm_rows, ignore_index=True)
        svm_df = svm_df[["mode", "shots", "mean_macro", "ci95_macro",
                         "mean_overall", "ci95_overall", "n_runs_macro"]]
        svm_df.to_csv(out_dir / "svm_summary.csv", index=False)
        print(f"\nSVM summary -> {out_dir/'svm_summary.csv'}")
        print(svm_df.to_string(index=False))
    else:
        svm_df = None

    if bc_rows:
        bc_df = pd.DataFrame(bc_rows)
        bc_df.to_csv(out_dir / "bioclip_summary.csv", index=False)
        print(f"\nBioCLIP summary -> {out_dir/'bioclip_summary.csv'}")
        print(bc_df.to_string(index=False))
    else:
        bc_df = None

    # ---- plots --------------------------------------------------------------
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    if svm_df is not None:
        fig, ax = plt.subplots(figsize=(9, 5.5))
        for mode in MODES:
            sub = svm_df[svm_df["mode"] == mode].sort_values("shots")
            if sub.empty:
                continue
            x = sub["shots"].to_numpy()
            y = sub["mean_macro"].to_numpy()
            ci = sub["ci95_macro"].to_numpy()
            ax.plot(x, y, marker="o", linewidth=2, color=MODE_COLORS[mode],
                    label=f"SVM ({mode})")
            ax.fill_between(x, y - ci, y + ci, color=MODE_COLORS[mode], alpha=0.15)
        if bc_df is not None:
            for _, r in bc_df.iterrows():
                m = r["mode"]
                ax.axhline(r["macro_acc"], color=MODE_COLORS[m], linestyle="--",
                           linewidth=1.2, alpha=0.7,
                           label=f"BioCLIP zero-shot ({m}) {r['macro_acc']*100:.1f}%")
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_xlabel("Shots per species")
        ax.set_ylabel("Macro accuracy")
        ax.set_title("Macro Accuracy vs. Shots — by view mode (mean ± 95% CI)")
        if svm_df is not None and not svm_df.empty:
            ax.set_xticks(sorted(svm_df["shots"].unique()))
        # Auto-fit y-range with padding so low 1-shot points (esp. ventral) aren't clipped
        all_lows = [svm_df["mean_macro"].min() - svm_df["ci95_macro"].max()]
        if bc_df is not None:
            all_lows.append(bc_df["macro_acc"].min())
        ymin = max(0.0, min(all_lows) - 0.05)
        ax.set_ylim(ymin, 1.0)
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(fontsize=9, loc="lower right")
        plt.tight_layout()
        plt.savefig(out_dir / "macro_curves.png", dpi=200)
        plt.close(fig)
        print(f"\nPlot -> {out_dir/'macro_curves.png'}")

        # Overall accuracy variant
        fig, ax = plt.subplots(figsize=(9, 5.5))
        for mode in MODES:
            sub = svm_df[svm_df["mode"] == mode].sort_values("shots")
            if sub.empty:
                continue
            x = sub["shots"].to_numpy()
            y = sub["mean_overall"].to_numpy()
            ci = sub["ci95_overall"].to_numpy()
            ax.plot(x, y, marker="s", linewidth=2, color=MODE_COLORS[mode],
                    label=f"SVM ({mode})")
            ax.fill_between(x, y - ci, y + ci, color=MODE_COLORS[mode], alpha=0.15)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_xlabel("Shots per species")
        ax.set_ylabel("Overall accuracy")
        ax.set_title("Overall Accuracy vs. Shots — by view mode (mean ± 95% CI)")
        ax.set_xticks(sorted(svm_df["shots"].unique()))
        # Auto-fit y-range with padding
        ymin = max(0.0, svm_df["mean_overall"].min() - svm_df["ci95_overall"].max() - 0.05)
        ax.set_ylim(ymin, 1.0)
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(fontsize=9, loc="lower right")
        plt.tight_layout()
        plt.savefig(out_dir / "overall_curves.png", dpi=200)
        plt.close(fig)
        print(f"Plot -> {out_dir/'overall_curves.png'}")

    # ---- Confusion matrix 3-ups --------------------------------------------
    # Need a common class order. Take it from whichever BioCLIP run has labels.
    class_order = None
    counts_by_mode = {}
    macro_bc_by_mode = {}
    for mode in MODES:
        run = resolved[mode]["bioclip"]
        if run is None: continue
        info = load_bc_macro(run)
        if info is None: continue
        macro, species = info
        if class_order is None:
            class_order = species
        # per-mode N from the npz counts array
        d = np.load(run / "bioclip_results.npz", allow_pickle=True)
        counts_by_mode[mode] = {sp: int(n) for sp, n in zip(d["large_species"], d["counts"])}
        macro_bc_by_mode[mode] = macro

    if class_order is not None:
        # BioCLIP confusion 3-up
        bc_mats = {m: bc_confusion(resolved[m]["bioclip"], class_order)
                   if resolved[m]["bioclip"] else None for m in MODES}
        plot_confusion_3up(
            bc_mats, class_order,
            title_suffix="BioCLIP 2 zero-shot — confusion (row-normalized)",
            macro_by_mode=macro_bc_by_mode,
            counts_by_mode=counts_by_mode,
            out_path=out_dir / "bioclip_confusion_3up.png",
        )
        print(f"Plot -> {out_dir/'bioclip_confusion_3up.png'}")

        # SVM 35-shot confusion 3-up
        max_k = 35
        if svm_df is not None and max_k in svm_df["shots"].unique():
            macro_svm_by_mode = {}
            for m in MODES:
                row = svm_df[(svm_df["mode"] == m) & (svm_df["shots"] == max_k)]
                if not row.empty:
                    macro_svm_by_mode[m] = float(row["mean_macro"].iloc[0])
            svm_mats = {m: svm_mean_confusion(resolved[m]["svm"], max_k, class_order)
                        if resolved[m]["svm"] else None for m in MODES}
            plot_confusion_3up(
                svm_mats, class_order,
                title_suffix=f"SVM {max_k}-shot — mean confusion across 100 runs (row-normalized)",
                macro_by_mode=macro_svm_by_mode,
                counts_by_mode=counts_by_mode,
                out_path=out_dir / f"svm_{max_k}shot_confusion_3up.png",
            )
            print(f"Plot -> {out_dir/f'svm_{max_k}shot_confusion_3up.png'}")

    # ---- readme -------------------------------------------------------------
    lines = ["# View-mode comparison",
             f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}",
             "",
             "## Resolved runs"]
    for m, r in resolved.items():
        lines.append(f"- **{m}** SVM: `{r['svm']}`  BioCLIP: `{r['bioclip']}`")
    if bc_df is not None:
        lines += ["", "## BioCLIP zero-shot macro accuracy"]
        for _, r in bc_df.iterrows():
            lines.append(f"- {r['mode']}: {r['macro_acc']*100:.2f}%  (n_species={int(r['n_species'])})")
    if svm_df is not None:
        lines += ["", "## SVM macro accuracy at max shots"]
        max_k = int(svm_df["shots"].max())
        for mode in MODES:
            row = svm_df[(svm_df["mode"] == mode) & (svm_df["shots"] == max_k)]
            if not row.empty:
                lines.append(f"- {mode} @ K={max_k}: "
                             f"{row['mean_macro'].iloc[0]*100:.2f}% "
                             f"± {row['ci95_macro'].iloc[0]*100:.2f}%")
    (out_dir / "README.md").write_text("\n".join(lines) + "\n")
    print(f"\nSummary -> {out_dir/'README.md'}")


if __name__ == "__main__":
    main()
