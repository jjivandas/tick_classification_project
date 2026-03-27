"""
Data Cleanup Script for Tick Classification Project
=====================================================
Reads the raw March CSV, cleans and normalizes the data, applies filters,
and outputs cleaned JSON files for downstream use.

Follows the same data loading approach as notebooks/01_data_exploration.ipynb
but adds:
  - Ixodes cookei duplicate removal (Sample IDs ending in "a")
  - Pathogen annotation (T-prefix → not_tested, blank → negative)
  - Tick condition field (fed/unfed/engorged/undetermined)
  - Pathogen name field
  - Species filtering by minimum specimen count (≥50)
"""

import json
import os
import re
import sys
import unicodedata
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd

# ── Setup paths ──────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
from utils.paths import load_paths

paths = load_paths()
CSV_PATH = paths["metadata_csv"]
IMAGE_DIR = Path(paths["image_dir"])
PROCESSED_DIR = Path(paths["processed_dir"])

# ── Config ───────────────────────────────────────────────────────────────────
MIN_SPECIMENS = 50

# Output paths (new files — never overwrite originals)
OUT_DATA_JSON = PROCESSED_DIR / "final_data_cleaned.json"
OUT_CLASS_NAMES_JSON = PROCESSED_DIR / "class_names_cleaned.json"
OUT_SUMMARY_JSON = PROCESSED_DIR / "cleanup_summary.json"

# Known broken specimens from error analysis (same as notebook 01)
BROKEN_SPECIMEN_IDS = {
    "149-01", "ZOE-0021-01", "561-02", "560-01", "110-01", "224-01",
    "180-02", "419-01", "53-04", "503-01", "53-05", "33-02", "53-03",
    "53-02", "53-01", "290-01", "31-01",
}

# Known typo patterns
TYPO_FIXES = {
    re.compile(r"^\s*dermacentor\s+variablis\s*$", re.I): "Dermacentor variabilis",
}

# ── Change log ───────────────────────────────────────────────────────────────
CHANGE_LOG = []


def log_change(category, message):
    CHANGE_LOG.append({"category": category, "message": message})
    print(f"[{category}] {message}")


# ── Label cleaning (from notebook 01) ────────────────────────────────────────
def normalize_spaces(s):
    s = unicodedata.normalize("NFKC", s or "")
    s = "".join(" " if unicodedata.category(ch).startswith("Z") else ch for ch in s)
    s = "".join(ch for ch in s if unicodedata.category(ch) not in {"Cc", "Cf"})
    return re.sub(r"\s+", " ", s).strip()


def fix_label(raw):
    s = normalize_spaces(raw)
    for pattern, replacement in TYPO_FIXES.items():
        if pattern.match(s):
            return replacement
    parts = s.split()
    if len(parts) >= 2:
        return parts[0].capitalize() + " " + parts[1].lower()
    return s


def main():
    print("=" * 60)
    print("  TICK DATA CLEANUP PIPELINE")
    print("=" * 60)

    # ── Step 1: Load CSV ─────────────────────────────────────────────────
    df_raw = pd.read_csv(CSV_PATH)
    log_change("LOAD", f"Loaded {len(df_raw)} records from CSV")

    # Drop unnamed/empty columns
    df_raw = df_raw.loc[:, ~df_raw.columns.str.startswith("Unnamed")]

    # Drop rows with no Sample ID or Species
    df = df_raw.dropna(subset=["Sample ID", "Species of Tick"]).copy()
    log_change("CLEAN", f"Dropped {len(df_raw) - len(df)} rows with missing Sample ID or Species")

    # Strip whitespace from string columns
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

    # ── Step 2: Deduplicate Sample IDs (keep first) ─────────────────────
    n_before = len(df)
    df = df.drop_duplicates(subset=["Sample ID"], keep="first").copy()
    n_dedup = n_before - len(df)
    log_change("DEDUP", f"Removed {n_dedup} duplicate Sample IDs (kept first occurrence)")

    # ── Step 3: Clean species labels ─────────────────────────────────────
    df["true_label"] = df["Species of Tick"].apply(fix_label)

    label_changes = []
    for orig, clean in zip(df["Species of Tick"], df["true_label"]):
        if orig != clean:
            label_changes.append((orig, clean))
    if label_changes:
        log_change("LABELS", f"Cleaned {len(label_changes)} labels ({len(set(label_changes))} unique)")
        for orig, clean in sorted(set(label_changes)):
            print(f"    '{orig}' → '{clean}'")

    # ── Step 4: Remove generic "Ixodes" (no species-level ID) ────────────
    mask_ixodes = df["true_label"] == "Ixodes"
    n_generic = mask_ixodes.sum()
    df = df[~mask_ixodes].copy()
    log_change("FILTER", f"Removed {n_generic} generic 'Ixodes' specimens (no species-level ID)")

    # ── Step 5: Remove Ixodes cookei duplicates (Sample ID ending in 'a')
    mask_cookei_dup = (
        df["Sample ID"].str.endswith("a") & (df["true_label"] == "Ixodes cookei")
    )
    n_cookei_dupes = mask_cookei_dup.sum()
    cookei_dupe_ids = df.loc[mask_cookei_dup, "Sample ID"].tolist()
    df = df[~mask_cookei_dup].copy()
    log_change("FILTER", f"Removed {n_cookei_dupes} Ixodes cookei duplicates (Sample ID ending in 'a')")
    if cookei_dupe_ids:
        print(f"    IDs: {', '.join(cookei_dupe_ids)}")

    # ── Step 6: Remove broken specimens ──────────────────────────────────
    mask_broken = df["Sample ID"].isin(BROKEN_SPECIMEN_IDS)
    n_broken = mask_broken.sum()
    df = df[~mask_broken].copy()
    log_change("FILTER", f"Removed {n_broken} broken specimens (data quality issues)")

    # ── Step 7: Annotate pathogen results ────────────────────────────────
    pathogen_stats = {"not_tested": 0, "negative": 0, "positive": 0}

    def annotate_pathogen(row):
        sid = str(row["Sample ID"])
        result = row["Pathogen Result"]
        if sid.startswith("T"):
            pathogen_stats["not_tested"] += 1
            return "not_tested"
        elif pd.isna(result) or result == "":
            pathogen_stats["negative"] += 1
            return "negative"
        else:
            pathogen_stats["positive"] += 1
            return result

    df["pathogen_result_clean"] = df.apply(annotate_pathogen, axis=1)
    df["pathogen_clean"] = df["Pathogen"].fillna("none")
    log_change("ANNOTATE", f"Pathogen results: {pathogen_stats}")

    # ── Step 8: Species distribution before filtering ────────────────────
    species_counts_before = df["true_label"].value_counts()
    print("\nSpecies distribution (before min-specimen filter):")
    for sp, count in species_counts_before.items():
        marker = " ✓" if count >= MIN_SPECIMENS else " ✗"
        print(f"  {sp}: {count}{marker}")

    # ── Step 9: Filter by minimum specimen count ─────────────────────────
    keep_species = species_counts_before[species_counts_before >= MIN_SPECIMENS].index.tolist()
    removed_species = species_counts_before[species_counts_before < MIN_SPECIMENS]
    df = df[df["true_label"].isin(keep_species)].copy()
    log_change("FILTER", f"Kept {len(keep_species)} species with ≥{MIN_SPECIMENS} specimens, removed {len(removed_species)}")

    # ── Step 10: Cross-reference with images (same as notebook 01) ───────
    all_image_files = {
        f.upper(): f for f in os.listdir(IMAGE_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    }
    log_change("LOAD", f"Found {len(all_image_files)} image files in directory")

    final_data = []
    missing_images = []
    included_ids = []

    for _, row in df.iterrows():
        base_id = str(row["Sample ID"]).upper()
        dorsal_path, ventral_path = None, None

        # Try multiple filename patterns (case-insensitive)
        for pattern in [f"{base_id}-01.JPG", f"{base_id}-1.JPG"]:
            if pattern in all_image_files:
                dorsal_path = str(IMAGE_DIR / all_image_files[pattern])
                break

        for pattern in [f"{base_id}-02.JPG", f"{base_id}-2.JPG"]:
            if pattern in all_image_files:
                ventral_path = str(IMAGE_DIR / all_image_files[pattern])
                break

        # Only include specimens with BOTH dorsal and ventral
        if dorsal_path and ventral_path:
            base_record = {
                "true_label": row["true_label"],
                "sample_id": row["Sample ID"],
                "sex": row.get("Tick Sex1") if pd.notna(row.get("Tick Sex1")) else None,
                "life_stage": row.get("Life Stage") if pd.notna(row.get("Life Stage")) else None,
                "attached": row.get("Attached?") if pd.notna(row.get("Attached?")) else None,
                "pathogen": row["pathogen_clean"],
                "pathogen_result": row["pathogen_result_clean"],
                "tick_condition": row.get("Tick Condition") if pd.notna(row.get("Tick Condition")) else None,
            }
            final_data.append({**base_record, "image_path": dorsal_path, "view": "dorsal"})
            final_data.append({**base_record, "image_path": ventral_path, "view": "ventral"})
            included_ids.append(row["Sample ID"])
        else:
            missing_images.append(row["Sample ID"])

    n_specimens = len(included_ids)
    n_images = len(final_data)
    log_change("BUILD", f"Created {n_images} image entries ({n_specimens} specimens × 2 views)")

    if missing_images:
        log_change("EXCLUDE", f"Excluded {len(missing_images)} specimens with missing image pairs")

    # ── Step 11: Write outputs ───────────────────────────────────────────
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    with open(OUT_DATA_JSON, "w") as f:
        json.dump(final_data, f, indent=4)
    log_change("SAVE", f"Wrote {OUT_DATA_JSON.name}")

    class_names = sorted(set(r["true_label"] for r in final_data))
    with open(OUT_CLASS_NAMES_JSON, "w") as f:
        json.dump(class_names, f, indent=4)
    log_change("SAVE", f"Wrote {OUT_CLASS_NAMES_JSON.name} ({len(class_names)} species)")

    # Per-species counts in final dataset
    specimen_counts = Counter(r["true_label"] for r in final_data if r["view"] == "dorsal")

    # Build summary dict
    summary = {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source_csv": str(CSV_PATH),
        "min_specimens": MIN_SPECIMENS,
        "generic_ixodes_removed": int(n_generic),
        "cookei_duplicates_removed": int(n_cookei_dupes),
        "cookei_duplicate_ids": cookei_dupe_ids,
        "broken_specimens_removed": int(n_broken),
        "pathogen_stats": pathogen_stats,
        "species_kept": {sp: int(specimen_counts[sp]) for sp in sorted(specimen_counts)},
        "species_removed": {sp: int(c) for sp, c in removed_species.items()},
        "final_specimens": n_specimens,
        "final_images": n_images,
        "missing_image_specimens": [str(s) for s in missing_images],
        "change_log": CHANGE_LOG,
    }
    with open(OUT_SUMMARY_JSON, "w") as f:
        json.dump(summary, f, indent=4)
    log_change("SAVE", f"Wrote {OUT_SUMMARY_JSON.name}")

    # ── Final summary ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  FINAL CLEANED DATASET")
    print("=" * 60)
    for sp in sorted(specimen_counts.keys()):
        print(f"  {sp:35s} {specimen_counts[sp]:4d} specimens")
    print(f"  {'TOTAL':35s} {n_specimens:4d} specimens ({n_images} images)")
    print("=" * 60)


if __name__ == "__main__":
    main()
