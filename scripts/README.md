# Scripts

## data_cleanup.py — Data Cleanup Pipeline

Reads the raw March CSV (`AI Image Data Mar 13 2026.csv`) and produces a cleaned, filtered dataset for downstream classification and embedding exploration.

### What it does

1. **Loads raw CSV** (995 records) and drops rows with missing Sample ID or Species
2. **Deduplicates** Sample IDs (keeps first occurrence) — removed 5 duplicates
3. **Normalizes species labels** — fixes typos (`variablis` → `variabilis`), non-breaking spaces in `Amblyomma americanum`, trailing whitespace
4. **Removes generic "Ixodes"** — 103 specimens with only genus-level ID (no species)
5. **Removes Ixodes cookei duplicates** — 22 specimens with Sample IDs ending in "a" (known duplicates per data provider notes)
6. **Removes broken specimens** — specimens flagged in error analysis (consistently misclassified)
7. **Annotates pathogen results**:
   - Sample ID starts with "T" → `not_tested` (ticks not tested for pathogens)
   - Blank pathogen result (non-T samples) → `negative`
   - Existing "Positive" values preserved
   - Stats: 212 not tested, 516 negative, 96 positive
8. **Filters by specimen count** — only species with ≥50 unique specimens kept
9. **Matches to images** — case-insensitive lookup for dorsal (`-01`) and ventral (`-02`) image pairs; specimens without both views excluded (11 specimens)

### Output files

All saved to `data/processed/` (originals archived in `data/processed/archive/`):

| File | Description |
|------|-------------|
| `final_data_cleaned.json` | 1442 image records (721 specimens × 2 views) |
| `class_names_cleaned.json` | 5 species names |
| `cleanup_summary.json` | Full pipeline stats and change log |

### Record format (final_data_cleaned.json)

```json
{
    "true_label": "Dermacentor variabilis",
    "sample_id": "100-01",
    "sex": "Female",
    "life_stage": "Adult",
    "attached": null,
    "pathogen": "none",
    "pathogen_result": "negative",
    "tick_condition": "Fed",
    "image_path": "/path/to/100-01-01.jpg",
    "view": "dorsal"
}
```

### Final dataset

| Species | Specimens |
|---------|-----------|
| Dermacentor variabilis | 318 |
| Ixodes scapularis | 174 |
| Amblyomma americanum | 124 |
| Haemaphysalis leporispalustris | 53 |
| Haemaphysalis longicornis | 52 |
| **Total** | **721** |

### Species removed (<50 specimens)

Ixodes cookei (30), Amblyomma maculatum (26), Ixodes dentatus (25), Ixodes kingi (6), Dermacentor andersoni (1), Ixodes banksi (1), Ixodes texanus (1), Ixodes brunneus (1), Ixodes muris (1)

### Usage

```bash
python scripts/data_cleanup.py
```

No arguments needed — reads paths from `config/paths.json`.
