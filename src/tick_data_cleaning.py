import re
import unicodedata
from typing import Optional

import pandas as pd


def normalize_spaces(s: Optional[str]) -> str:
    """
    Normalize unicode spaces to ASCII spaces, remove control/format characters,
    collapse whitespace, and trim. Safe for None.
    """
    s = unicodedata.normalize("NFKC", s or "")
    # Turn ANY Unicode space (including \u00A0 NBSP) into ASCII space
    s = "".join(" " if unicodedata.category(ch).startswith("Z") else ch for ch in s)
    # Remove control/format chars (zero-width joiner, LRM/RLM, etc.)
    s = "".join(ch for ch in s if unicodedata.category(ch) not in {"Cc", "Cf"})
    # Collapse runs of whitespace and trim
    return re.sub(r"\s+", " ", s).strip()


def fix_label_minimal(raw: str) -> str:
    """
    Minimal label cleanup used in notebooks:
    - Normalize spaces and remove stray unicode controls
    - Fix exact typo: "Dermacentor variablis" -> "Dermacentor variabilis"
    - Standardize binomial casing: Genus species
    """
    s = normalize_spaces(raw)
    # Exact typo you mentioned
    if s.lower() == "dermacentor variablis":
        return "Dermacentor variabilis"
    # Standardize simple binomial casing
    parts = s.split()
    if len(parts) >= 2:
        return parts[0].capitalize() + " " + parts[1].lower()
    return s


def load_and_clean_csv(csv_path: str, species_col: str = "Species of Tick") -> pd.DataFrame:
    """
    Convenience loader that reads a CSV and creates a `true_label` column
    by applying `fix_label_minimal` to the species column.
    Includes the simple sanity checks you requested.
    """
    df = pd.read_csv(csv_path)
    df["true_label"] = df[species_col].astype(str).apply(fix_label_minimal)

    # (optional) quick sanity checks
    assert not df["true_label"].str.contains("\u00A0", regex=False).any(), "NBSP still present"
    assert (df["true_label"].str.lower() == "dermacentor variablis").sum() == 0, "typo still present"
    return df


__all__ = [
    "normalize_spaces",
    "fix_label_minimal",
    "load_and_clean_csv",
]

