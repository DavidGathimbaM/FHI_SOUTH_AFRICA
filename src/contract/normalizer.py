import pandas as pd
import re

MISSING_TOKENS = {
    "", " ", "  ", "n/a", "na", "none", "null", "nan",
    "refused", "prefer not to say", "not applicable"
}

def normalize_text_value(x) -> str:
    if pd.isna(x):
        return "missing"

    s = str(x).strip()
    s = s.replace("’", "'").replace("‘", "'")
    s = re.sub(r"\s+", " ", s).strip()

    if s.lower() in MISSING_TOKENS:
        return "missing"

    if s.lower() in ["don't know", "dont know", "don’t know"]:
        return "dont_know"

    return s

def normalize_categoricals(df: pd.DataFrame, categorical_cols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """
    Defensive: only normalize columns that exist in df.
    """
    df = df.copy()
    notes = []

    existing_cols = [c for c in categorical_cols if c in df.columns]
    missing_cols = [c for c in categorical_cols if c not in df.columns]

    for c in existing_cols:
        df[c] = df[c].astype("object").map(normalize_text_value)

    notes.append(f"Normalized {len(existing_cols)} categorical columns.")
    if missing_cols:
        notes.append(f"Skipped {len(missing_cols)} categorical columns not present after mapping/alignment: {missing_cols[:10]}")

    return df, notes
