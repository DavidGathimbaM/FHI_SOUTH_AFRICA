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
    s = s.replace("’", "'").replace("‘", "'")   # normalize curly apostrophes
    s = re.sub(r"\s+", " ", s).strip()          # collapse whitespace

    if s.lower() in MISSING_TOKENS:
        return "missing"

    if s.lower() in ["don't know", "dont know", "don’t know"]:
        return "dont_know"

    return s

def normalize_categoricals(df: pd.DataFrame, categorical_cols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    df = df.copy()
    notes = []
    for c in categorical_cols:
        df[c] = df[c].astype("object").map(normalize_text_value)
    notes.append(f"Normalized {len(categorical_cols)} categorical columns.")
    return df, notes
