import pandas as pd
import numpy as np

from .schema_mapper import map_schema
from .normalizer import normalize_categoricals
from .feature_derivation import derive_features
from .compatibility import grade_compatibility

def build_mapped_canonical_features(canonical_features_raw: list[str]) -> list[str]:
    """
    Converts raw canonical schema into post-mapping canonical schema.
    - Replaces 'ID' with 'business_id'
    - Removes 'ID' entirely
    - De-duplicates while preserving order
    """
    out = []
    for c in canonical_features_raw:
        if c == "ID":
            c = "business_id"
        if c not in out:
            out.append(c)
    # ensure 'ID' does not exist
    out = [c for c in out if c != "ID"]
    return out

def run_contract_engine(
    df: pd.DataFrame,
    canonical_features_raw: list[str],
    categorical_cols: list[str],
) -> dict:
    """
    Shared trunk:
    1) schema mapping (ID -> business_id)
    2) schema alignment against mapped canonical schema (never re-add ID)
    3) normalization
    4) derivation
    5) compatibility grading
    """
    notes_all: list[str] = []

    # 1) map schema
    df1, notes = map_schema(df)
    notes_all += notes

    # 2) align schema using mapped canonical features
    canonical_features = build_mapped_canonical_features(canonical_features_raw)

    for col in canonical_features:
        if col not in df1.columns:
            df1[col] = np.nan
            notes_all.append(f"Added missing column '{col}' as NaN (schema alignment).")

    # Safety: never allow ID to reappear
    if "ID" in df1.columns:
        df1 = df1.drop(columns=["ID"])
        notes_all.append("Dropped 'ID' column after alignment (deprecated).")

    # 3) normalize categoricals
    categorical_cols_mapped = map_cols_to_mapped_schema(categorical_cols)
    df2, notes = normalize_categoricals(df1, categorical_cols_mapped)
    notes_all += [f"Categorical cols mapped (raw->{len(categorical_cols)} to mapped->{len(categorical_cols_mapped)})."]
    notes_all += notes

    # 4) derive features
    df3, notes = derive_features(df2)
    notes_all += notes

    # 5) compatibility
    grade, gnotes = grade_compatibility(df3)
    notes_all += gnotes

    return {
        "df": df3,
        "compatibility_grade": grade,
        "compatibility_notes": notes_all,
        "canonical_features_mapped": canonical_features,
    }

def map_cols_to_mapped_schema(cols: list[str]) -> list[str]:
    """
    Maps raw column names to post-mapping names.
    - ID -> business_id
    - removes ID duplicates
    """
    out = []
    for c in cols:
        if c == "ID":
            c = "business_id"
        if c not in out and c != "ID":
            out.append(c)
    return out



