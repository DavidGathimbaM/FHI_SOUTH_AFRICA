import pandas as pd
import numpy as np

from .schema_mapper import map_schema
from .normalizer import normalize_categoricals
from .feature_derivation import derive_features
from .compatibility import grade_compatibility

def run_contract_engine(
    df: pd.DataFrame,
    canonical_features: list[str],
    categorical_cols: list[str],
) -> dict:
    """
    Shared trunk:
    1) schema mapping
    2) schema alignment (add missing canonical cols)
    3) normalization
    4) derivation
    5) compatibility grading
    """
    notes_all: list[str] = []

    df1, notes = map_schema(df)
    notes_all += notes

    # Align schema: add any missing canonical feature columns as NaN
    for col in canonical_features:
        if col not in df1.columns:
            df1[col] = np.nan
            notes_all.append(f"Added missing column '{col}' as NaN (schema alignment).")

    df2, notes = normalize_categoricals(df1, categorical_cols)
    notes_all += notes

    df3, notes = derive_features(df2)
    notes_all += notes

    grade, gnotes = grade_compatibility(df3)
    notes_all += gnotes

    return {
        "df": df3,
        "compatibility_grade": grade,
        "compatibility_notes": notes_all,
    }
