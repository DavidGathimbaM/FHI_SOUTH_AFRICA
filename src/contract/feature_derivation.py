import pandas as pd
import numpy as np

def derive_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Derives business_age_years/months where possible.
    Only safe math; no guessing of money units.
    """
    df = df.copy()
    notes = []

    if "business_age_months" in df.columns and "business_age_years" in df.columns:
        months_numeric = pd.to_numeric(df["business_age_months"], errors="coerce")
        years_numeric  = pd.to_numeric(df["business_age_years"], errors="coerce")

        months_missing = df["business_age_months"].isna() | (df["business_age_months"].astype(str) == "missing")
        years_missing  = df["business_age_years"].isna() | (df["business_age_years"].astype(str) == "missing")

        if months_missing.any():
            df.loc[months_missing, "business_age_months"] = (years_numeric[months_missing] * 12).round(0)
            notes.append("Derived business_age_months from business_age_years where missing.")

        if years_missing.any():
            df.loc[years_missing, "business_age_years"] = np.floor(months_numeric[years_missing] / 12)
            notes.append("Derived business_age_years from business_age_months where missing.")

    return df, notes
