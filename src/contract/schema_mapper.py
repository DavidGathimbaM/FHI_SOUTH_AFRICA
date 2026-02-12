import pandas as pd

def map_schema(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Standardizes identifier column naming:
    - If ID exists, rename to business_id and drop original ID (by rename)
    - If business_id exists, keep
    - If neither exists, generate business_id
    Ensures no identifier columns leak into features later.
    """
    notes = []
    df = df.copy()

    if "business_id" in df.columns:
        notes.append("business_id already present.")
    elif "ID" in df.columns:
        df = df.rename(columns={"ID": "business_id"})
        notes.append("Renamed ID -> business_id.")
    else:
        df["business_id"] = [f"auto_{i:06d}" for i in range(1, len(df) + 1)]
        notes.append("No ID/business_id found. Generated business_id.")

    # Safety: if an ID column still exists (e.g., user uploaded both), drop it
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])
        notes.append("Dropped leftover ID column to prevent feature leakage.")

    return df, notes
