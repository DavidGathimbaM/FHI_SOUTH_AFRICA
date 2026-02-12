import pandas as pd

def map_schema(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Standardizes identifier column naming:
    - If ID exists, rename to business_id
    - If business_id exists, keep
    - If neither exists, generate business_id
    """

    notes = []
    df = df.copy()

    if "business_id" in df.columns:
        notes.append("business_id already present.")
    elif "ID" in df.columns:
        df = df.rename(columns={"ID": "business_id"})
        notes.append("Renamed ID -> business_id.")
    else:
        df["business_id"] = [f"auto_{i:60d}" for i in range(1, len(df) + 1)]
        notes.append("No business_id found. Generated business_id as auto_000001...")

    return df, notes