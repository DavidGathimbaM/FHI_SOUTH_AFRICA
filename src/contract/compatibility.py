import pandas as pd

ACCESS_RESILIENCE_SIGNALS = [
    "has_mobile_money", "has_internet_banking", "has_debit_card", "has_credit_card", "has_loan_account",
    "funeral_insurance", "medical_insurance", "motor_vehicle_insurance",
    "uses_informal_lender", "uses_friends_family_savings",
    "current_problem_cash_flow", "problem_sourcing_money",
]

BASIC_SIGNALS = ["owner_age", "business_age_years", "business_age_months"]
FINANCIAL_ACTIVITY = ["business_turnover", "business_expenses", "personal_income"]

def resolve_country_col(df: pd.DataFrame) -> str | None:
    for c in df.columns:
        if c.strip().lower() == "country":
            return c
    return None

def count_present_signals(df: pd.DataFrame, cols: list[str]) -> int:
    present = 0
    for c in cols:
        if c in df.columns:
            s = df[c]
            # signal counts if not almost entirely missing
            if s.isna().mean() < 0.95:
                present += 1
    return present

def grade_compatibility(df: pd.DataFrame) -> tuple[int, list[str]]:
    """
    Grade 1: enough signals for reliable scoring
    Grade 2: partial signals; scoring allowed with warnings
    Grade 3: not compatible
    """
    notes = []
    df = df.copy()

    country_col = resolve_country_col(df)
    if country_col is None:
        return 3, ["Missing required column: country. Cannot score."]

    notes.append(f"Detected country column: '{country_col}'")

    basics_present = count_present_signals(df, BASIC_SIGNALS)
    fin_present = count_present_signals(df, FINANCIAL_ACTIVITY)
    access_present = count_present_signals(df, ACCESS_RESILIENCE_SIGNALS)

    notes.append(f"Signal summary: basics={basics_present}, financial_activity={fin_present}, access_resilience={access_present}")

    if basics_present >= 1 and fin_present >= 1 and access_present >= 2:
        notes.append("Grade 1: Sufficient signals for reliable scoring.")
        return 1, notes

    if basics_present >= 1 and (fin_present >= 1 or access_present >= 2):
        notes.append("Grade 2: Partial signals; scoring allowed with warnings.")
        return 2, notes

    notes.append("Grade 3: Insufficient signals; cannot score reliably.")
    return 3, notes
