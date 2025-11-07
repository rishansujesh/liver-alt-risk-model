# 00_clean_dataset.py
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "nhanes_merged.csv"
OUTPUT = ROOT / "nhanes_cleaned.csv"

# ---- Columns to keep ----
core_features = [
    # IDs / demo / SES
    "SEQN", "RIAGENDR", "RIDAGEYR", "RIDRETH1", "DMDEDUC2", "INDFMPIR", "DMDMARTZ",
    # Alcohol behavior
    "ALQ111", "ALQ121", "ALQ130", "ALQ142", "ALQ170", "ALQ270",
    # Liver-related labs (as predictors, NOT the target)
    "LBXSASSI", "LBXSAPSI", "LBXSAL", "LBXSGB", "LBXSGTSI",
]

optional_features = [
    # Light extras for EDA/experiments
    "SDDSRVYR", "DMDBORN4", "RIDEXMON",
    "ALQ151", "ALQ280", "ALQ290",
    "LBXSCA", "LBXSCR", "LBXSGL", "LBXSIR",
    "LBXSCH", "LBXSTR", "LBXSUA",
]

target_col = "LBXSATSI"  # ALT (used to derive binary target)

def main():
    if not INPUT.exists():
        print(f"❌ Could not find input file: {INPUT}")
        sys.exit(1)

    df = pd.read_csv(INPUT, low_memory=False)

    # Check presence of required/optional columns
    requested = set(core_features + optional_features + [target_col])
    present = set(df.columns)
    missing = sorted(list(requested - present))
    if missing:
        print("⚠️  WARNING: These requested columns are missing and will be skipped:")
        for c in missing:
            print("   -", c)

    keep_cols = [c for c in (core_features + optional_features + [target_col]) if c in present]
    if target_col not in keep_cols:
        print(f"❌ Required target column '{target_col}' not found in the input. Aborting.")
        sys.exit(1)

    df_clean = df[keep_cols].copy()

    # Create binary target
    # ALT > 40 IU/L as elevated (sex-agnostic baseline)
    df_clean["elevated_alt"] = (pd.to_numeric(df_clean[target_col], errors="coerce") > 40).astype("Int64")

    # Optional: basic dtype coercion for known numeric fields
    numeric_like = [
        "RIDAGEYR", "INDFMPIR",
        "ALQ111", "ALQ121", "ALQ130", "ALQ142", "ALQ170", "ALQ270",
        "LBXSASSI", "LBXSAPSI", "LBXSAL", "LBXSGB", "LBXSGTSI", target_col,
        "ALQ151", "ALQ280", "ALQ290",
        "LBXSCA", "LBXSCR", "LBXSGL", "LBXSIR",
        "LBXSCH", "LBXSTR", "LBXSUA",
    ]
    for col in numeric_like:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

    # Save
    df_clean.to_csv(OUTPUT, index=False)

    # Report
    n_rows, n_cols = df_clean.shape
    kept = sorted(df_clean.columns)
    print("✅ Cleaned dataset saved.")
    print(f"   Path: {OUTPUT}")
    print(f"   Shape: {n_rows:,} rows × {n_cols} columns")
    print("   Columns kept (sorted):")
    for c in kept:
        print("    -", c)

    # Quick sanity: show target prevalence (excluding NA)
    if df_clean["elevated_alt"].notna().any():
        prev = df_clean["elevated_alt"].dropna().mean()
        print(f"   Elevated ALT prevalence (ALT>40): {prev:.3f}")

if __name__ == "__main__":
    main()
