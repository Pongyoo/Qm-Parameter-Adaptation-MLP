# ============================================================
# stepQm0_exp3_clean_fix_full.py
# Clean full CSV (exported from TDMS) -> clean CSV for pipeline
# ============================================================

import os
import pandas as pd

# ================= USER CONFIG =================
#INPUT_CSV = r"C:\Users\ploy\Desktop\raw_data\exp6(2026.4.2)\1.5_tb_fix18.1_str_exp6.csv"
INPUT_CSV = r"E:\raw_data\exp1(2025.10.28)\3nl-fix17.37.csv"


#OUT_DIR = r"C:\Users\ploy\Desktop\ML\GRU_2\processed_Qm_sessions\exp6.1_2.5_tb_fix17.13_in\stepQm0_clean"
OUT_DIR = r"E:\processed_Qm\exp1(2025.10.28)\3nl-fix17.37\stepQm0_clean"
os.makedirs(OUT_DIR, exist_ok=True)

OUT_CSV = os.path.join(OUT_DIR, "3nl-fix17.37_exp1_clean.csv")

USE_COLS = ["Time", "Current", "Displacement", "PZT", "Tool_temp", "Voltage"]
# ============================================================

def to_num(s):
    return pd.to_numeric(s, errors="coerce")

print("\n================ StepQm0 EXP3: Clean FIX FULL =================")

print("INPUT:", INPUT_CSV)
df = pd.read_csv(INPUT_CSV)

print("Loaded shape:", df.shape)
print("Columns:", df.columns.tolist())

# --- select columns ---
for c in USE_COLS:
    if c not in df.columns:
        raise ValueError(f"Missing column: {c}")

out = pd.DataFrame()
out["Time"]         = to_num(df["Time"])
out["Current"]      = to_num(df["Current"])
out["Displacement"] = to_num(df["Displacement"])
out["PZT"]          = to_num(df["PZT"])
out["Tool_temp"]    = to_num(df["Tool_temp"])
out["V_level"]      = to_num(df["Voltage"])

# --- debug NaN ---
nan_ratio = out.isna().mean()
print("NaN ratio:", {k: float(v) for k, v in nan_ratio.items()})

# --- drop bad rows ---
MUST = ["Displacement", "Current", "PZT", "Tool_temp", "V_level"]

before = len(out)
out = out.dropna(subset=MUST).reset_index(drop=True)
after = len(out)

print(f"Rows kept: {after}/{before}  (dropped {before-after})")

# --- sanity check ---
print("\nSanity check:")
print("Time min/max:", float(out["Time"].min()), float(out["Time"].max()))
print("Displacement min/max:", float(out["Displacement"].min()), float(out["Displacement"].max()))
print("Current min/max:", float(out["Current"].min()), float(out["Current"].max()))

# --- save ---
out.to_csv(OUT_CSV, index=False)

print("\nSaved clean file:")
print(OUT_CSV)
print("Final shape:", out.shape)

print("\nDone StepQm0 EXP3")