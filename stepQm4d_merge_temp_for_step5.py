# ============================================================
# stepQm4d_merge_temp_for_step5.py
# Step 4d: Merge temperature channels into Step 4c CSV for Step 5 training
#
# Purpose:
#   - Load Step 4c bound-check CSV (contains Qm_slow target and physics columns)
#   - Load raw/temp CSV that contains temperature channels
#   - Merge selected temperature channels into Step 4c timeline
#   - Exclude Flange_temp (sensor issue)
#   - Save a new CSV for Step 5 feature learning
#
# Input:
#   processed_Qm/stepQm4c_boundcheck/data/<BASE>_qm_required_boundcheck.csv
#   raw or processed temp CSV with columns such as:
#       Time / t_sec, PZT, Room_temp, Tool_temp, Flange_temp
#
# Output:
#   processed_Qm/stepQm4d_merge_temp/data/<BASE>_qm_required_with_temp.csv
#
# Notes:
#   - This step does NOT change target Qm_slow
#   - It only augments feature columns for Step 5
#   - Flange_temp is intentionally excluded
# ============================================================

import os
import numpy as np
import pandas as pd

# ================= USER CONFIG =================
ROOT = r"C:\Users\ploy\Desktop\ML\GRU_2"
BASE = "2.5nl-fix17.37"

# Step 4c output
STEP4C_CSV = os.path.join(
    ROOT, "processed_Qm", "stepQm4c_boundcheck", "data",
    f"{BASE}_qm_required_boundcheck.csv"
)

# Raw / external CSV that contains temp channels
# >>> แก้ path นี้ให้ตรงกับไฟล์ temp จริงของเธอ <<<
TEMP_CSV = r"C:\Users\ploy\Desktop\ML\GRU_2\raw_fixed_old\2.5nl-fix17.37.csv"

OUT_DIR = os.path.join(ROOT, "processed_Qm", "stepQm4d_merge_temp")
DATA_DIR = os.path.join(OUT_DIR, "data")

# merge tolerance in seconds for nearest match
MERGE_TOL_SEC = 0.01   # 10 ms
# =================================================

os.makedirs(DATA_DIR, exist_ok=True)


def pick_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def parse_clock_time_to_seconds(series):
    """
    Convert a datetime-like or clock-like time column into relative seconds
    from the first sample.

    Supports examples like:
      2025/10/28 18:22:20.7
      2025-10-28 18:22:20.700
      22:20.7
      01:02:03.5
    """
    s = series.astype(str).str.strip()

    # try full datetime parsing first
    dt = pd.to_datetime(s, errors="coerce")

    if dt.notna().all():
        sec = (dt - dt.iloc[0]).dt.total_seconds().to_numpy(dtype=np.float64)
        return sec

    # fallback: parse clock-like strings manually
    vals = []
    for x in s:
        parts = x.split(":")
        if len(parts) == 2:
            mm = float(parts[0])
            ss = float(parts[1])
            total = mm * 60.0 + ss
        elif len(parts) == 3:
            hh = float(parts[0])
            mm = float(parts[1])
            ss = float(parts[2])
            total = hh * 3600.0 + mm * 60.0 + ss
        else:
            raise ValueError(f"Unsupported Time format: {x}")

        vals.append(total)

    vals = np.asarray(vals, dtype=np.float64)
    vals = vals - vals[0]
    return vals


def build_temp_time_axis(df_temp):
    """
    Priority:
      1) use t_sec directly if present
      2) parse Time column if present
    """
    tcol_direct = pick_col(df_temp, ["t_sec", "t", "time_sec"])
    if tcol_direct is not None:
        t = df_temp[tcol_direct].to_numpy(dtype=np.float64)
        t = t - t[0]
        return t, tcol_direct

    tcol_clock = pick_col(df_temp, ["Time", "time", "TIME"])
    if tcol_clock is not None:
        t = parse_clock_time_to_seconds(df_temp[tcol_clock])
        return t, tcol_clock

    raise KeyError("Cannot find time column in TEMP_CSV. Expected one of: t_sec / t / Time")


def main():
    if not os.path.exists(STEP4C_CSV):
        raise FileNotFoundError(f"Missing Step4c CSV:\n{STEP4C_CSV}")
    if not os.path.exists(TEMP_CSV):
        raise FileNotFoundError(f"Missing TEMP CSV:\n{TEMP_CSV}")

    df4 = pd.read_csv(STEP4C_CSV)
    dft = pd.read_csv(TEMP_CSV)

    # ---------------- Step4c time ----------------
    t4_col = pick_col(df4, ["t_sec", "t", "Time"])
    if t4_col is None:
        raise KeyError("Cannot find time column in Step4c CSV")
    t4 = df4[t4_col].to_numpy(dtype=np.float64)
    t4 = t4 - t4[0]

    # ---------------- temp time ----------------
    tt, tt_col = build_temp_time_axis(dft)

    # ---------------- select temp columns ----------------
    # Exclude Flange_temp intentionally
    pzt_col   = pick_col(dft, ["PZT", "Pzt", "pzt"])
    room_col  = pick_col(dft, ["Room_temp", "room_temp", "RoomTemp", "room"])
    tool_col  = pick_col(dft, ["Tool_temp", "tool_temp", "ToolTemp", "tool"])
    flange_col = pick_col(dft, ["Flange_temp", "flange_temp", "FlangeTemp", "flange"])

    selected = {}
    if pzt_col is not None:
        selected["PZT"] = dft[pzt_col].to_numpy(dtype=np.float64)
    if room_col is not None:
        selected["Room_temp"] = dft[room_col].to_numpy(dtype=np.float64)
    if tool_col is not None:
        selected["Tool_temp"] = dft[tool_col].to_numpy(dtype=np.float64)

    if len(selected) == 0:
        raise KeyError("No usable temp columns found. Need at least one of: PZT, Room_temp, Tool_temp")

    print("\n[INFO] Temp columns found:")
    print("  time column   :", tt_col)
    print("  PZT           :", pzt_col)
    print("  Room_temp     :", room_col)
    print("  Tool_temp     :", tool_col)
    print("  Flange_temp   :", flange_col, "(excluded)" if flange_col is not None else "(not found)")

    # ---------------- prepare merge_asof ----------------
    left = pd.DataFrame({"t_sec_merge": t4})
    right = pd.DataFrame({"t_sec_merge": tt})

    for k, v in selected.items():
        right[k] = v

    left = left.sort_values("t_sec_merge").reset_index(drop=True)
    right = right.sort_values("t_sec_merge").reset_index(drop=True)

    merged = pd.merge_asof(
        left,
        right,
        on="t_sec_merge",
        direction="nearest",
        tolerance=MERGE_TOL_SEC
    )

    # ---------------- attach to step4c ----------------
    out = df4.copy()
    for k in selected.keys():
        out[k] = merged[k]

    # optional quality check
    nan_counts = {k: int(out[k].isna().sum()) for k in selected.keys()}

    out_csv = os.path.join(DATA_DIR, f"{BASE}_qm_required_with_temp.csv")
    out.to_csv(out_csv, index=False)

    print("\n================ StepQm4d DONE: merge temp for Step 5 ================")
    print("Step4c CSV   :", STEP4C_CSV)
    print("Temp CSV     :", TEMP_CSV)
    print("Saved CSV    :", out_csv)
    print("Merge tol(s) :", MERGE_TOL_SEC)
    print("Rows         :", len(out))
    print("NaN counts after merge:")
    for k, v in nan_counts.items():
        print(f"  {k}: {v}")
    print("Included temp columns:", list(selected.keys()))
    print("Excluded column: Flange_temp")
    print("======================================================================")

if __name__ == "__main__":
    main()