# ============================================================
# tdms_to_csv_exp3_fix_full.py
# Export full TDMS -> CSV for EXP3 fix file
# ============================================================

import os
import numpy as np
import pandas as pd
from nptdms import TdmsFile

# ================= USER CONFIG =================
TDMS_PATH = r"C:\Users\ploy\Desktop\UTHwithT\data\1_tb_fix18.3_exp6.tdms"
OUT_CSV   = r"E:\raw_data\exp6(2026.4.2)\1_tb_fix18.3_exp6.csv"

GROUP_NAME = "Log"
FS = 100_000.0
# =================================================

print("Reading TDMS:", TDMS_PATH)
tdms = TdmsFile.read(TDMS_PATH)

group = tdms[GROUP_NAME]

current      = group["Current"][:]
displacement = group["Displacement"][:]
flange_temp  = group["Flange_temp"][:]
pzt          = group["PZT"][:]
room_temp    = group["Room_temp"][:]
tool_temp    = group["Tool_temp"][:]
voltage      = group["Voltage"][:]

N = len(current)
print("Samples =", N)
print("Duration (sec) =", N / FS)

# build time from sample index
time_sec = np.arange(N, dtype=np.float64) / FS

df = pd.DataFrame({
    "Time": time_sec,
    "Current": current,
    "Displacement": displacement,
    "Flange_temp": flange_temp,
    "PZT": pzt,
    "Room_temp": room_temp,
    "Tool_temp": tool_temp,
    "Voltage": voltage,
})

print("Saving CSV:", OUT_CSV)
df.to_csv(OUT_CSV, index=False)

print("Done.")
print("Saved shape:", df.shape)
print("Head:")
print(df.head(3))
print("Tail:")
print(df.tail(3))