# ============================================================
# ALL-IN-ONE: CLEAN + PROCESS + PLOT
# สำหรับไฟล์ที่มีแค่ Time, Displacement
# Time เป็น datetime เช่น 4/2/2026 3:25:07
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# ================= USER CONFIG =================
INPUT_CSV = r"C:\Users\ploy\Desktop\raw_data\exp6(2026.4.2)\ref.csv"

OUT_DIR = r"C:\Users\ploy\Desktop\ML\GRU_2\processed_Qm_sessions\exp6_ref\stepQm0_clean"
os.makedirs(OUT_DIR, exist_ok=True)

OUT_CLEAN_CSV = os.path.join(OUT_DIR, "clean_time_disp.csv")
OUT_AMEAS_CSV = os.path.join(OUT_DIR, "A_meas_processed.csv")

TIME_COL = "Time"
DISP_COL = "Displacement"

# ---- processing config ----
FS_FALLBACK = 100000      # ถ้า infer sampling rate ไม่ได้ จะใช้ค่านี้
BPF_LO = 150           # bandpass low cutoff (Hz)
BPF_HI = 170           # bandpass high cutoff (Hz)
FILTER_ORDER = 4
RMS_WIN_MS = 50           # rolling RMS window (ms)
DS_FACTOR = 100           # downsample factor

# ---- plot config ----
ZOOM_RAW_SEC = 5
ZOOM_BP_SEC = 0.02
ZOOM_AMEAS_SEC = 5
# ============================================================


def butter_bandpass(x, fs, lo, hi, order=4):
    nyq = 0.5 * fs
    lo_n = lo / nyq
    hi_n = hi / nyq

    if not (0 < lo_n < hi_n < 1):
        raise ValueError(
            f"Bandpass cutoff invalid: fs={fs}, lo={lo}, hi={hi}, "
            f"normalized=({lo_n:.6f}, {hi_n:.6f})"
        )

    b, a = butter(order, [lo_n, hi_n], btype="bandpass")
    return filtfilt(b, a, x)


def rolling_rms_valid(x, win):
    x = np.asarray(x, dtype=np.float64)
    x2 = x * x
    cs = np.cumsum(np.insert(x2, 0, 0.0))
    y = np.sqrt((cs[win:] - cs[:-win]) / win)
    return y


def infer_fs_from_time(t_sec):
    t_sec = np.asarray(t_sec, dtype=np.float64)

    if len(t_sec) < 3:
        return None

    dt = np.diff(t_sec)
    dt = dt[np.isfinite(dt)]
    dt = dt[dt > 0]

    if len(dt) == 0:
        return None

    dt_med = np.median(dt)
    if dt_med <= 0:
        return None

    return 1.0 / dt_med


print("\n================ ALL-IN-ONE CLEAN + PROCESS + PLOT =================")
print("INPUT:", INPUT_CSV)

# --------------------------------------------------
# 1) LOAD
# --------------------------------------------------
df = pd.read_csv(INPUT_CSV)
print("Loaded shape:", df.shape)
print("Columns:", df.columns.tolist())

if TIME_COL not in df.columns:
    raise ValueError(f"Missing column: {TIME_COL}")
if DISP_COL not in df.columns:
    raise ValueError(f"Missing column: {DISP_COL}")

# --------------------------------------------------
# 2) PARSE
# --------------------------------------------------
time_dt = pd.to_datetime(df[TIME_COL], errors="coerce")
disp = pd.to_numeric(df[DISP_COL], errors="coerce")

tmp = pd.DataFrame({
    "Datetime": time_dt,
    "Displacement": disp
})

tmp["ok"] = tmp["Datetime"].notna() & tmp["Displacement"].notna()

print("\nNaN ratio:")
print({
    "Datetime": float(tmp["Datetime"].isna().mean()),
    "Displacement": float(tmp["Displacement"].isna().mean())
})

# --------------------------------------------------
# 3) RAW PLOT WITH BAD ROWS
# --------------------------------------------------
tmp_plot = tmp.copy()

# สำหรับ plot raw ต้องมี time sec ชั่วคราว
if tmp_plot["Datetime"].notna().any():
    first_valid_dt = tmp_plot.loc[tmp_plot["Datetime"].notna(), "Datetime"].iloc[0]
    tmp_plot["Time"] = (tmp_plot["Datetime"] - first_valid_dt).dt.total_seconds()
else:
    tmp_plot["Time"] = np.nan

plt.figure(figsize=(14, 5))
good = tmp_plot["ok"]
bad = ~good

if good.any():
    plt.plot(
        tmp_plot.loc[good, "Time"],
        tmp_plot.loc[good, "Displacement"],
        linewidth=1.0,
        label="valid rows"
    )

if bad.any():
    plt.scatter(
        tmp_plot.loc[bad, "Time"],
        tmp_plot.loc[bad, "Displacement"],
        s=10,
        label="bad rows"
    )

plt.title("Raw data (with bad rows)")
plt.xlabel("Elapsed time (s)")
plt.ylabel("Displacement")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "plot_01_raw_with_bad_rows.png"), dpi=150)
plt.close()

# --------------------------------------------------
# 4) CLEAN
# --------------------------------------------------
before = len(tmp)

clean = tmp.loc[tmp["ok"], ["Datetime", "Displacement"]].copy()
clean = clean.sort_values("Datetime").reset_index(drop=True)

# ลบเวลาซ้ำ / เวลาย้อน
dt_diff = clean["Datetime"].diff().dt.total_seconds()
keep = dt_diff.isna() | (dt_diff > 0)
clean = clean.loc[keep].reset_index(drop=True)

after = len(clean)
print(f"\nRows kept: {after}/{before} (dropped {before - after})")

if len(clean) == 0:
    raise RuntimeError("ไม่มีข้อมูลเหลือหลัง clean")

# reset เวลาให้เริ่มจาก 0
clean["Time"] = (clean["Datetime"] - clean["Datetime"].iloc[0]).dt.total_seconds()

# save clean csv
clean_out = clean[["Datetime", "Time", "Displacement"]].copy()
clean_out.to_csv(OUT_CLEAN_CSV, index=False)

# --------------------------------------------------
# 5) SANITY CHECK + FS
# --------------------------------------------------
t = clean["Time"].to_numpy(dtype=np.float64)
disp_raw = clean["Displacement"].to_numpy(dtype=np.float64)

fs_est = infer_fs_from_time(t)
if fs_est is None:
    fs_used = FS_FALLBACK
    print(f"\nInfer FS ไม่ได้ -> ใช้ fallback FS = {FS_FALLBACK}")
else:
    fs_used = fs_est
    print(f"\nEstimated FS = {fs_used:.6f} Hz")

print("\nSanity check:")
print("Datetime min/max:", clean["Datetime"].min(), clean["Datetime"].max())
print("Time min/max:", float(np.min(t)), float(np.max(t)))
print("Displacement min/max:", float(np.min(disp_raw)), float(np.max(disp_raw)))
print("Displacement mean/std:", float(np.mean(disp_raw)), float(np.std(disp_raw)))

# --------------------------------------------------
# 6) PLOTS AFTER CLEAN
# --------------------------------------------------
# full clean
plt.figure(figsize=(14, 5))
plt.plot(t, disp_raw, linewidth=1.0)
plt.title("Clean displacement (full)")
plt.xlabel("Elapsed time (s)")
plt.ylabel("Displacement")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "plot_02_clean_full.png"), dpi=150)
plt.close()

# zoom clean
m_raw = t <= ZOOM_RAW_SEC
plt.figure(figsize=(14, 5))
plt.plot(t[m_raw], disp_raw[m_raw], linewidth=1.0)
plt.title(f"Clean displacement (first {ZOOM_RAW_SEC} s)")
plt.xlabel("Elapsed time (s)")
plt.ylabel("Displacement")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "plot_03_clean_zoom.png"), dpi=150)
plt.close()

# detrend
disp_det = disp_raw - np.mean(disp_raw)

plt.figure(figsize=(14, 5))
plt.plot(t, disp_det, linewidth=1.0)
plt.title("Detrended displacement (full)")
plt.xlabel("Elapsed time (s)")
plt.ylabel("Detrended displacement")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "plot_04_detrended_full.png"), dpi=150)
plt.close()

plt.figure(figsize=(14, 5))
plt.plot(t[m_raw], disp_det[m_raw], linewidth=1.0)
plt.title(f"Detrended displacement (first {ZOOM_RAW_SEC} s)")
plt.xlabel("Elapsed time (s)")
plt.ylabel("Detrended displacement")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "plot_05_detrended_zoom.png"), dpi=150)
plt.close()

# --------------------------------------------------
# 7) BANDPASS
# --------------------------------------------------
# หมายเหตุ:
# ถ้า time จากไฟล์ละเอียดไม่พอ fs_est อาจต่ำมาก
# แต่ถ้าอยากใช้ logic เดิมแบบงานเก่า ก็จะไปใช้ FS_FALLBACK แทน
print("\nBandpass processing...")
disp_bp = butter_bandpass(disp_det, fs_used, BPF_LO, BPF_HI, order=FILTER_ORDER)

bp_stats = {
    "bp_mean": float(np.mean(disp_bp)),
    "bp_std": float(np.std(disp_bp)),
    "bp_min": float(np.min(disp_bp)),
    "bp_max": float(np.max(disp_bp)),
}
print("Bandpassed stats:", bp_stats)

# plot bandpass full
plt.figure(figsize=(14, 5))
plt.plot(t, disp_bp, linewidth=1.0)
plt.title("Bandpassed displacement (full)")
plt.xlabel("Elapsed time (s)")
plt.ylabel("Bandpassed displacement")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "plot_06_bandpassed_full.png"), dpi=150)
plt.close()

# plot bandpass zoom
m_bp = t <= ZOOM_BP_SEC
plt.figure(figsize=(14, 5))
plt.plot(t[m_bp], disp_bp[m_bp], linewidth=1.0)
plt.title(f"Bandpassed displacement (first {ZOOM_BP_SEC} s)")
plt.xlabel("Elapsed time (s)")
plt.ylabel("Bandpassed displacement")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "plot_07_bandpassed_zoom.png"), dpi=150)
plt.close()

# --------------------------------------------------
# 8) A_meas = rolling RMS + downsample
# --------------------------------------------------
print("\nA_meas processing...")
win = max(1, int(round(RMS_WIN_MS * 1e-3 * fs_used)))
print("RMS window samples:", win)

if win >= len(disp_bp):
    raise RuntimeError(
        f"RMS window ใหญ่เกินข้อมูล: win={win}, len={len(disp_bp)}. "
        f"ลองลด RMS_WIN_MS หรือเช็ก FS."
    )

A_env = rolling_rms_valid(disp_bp, win)
t_env = t[win - 1:]

A_ds = A_env[::DS_FACTOR]
t_ds = t_env[::DS_FACTOR]

ameas_df = pd.DataFrame({
    "Time": t_ds,
    "A_meas": A_ds
})
ameas_df.to_csv(OUT_AMEAS_CSV, index=False)

ameas_stats = {
    "A_meas_mean": float(np.mean(A_ds)),
    "A_meas_std": float(np.std(A_ds)),
    "A_meas_min": float(np.min(A_ds)),
    "A_meas_max": float(np.max(A_ds)),
}
print("A_meas stats:", ameas_stats)

# plot A_meas full
plt.figure(figsize=(14, 5))
plt.plot(t_ds, A_ds, linewidth=1.0)
plt.title("A_meas (full)")
plt.xlabel("Elapsed time (s)")
plt.ylabel("A_meas")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "plot_08_A_meas_full.png"), dpi=150)
plt.close()

# plot A_meas zoom
m_am = t_ds <= ZOOM_AMEAS_SEC
plt.figure(figsize=(14, 5))
plt.plot(t_ds[m_am], A_ds[m_am], linewidth=1.0)
plt.title(f"A_meas (first {ZOOM_AMEAS_SEC} s)")
plt.xlabel("Elapsed time (s)")
plt.ylabel("A_meas")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "plot_09_A_meas_zoom.png"), dpi=150)
plt.close()

# --------------------------------------------------
# 9) SAVE SUMMARY
# --------------------------------------------------
summary = pd.DataFrame([{
    "input_csv": INPUT_CSV,
    "rows_before": before,
    "rows_after": after,
    "fs_used": fs_used,
    "raw_mean": float(np.mean(disp_raw)),
    "raw_std": float(np.std(disp_raw)),
    "raw_min": float(np.min(disp_raw)),
    "raw_max": float(np.max(disp_raw)),
    "det_mean": float(np.mean(disp_det)),
    "det_std": float(np.std(disp_det)),
    "det_min": float(np.min(disp_det)),
    "det_max": float(np.max(disp_det)),
    "bp_mean": float(np.mean(disp_bp)),
    "bp_std": float(np.std(disp_bp)),
    "bp_min": float(np.min(disp_bp)),
    "bp_max": float(np.max(disp_bp)),
    "A_meas_mean": float(np.mean(A_ds)),
    "A_meas_std": float(np.std(A_ds)),
    "A_meas_min": float(np.min(A_ds)),
    "A_meas_max": float(np.max(A_ds)),
}])

summary_csv = os.path.join(OUT_DIR, "summary_stats.csv")
summary.to_csv(summary_csv, index=False)

print("\nSaved files:")
print("-", OUT_CLEAN_CSV)
print("-", OUT_AMEAS_CSV)
print("-", summary_csv)
print("-", os.path.join(OUT_DIR, "plot_01_raw_with_bad_rows.png"))
print("-", os.path.join(OUT_DIR, "plot_02_clean_full.png"))
print("-", os.path.join(OUT_DIR, "plot_03_clean_zoom.png"))
print("-", os.path.join(OUT_DIR, "plot_04_detrended_full.png"))
print("-", os.path.join(OUT_DIR, "plot_05_detrended_zoom.png"))
print("-", os.path.join(OUT_DIR, "plot_06_bandpassed_full.png"))
print("-", os.path.join(OUT_DIR, "plot_07_bandpassed_zoom.png"))
print("-", os.path.join(OUT_DIR, "plot_08_A_meas_full.png"))
print("-", os.path.join(OUT_DIR, "plot_09_A_meas_zoom.png"))

print("\nDone.")