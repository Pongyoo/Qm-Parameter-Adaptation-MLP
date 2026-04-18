import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# ==================6.1_2.5_tb_fix6.1==========================================
# Compare exp1 / exp2 / exp3 using the SAME processing logic
# Purpose:
#   1) raw displacement stats
#   2) detrended displacement stats
#   3) bandpassed displacement stats
#   4) A_meas stats (rolling RMS + downsample)
# ============================================================

# ================= USER CONFIG =================
FS_RAW = 100_000
BPF_LO = 15_000
BPF_HI = 25_000
FILTER_ORDER = 4
RMS_WIN_MS = 50
DS_FACTOR = 100
FS_DS = FS_RAW // DS_FACTOR

# ---- put your clean/full csv paths here ----
FILES = {
    #"exp6.1_2.5_tb_fix6.1": r"C:\Users\ploy\Desktop\ML\GRU_2\processed_Qm_sessions\exp6.1_2.5_tb_fix6.1\stepQm0_clean\2.5_tb_fix17.14_exp6.1_clean.csv",
    "0.5_tb_fix18.5_exp6": r"E:\processed_Qm\exp6(2026.4.2)\0.5_tb_fix18.5_exp6\stepQm0_clean\0.5_tb_fix18.5_exp6_clean.csv",
    "1_tb_fix18.3_exp6": r"E:\processed_Qm\exp6(2026.4.2)\1_tb_fix18.3_exp6\stepQm0_clean\1_tb_fix18.3_exp6_clean.csv",
    "exp6_1.5_tb_fix18.1_str": r"C:\Users\ploy\Desktop\ML\GRU_2\processed_Qm_sessions\exp6_1.5_tb_fix18.1_str\stepQm0_clean\1.5_tb_fix18.1_str_exp6_clean.csv",
}

OUT_DIR = r"C:\Users\ploy\Desktop\ML\GRU_2\compare_stats_across_experiments\exp6"
os.makedirs(OUT_DIR, exist_ok=True)
# ============================================================


def butter_bandpass(x, fs, lo, hi, order=4):
    nyq = 0.5 * fs
    lo_n = lo / nyq
    hi_n = hi / nyq
    b, a = butter(order, [lo_n, hi_n], btype="bandpass")
    return filtfilt(b, a, x)


def rolling_rms_valid(x, win):
    x = np.asarray(x, dtype=np.float64)
    x2 = x * x
    cs = np.cumsum(np.insert(x2, 0, 0.0))
    y = np.sqrt((cs[win:] - cs[:-win]) / win)
    return y


def pick_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


all_stats = []
plot_data = {}

print("\n================ Compare displacement stats across experiments ================")

for name, path in FILES.items():
    print(f"\n--- {name} ---")
    print("PATH:", path)
    print("EXISTS:", os.path.isfile(path))

    if not os.path.isfile(path):
        print(">> file not found, skip")
        continue

    df = pd.read_csv(path)
    print("shape:", df.shape)
    print("columns:", df.columns.tolist())

    disp_col = pick_col(df, ["Displacement", "displacement", "Disp", "disp"])
    time_col = pick_col(df, ["Time", "time", "t_sec", "t"])

    if disp_col is None:
        print(">> no displacement column, skip")
        continue

    disp_raw = pd.to_numeric(df[disp_col], errors="coerce").to_numpy(dtype=np.float64)
    m = np.isfinite(disp_raw)
    disp_raw = disp_raw[m]

    if len(disp_raw) == 0:
        print(">> displacement empty after NaN removal, skip")
        continue

    # time
    if time_col is not None:
        t_raw = pd.to_numeric(df.loc[m, time_col], errors="coerce").to_numpy(dtype=np.float64)
        if np.isfinite(t_raw).sum() < 2:
            t_raw = np.arange(len(disp_raw), dtype=np.float64) / FS_RAW
    else:
        t_raw = np.arange(len(disp_raw), dtype=np.float64) / FS_RAW

    # detrend
    disp_det = disp_raw - np.mean(disp_raw)

    # bandpass
    disp_bp = butter_bandpass(disp_det, FS_RAW, BPF_LO, BPF_HI, order=FILTER_ORDER)

    # A_meas logic (same style as StepQm1)
    win = max(1, int(round(RMS_WIN_MS * 1e-3 * FS_RAW)))
    A_env = rolling_rms_valid(disp_bp, win)
    t_env = t_raw[win - 1:]

    A_ds = A_env[::DS_FACTOR]
    t_ds = t_env[::DS_FACTOR]

    # stats
    stat = {
        "experiment": name,
        "N_raw": len(disp_raw),
        "duration_sec_est": len(disp_raw) / FS_RAW,

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
    }
    all_stats.append(stat)

    print("raw mean/std/min/max =", stat["raw_mean"], stat["raw_std"], stat["raw_min"], stat["raw_max"])
    print("det mean/std/min/max =", stat["det_mean"], stat["det_std"], stat["det_min"], stat["det_max"])
    print("bp  mean/std/min/max =", stat["bp_mean"], stat["bp_std"], stat["bp_min"], stat["bp_max"])
    print("A_meas mean/std/min/max =", stat["A_meas_mean"], stat["A_meas_std"], stat["A_meas_min"], stat["A_meas_max"])

    plot_data[name] = {
        "t_raw": t_raw,
        "disp_raw": disp_raw,
        "disp_det": disp_det,
        "disp_bp": disp_bp,
        "t_ds": t_ds,
        "A_ds": A_ds,
    }

# ---------------- save stats table ----------------
if len(all_stats) == 0:
    raise RuntimeError("No valid files loaded.")

df_stats = pd.DataFrame(all_stats)
stats_csv = os.path.join(OUT_DIR, "compare_stats.csv")
df_stats.to_csv(stats_csv, index=False)
print("\nSaved stats table:", stats_csv)

# ---------------- plots ----------------
# Plot 1: raw displacement (first 0.2 s)
plt.figure(figsize=(14, 6))
for name, d in plot_data.items():
    n_zoom = min(len(d["t_raw"]), int(0.2 * FS_RAW))
    plt.plot(d["t_raw"][:n_zoom], d["disp_raw"][:n_zoom], label=name, linewidth=1.2)
plt.title("Raw displacement (first 0.2 s)")
plt.xlabel("Time (s)")
plt.ylabel("Displacement")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "raw_displacement_zoom.png"), dpi=150)
plt.close()

# Plot 2: detrended displacement (first 0.2 s)
plt.figure(figsize=(14, 6))
for name, d in plot_data.items():
    n_zoom = min(len(d["t_raw"]), int(0.2 * FS_RAW))
    plt.plot(d["t_raw"][:n_zoom], d["disp_det"][:n_zoom], label=name, linewidth=1.2)
plt.title("Detrended displacement (first 0.2 s)")
plt.xlabel("Time (s)")
plt.ylabel("Displacement detrended")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "detrended_displacement_zoom.png"), dpi=150)
plt.close()

# Plot 3: bandpassed displacement (first 0.02 s)
plt.figure(figsize=(14, 6))
for name, d in plot_data.items():
    n_zoom = min(len(d["t_raw"]), int(0.02 * FS_RAW))
    plt.plot(d["t_raw"][:n_zoom], d["disp_bp"][:n_zoom], label=name, linewidth=1.2)
plt.title("Bandpassed displacement (first 0.02 s)")
plt.xlabel("Time (s)")
plt.ylabel("Bandpassed displacement")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "bandpassed_displacement_zoom.png"), dpi=150)
plt.close()

# Plot 4: A_meas full
plt.figure(figsize=(14, 6))
for name, d in plot_data.items():
    plt.plot(d["t_ds"], d["A_ds"], label=name, linewidth=1.2)
plt.title("A_meas full comparison")
plt.xlabel("Time (s)")
plt.ylabel("A_meas")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "A_meas_full_comparison.png"), dpi=150)
plt.close()

# Plot 5: A_meas zoom (first 5 s)
plt.figure(figsize=(14, 6))
for name, d in plot_data.items():
    m = d["t_ds"] <= 5.0
    plt.plot(d["t_ds"][m], d["A_ds"][m], label=name, linewidth=1.2)
plt.title("A_meas zoom comparison (first 5 s)")
plt.xlabel("Time (s)")
plt.ylabel("A_meas")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "A_meas_zoom_comparison.png"), dpi=150)
plt.close()

print("\nSaved plots to:", OUT_DIR)
print("Done.")