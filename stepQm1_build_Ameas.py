# ============================================================
# stepQm1_build_Ameas.py
# Step 1: Build measured amplitude A_meas(t)
# (FIX: Case 2 - Time column bad / not finite / wrong unit)
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# ================= USER CONFIG =================
ROOT = r"C:\Users\ploy\Desktop\ML\GRU_2"
BASE = "2.5nl-fix17.37"

CLEAN_PATH = os.path.join(
    ROOT, "processed", "01_clean_fix", f"{BASE}_clean.csv"
)

OUT_DIR = os.path.join(
    ROOT, "processed_Qm", "stepQm1_meas"
)
PLOT_DIR = os.path.join(OUT_DIR, "plots")
DATA_DIR = os.path.join(OUT_DIR, "data")

FS_RAW = 100_000
RMS_WIN_MS = 50
BPF_LO = 15_000
BPF_HI = 25_000
FILTER_ORDER = 4
DS_FACTOR = 100  # -> 1 kHz
# =================================================

os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# ---------------- utils ----------------
def butter_bandpass(x, fs, lo, hi, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [lo/nyq, hi/nyq], btype="bandpass")
    return filtfilt(b, a, x)

def rolling_rms(x, win):
    x2 = x**2
    cs = np.cumsum(np.insert(x2, 0, 0.0))
    return np.sqrt((cs[win:] - cs[:-win]) / win)

# ---------------- load data ----------------
df = pd.read_csv(CLEAN_PATH)

disp_col = [c for c in df.columns if "displac" in c.lower()][0]
disp = df[disp_col].to_numpy(dtype=float)

# --- read time ---
t = df["Time"].to_numpy(dtype=float)

# ===== (1) TIME SANITY + rebuild if bad =====
dt = np.diff(t)
finite_ratio = np.isfinite(t).mean()

print("Time sanity:")
print("  finite_ratio =", finite_ratio)
print("  t min/max =", np.nanmin(t), np.nanmax(t))
print("  dt median/min/max =", np.nanmedian(dt), np.nanmin(dt), np.nanmax(dt))

bad_time = (
    (finite_ratio < 0.99) or
    ((np.nanmax(t) - np.nanmin(t)) < 1.0) or
    (not np.isfinite(np.nanmedian(dt))) or
    (np.nanmedian(dt) <= 0)
)

if bad_time:
    print("⚠️ Time column looks bad. Rebuild t from FS_RAW.")
    t = np.arange(len(disp), dtype=float) / FS_RAW

# ---------------- detrend displacement ----------------
disp -= np.mean(disp)

# ===== (A) RAW displacement sanity =====
print("RAW displacement:")
print("  len =", len(disp))
print("  min/max/mean =", np.min(disp), np.max(disp), np.mean(disp))

# ---------------- filtering ----------------
disp_f = butter_bandpass(disp, FS_RAW, BPF_LO, BPF_HI, FILTER_ORDER)

# ===== (B) FILTERED displacement sanity =====
print("FILTERED displacement:")
print("  min/max/mean =", np.min(disp_f), np.max(disp_f), np.mean(disp_f))

# ---------------- RMS ----------------
RMS_WIN = int(FS_RAW * RMS_WIN_MS / 1000)
if RMS_WIN < 2:
    raise ValueError("RMS_WIN too small. Check FS_RAW or RMS_WIN_MS.")

A_rms = rolling_rms(disp_f, RMS_WIN)
t_rms = t[RMS_WIN - 1:]

# ===== (C) RMS raw sanity =====
print("RMS raw:")
print("  len =", len(A_rms))
print("  min/max/mean =", np.min(A_rms), np.max(A_rms), np.mean(A_rms))

# ---------------- downsample ----------------
A_meas = A_rms[::DS_FACTOR]
t_meas = t_rms[::DS_FACTOR]

# ===== (D) A_meas @1kHz sanity =====
print("A_meas (1kHz):")
print("  len =", len(A_meas))
print("  min/max/mean =", np.min(A_meas), np.max(A_meas), np.mean(A_meas))
print("  t_meas min/max =", float(t_meas[0]), float(t_meas[-1]))

# ---------------- save data ----------------
np.savez(
    os.path.join(DATA_DIR, "A_meas_1k.npz"),
    t=t_meas.astype(np.float32),
    A_meas=A_meas.astype(np.float32),
)

# ===== (3) Save CSV for easier debug =====
pd.DataFrame({"t_sec": t_meas, "A_meas": A_meas}).to_csv(
    os.path.join(DATA_DIR, "A_meas_1k.csv"), index=False
)

# ---------------- plots ----------------
X_LABEL = "Time (s)"
Y_DISP_LABEL = "Displacement (µm)"
Y_RMS_LABEL  = "RMS amplitude (µm)"

# raw vs filtered (first 20000 samples)
plt.figure(figsize=(10,4))
plt.plot(t[:20000], disp[:20000], label="raw", lw=1.0)
plt.plot(t[:20000], disp_f[:20000], label="filtered", lw=1.0)
plt.title("Displacement: raw vs filtered")
plt.xlabel(X_LABEL)
plt.ylabel(Y_DISP_LABEL)
plt.legend()
plt.grid(True, alpha=0.3)

plt.xlim(t[0], t[min(19999, len(t)-1)])
ymin = min(disp[:20000].min(), disp_f[:20000].min())
ymax = max(disp[:20000].max(), disp_f[:20000].max())
if np.isfinite(ymin) and np.isfinite(ymax) and ymax > ymin:
    plt.ylim(ymin*1.02 if ymin < 0 else ymin*0.98, ymax*1.02)

plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "disp_raw_vs_filtered.png"), dpi=150)
plt.close()

# A_meas zoom (first 3000 points)
n_zoom = min(3000, len(t_meas))
plt.figure(figsize=(10,4))
plt.plot(t_meas[:n_zoom], A_meas[:n_zoom], lw=1.0)
plt.title("A_meas (1 kHz, zoom)")
plt.xlabel(X_LABEL)
plt.ylabel(Y_RMS_LABEL)
plt.grid(True, alpha=0.3)

plt.xlim(t_meas[0], t_meas[n_zoom-1])
amin = A_meas[:n_zoom].min()
amax = A_meas[:n_zoom].max()
if amax > amin:
    plt.ylim(amin*0.98, amax*1.02)

plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "A_meas_raw_1k.png"), dpi=150)
plt.close()

# A_meas full
plt.figure(figsize=(10,4))
plt.plot(t_meas, A_meas, lw=1.0)
plt.title("A_meas trend (full)")
plt.xlabel(X_LABEL)
plt.ylabel(Y_RMS_LABEL)
plt.grid(True, alpha=0.3)

plt.xlim(t_meas[0], t_meas[-1])
amin = A_meas.min()
amax = A_meas.max()
if amax > amin:
    plt.ylim(amin*0.98, amax*1.02)

plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "A_meas_trend.png"), dpi=150)
plt.close()
