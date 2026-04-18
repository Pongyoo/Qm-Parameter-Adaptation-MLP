# ============================================================
# stepQm1_exp3_build_Ameas.py
# Step 1 for EXP3: Build measured amplitude A_meas(t)
#
# Session:
#   folder   : C:\Users\ploy\Desktop\raw_data\exp(2026.3.13)
#   fix file : 2.5-fix17.0  (set full filename below)
#
# Output:
#   C:\Users\ploy\Desktop\ML\GRU_2\processed_Qm_sessions\<SESSION_TAG>\stepQm1_meas\data\A_meas_1k.csv
#   C:\Users\ploy\Desktop\ML\GRU_2\processed_Qm_sessions\<SESSION_TAG>\stepQm1_meas\data\A_meas_1k.npz
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# ================= USER CONFIG =================
ROOT = r"C:\Users\ploy\Desktop"

INPUT_PATH = r"C:\Users\ploy\Desktop\ML\GRU_2\processed_Qm_sessions\exp1_3nl-fix17.37\stepQm0_clean\3nl-fix17.37_exp1_clean.csv"

SESSION_TAG = "exp1_3nl_fix17.37"
SESSION_ROOT = os.path.join(ROOT, "ML", "GRU_2", "processed_Qm_sessions", SESSION_TAG)

OUT_DIR = os.path.join(SESSION_ROOT, "stepQm1_meas")
PLOT_DIR = os.path.join(OUT_DIR, "plots")
DATA_DIR = os.path.join(OUT_DIR, "data")

FS_RAW = 100_000
RMS_WIN_MS = 50
BPF_LO = 15_000
BPF_HI = 25_000
FILTER_ORDER = 4
DS_FACTOR = 100      # 100 kHz -> 1 kHz
FS_DS = FS_RAW // DS_FACTOR
# =================================================

os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)


def load_table(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.read_csv(path, encoding="latin1")
    elif ext == ".txt":
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.read_csv(path, sep="\t", encoding="latin1")
    elif ext in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")


def pick_first_existing(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def find_displacement_col(df):
    cols = list(df.columns)

    # keyword search
    keywords = ["displacement", "displac", "disp", "amplitude"]
    for c in cols:
        cl = str(c).lower()
        if any(k in cl for k in keywords):
            return c

    # common exact names
    for c in ["Displacement", "displacement", "Disp", "disp"]:
        if c in df.columns:
            return c

    raise KeyError(
        "Cannot find displacement column.\n"
        f"Available columns:\n{cols}"
    )


def find_time_col(df):
    candidates = ["Time", "time", "t_sec", "t", "Time(s)", "time_s", "timestamp"]
    return pick_first_existing(df, candidates)


def butter_bandpass(x, fs, lo, hi, order=4):
    nyq = 0.5 * fs
    lo_n = lo / nyq
    hi_n = hi / nyq
    if not (0 < lo_n < hi_n < 1):
        raise ValueError(f"Invalid bandpass range: lo={lo}, hi={hi}, fs={fs}")
    b, a = butter(order, [lo_n, hi_n], btype="bandpass")
    return filtfilt(b, a, x)


def rolling_rms_valid(x, win):
    x = np.asarray(x, dtype=np.float64)
    x2 = x * x
    cs = np.cumsum(np.insert(x2, 0, 0.0))
    y = np.sqrt((cs[win:] - cs[:-win]) / win)
    return y


# ================= debug print =================
print("\n================ StepQm1 EXP3: Build A_meas ================")
print("INPUT_PATH    =", INPUT_PATH)
print("INPUT EXISTS? =", os.path.isfile(INPUT_PATH))

if not os.path.isfile(INPUT_PATH):
    raise FileNotFoundError(
        f"Input file not found:\n{INPUT_PATH}"
    )

# ================= load input =================
df = load_table(INPUT_PATH)
print("Loaded shape:", df.shape)
print("Columns:", list(df.columns))

disp_col = find_displacement_col(df)
disp = pd.to_numeric(df[disp_col], errors="coerce").to_numpy(dtype=np.float64)

valid_disp = np.isfinite(disp)
if valid_disp.mean() < 0.95:
    print(f"Warning: displacement finite ratio = {valid_disp.mean():.4f}")

df = df.loc[valid_disp].reset_index(drop=True)
disp = disp[valid_disp]

time_col = find_time_col(df)
if time_col is not None:
    t_raw = pd.to_numeric(df[time_col], errors="coerce").to_numpy(dtype=np.float64)
else:
    t_raw = np.full(len(df), np.nan, dtype=np.float64)

# ================= time sanity =================
if len(t_raw) == 0:
    raise ValueError("Input file has zero rows after cleaning.")

finite_ratio = np.isfinite(t_raw).mean() if len(t_raw) > 0 else 0.0
dt = np.diff(t_raw[np.isfinite(t_raw)]) if np.isfinite(t_raw).sum() >= 2 else np.array([np.nan])

t_span = float(np.nanmax(t_raw) - np.nanmin(t_raw)) if np.isfinite(t_raw).any() else np.nan
dt_med = float(np.nanmedian(dt)) if np.isfinite(dt).any() else np.nan
dt_min = float(np.nanmin(dt)) if np.isfinite(dt).any() else np.nan
dt_max = float(np.nanmax(dt)) if np.isfinite(dt).any() else np.nan

print("\nTime sanity:")
print("  time_col           =", time_col)
print("  finite_ratio       =", finite_ratio)
print("  t_span             =", t_span)
print("  dt median/min/max  =", dt_med, dt_min, dt_max)

bad_time = (
    (time_col is None) or
    (finite_ratio < 0.99) or
    (not np.isfinite(t_span)) or
    (t_span < 1.0) or
    (not np.isfinite(dt_med)) or
    (dt_med <= 0)
)

if bad_time:
    print("Time column bad or missing -> rebuild from FS_RAW")
    t = np.arange(len(disp), dtype=np.float64) / FS_RAW
else:
    t = t_raw.copy()

# ================= preprocess =================
disp = disp - np.nanmean(disp)
disp_bp = butter_bandpass(disp, FS_RAW, BPF_LO, BPF_HI, order=FILTER_ORDER)

win = max(1, int(round(RMS_WIN_MS * 1e-3 * FS_RAW)))
A_env = rolling_rms_valid(disp_bp, win)
t_env = t[win - 1:]

# downsample to 1 kHz
A_ds = A_env[::DS_FACTOR]
t_ds = t_env[::DS_FACTOR]
k_ds = np.arange(len(A_ds), dtype=np.int64)

# ================= save output =================
df_out = pd.DataFrame({
    "k": k_ds,
    "t_sec": t_ds.astype(np.float64),
    "A_meas": A_ds.astype(np.float64),
})

print("A_ds min/max/mean =", A_ds.min(), A_ds.max(), A_ds.mean())
print("df_out A_meas min/max/mean =", df_out["A_meas"].min(), df_out["A_meas"].max(), df_out["A_meas"].mean())

csv_path = os.path.join(DATA_DIR, "A_meas_1k.csv")
npz_path = os.path.join(DATA_DIR, "A_meas_1k.npz")

df_out.to_csv(csv_path, index=False)
np.savez(
    npz_path,
    k=k_ds,
    t_sec=t_ds.astype(np.float64),
    A_meas=A_ds.astype(np.float64),
)

print("\nSaved:")
print(" ", csv_path)
print(" ", npz_path)
print("Output length =", len(df_out))

print("raw mean/std =", np.mean(disp), np.std(disp))
print("filtered mean/std =", np.mean(disp_bp), np.std(disp_bp))
print("raw min/max =", np.min(disp), np.max(disp))
print("filtered min/max =", np.min(disp_bp), np.max(disp_bp))

# ================= plots =================
import matplotlib.pyplot as plt

fix_label = os.path.splitext(os.path.basename(INPUT_PATH))[0]

# ---------- Plot 1: displacement raw vs filtered (zoom first 0.2 s) ----------
zoom_sec_disp = 0.2
n_zoom_disp = min(len(t), int(zoom_sec_disp * FS_RAW))

plt.figure(figsize=(15, 7))
plt.plot(t[:n_zoom_disp], disp[:n_zoom_disp], label="raw", linewidth=1.2)
plt.plot(t[:n_zoom_disp], disp_bp[:n_zoom_disp], label="filtered", linewidth=1.2)
plt.xlabel("Time (s)", fontsize=18)
plt.ylabel("Displacement (µm)", fontsize=18)
plt.title("Displacement: raw vs filtered", fontsize=24)
plt.legend(fontsize=18)
plt.grid(True, alpha=0.3)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, f"{fix_label}_disp_raw_vs_filtered.png"), dpi=150)
plt.close()

# ---------- Plot 2: A_meas full trend ----------
plt.figure(figsize=(15, 7))
plt.plot(df_out["t_sec"], df_out["A_meas"], linewidth=1.5)
plt.xlabel("Time (s)", fontsize=18)
plt.ylabel("RMS amplitude (µm)", fontsize=18)
plt.title("A_meas trend (full)", fontsize=24)
plt.grid(True, alpha=0.3)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, f"{fix_label}_A_meas_full.png"), dpi=150)
plt.close()

# ---------- Plot 3: A_meas zoom ----------
zoom_sec_ameas = 3.05
mask_zoom = df_out["t_sec"] <= zoom_sec_ameas

plt.figure(figsize=(15, 7))
plt.plot(df_out.loc[mask_zoom, "t_sec"], df_out.loc[mask_zoom, "A_meas"], linewidth=1.5)
plt.xlabel("Time (s)", fontsize=18)
plt.ylabel("RMS amplitude (µm)", fontsize=18)
plt.title("A_meas (1 kHz, zoom)", fontsize=24)
plt.grid(True, alpha=0.3)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, f"{fix_label}_A_meas_zoom.png"), dpi=150)
plt.close()

print("\nSaved plots:")
print(" ", os.path.join(PLOT_DIR, f"{fix_label}_disp_raw_vs_filtered.png"))
print(" ", os.path.join(PLOT_DIR, f"{fix_label}_A_meas_full.png"))
print(" ", os.path.join(PLOT_DIR, f"{fix_label}_A_meas_zoom.png"))