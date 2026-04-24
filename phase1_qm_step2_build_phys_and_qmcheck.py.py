# ============================================================
# stepQm_build_phys_and_qmcheck.py
#
# รวม stepQm2 + stepQm4c + stepQm4d เป็นไฟล์เดียว
#
# Pipeline:
#   sweep params json  (fn, Qm)   ← จาก stepQm1p (รันแยกแล้ว)
#   + A_meas / I_rms / t          ← จาก phase0_preprocessing.py
#   + raw CSV (for temp cols)     ← original fix CSV
#
#   STEP A  Build physics baseline
#           f_inst tracking (Welch) → G_res → A_phys
#
#   STEP B  Qm-required bound check
#           Qm_req = A_meas / (I_rms * G_res_ref)
#           EMA → Qm_slow, Qm_fast
#           A_hat = I_rms * G_res_ref * (Qm_slow / Qm_ref)
#           Feasibility metrics (full + test split)
#
#   STEP C  Merge temperature columns (PZT, Tool_temp)
#           Align raw temp @ 100kHz → 1kHz grid
#
#   Output  CSV + NPZ + 12 plots + metrics.json
#
# Inputs:
#   IN_AMEAS_DIR/   t.npy, A_meas.npy, I_rms.npy  (from phase0)
#   SWEEP_JSON      params_<SWEEP_NAME>.json        (from stepQm1p)
#   FIX_CSV         original fix CSV               (for temp cols)
#
# Outputs (all under OUT_DIR/):
#   data/  A_phys_1k.csv  A_phys_1k.npz
#          qm_boundcheck.csv  metrics.json
#          merged_1k.csv  merged_1k.npz   ← final merged file for stepQm5
#   plots/ 12 diagnostic plots
# ============================================================

import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, welch

# ══════════════════════════════════════════════════════════════════
#  CONFIG  ← แก้ตรงนี้เท่านั้น
# ══════════════════════════════════════════════════════════════════
BASE        = "1.5_exp5"
SWEEP_NAME  = "1.5_swp30s_exp5"

# input dirs
IN_AMEAS_DIR = r"E:\raw_data\exp5(2026.3.27)\1.5_fix18.0_processed_Qm\stepQm1_clean\data"
SWEEP_JSON    = r"E:\raw_data\exp5(2026.3.27)\1.5_fix18.0_processed_Qm\stepQm1_para_sweep\1.5_swp30s_exp5\stepQm1_params_sweep\params_1.5_swp30s_exp5.json"
FIX_CSV      = r"E:\raw_data\exp5(2026.3.27)\1.5_fix18.0_exp5.csv"

OUT_DIR = r"E:\raw_data\exp5(2026.3.27)\1.5_fix18.0_processed_Qm\stepQm2_build_phys_and_qmcheck_aclean"

# Signal constants
FS_RAW      = 100_000
DS_FACTOR   = 100          # → 1 kHz
BPF_LO      = 15_000.0
BPF_HI      = 25_000.0
FILTER_ORD  = 4

# Welch frequency tracking
FTRACK_WIN_SEC  = 0.20
FTRACK_HOP_SEC  = 0.05
FN_EST_T0       = 1.0       # region to estimate fn_fix from fix data
FN_EST_T1       = 6.0

C_SCALE = 1.0

# EMA time constant for Qm_slow (seconds)
TAU_S = 10

# y-space reference / train split (consistent with stepQm5/6)
REF_N        = 1000
TRAIN_RATIO  = 0.70

EPS = 1e-12
# ══════════════════════════════════════════════════════════════════

PLOT_DIR = os.path.join(OUT_DIR, "plots")
DATA_DIR = os.path.join(OUT_DIR, "data")
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)


# ──────────────────────────────────────────────────────────────────
#  HELPERS
# ──────────────────────────────────────────────────────────────────
def butter_bandpass(x, fs, lo, hi, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [lo / nyq, hi / nyq], btype="bandpass")
    return filtfilt(b, a, x)

def rolling_rms(x, win):
    x = x.astype(np.float64)
    cs = np.cumsum(np.insert(x**2, 0, 0.0))
    return np.sqrt((cs[win:] - cs[:-win]) / win)

def resonance_gain(f, fn, Qm):
    r = f / (fn + EPS)
    z = 1.0 / (2.0 * (Qm + EPS))
    return 1.0 / np.sqrt((1.0 - r**2)**2 + (2.0 * z * r)**2 + EPS)

def peak_freq_welch(seg, fs, fmin, fmax):
    f, P = welch(seg, fs=fs, nperseg=min(len(seg), 4096))
    m = (f >= fmin) & (f <= fmax)
    if not np.any(m) or not np.all(np.isfinite(P[m])):
        return np.nan
    return float(f[m][np.argmax(P[m])])

def track_frequency(x_bp, t, fs, win_sec, hop_sec, fmin, fmax):
    win = max(int(win_sec * fs), 1024)
    hop = max(int(hop_sec * fs), 1)
    f_list, tc_list = [], []
    for i0 in range(0, len(x_bp) - win + 1, hop):
        f_list.append(peak_freq_welch(x_bp[i0:i0+win], fs, fmin, fmax))
        tc_list.append(float(t[i0 + win // 2]))
    return np.array(tc_list), np.array(f_list)

def ema(x, alpha):
    y = np.empty_like(x, dtype=np.float64)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = alpha * x[i] + (1 - alpha) * y[i-1]
    return y

def pick_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def savefig(fig, name):
    p = os.path.join(PLOT_DIR, name)
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  [plot] {p}")


# ──────────────────────────────────────────────────────────────────
#  LOAD  inputs
# ──────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print(f"stepQm_build_phys_and_qmcheck  |  BASE={BASE}")
print("="*60)

t       = np.load(os.path.join(IN_AMEAS_DIR, "t.npy")).astype(np.float64)
A_meas  = np.load(os.path.join(IN_AMEAS_DIR, "A_meas.npy")).astype(np.float64)
A_clean = np.load(os.path.join(IN_AMEAS_DIR, "A_clean.npy")).astype(np.float64)
I_rms   = np.load(os.path.join(IN_AMEAS_DIR, "I_rms.npy")).astype(np.float64)
FS_DS   = float(round(1.0 / np.median(np.diff(t))))

with open(SWEEP_JSON, "r") as f:
    sweep = json.load(f)
Qm_ref   = float(sweep["Qm"])
fn_sweep = float(sweep["fn_hz"])

N = min(len(t), len(A_meas), len(I_rms))
t, A_meas, A_clean, I_rms = t[:N], A_meas[:N], A_clean[:N], I_rms[:N]

print(f"  n={N:,}  duration={t[-1]:.1f}s  fs_ds={FS_DS:.0f}Hz")
print(f"  Qm_ref={Qm_ref:.3f}  fn_sweep={fn_sweep:.3f}Hz")


# ──────────────────────────────────────────────────────────────────
#  STEP A — Build A_phys (frequency tracking + G_res)
# ──────────────────────────────────────────────────────────────────
print("\n[STEP A] Build A_phys ...")

# Load raw fix CSV (Current only, for frequency tracking)
df_fix    = pd.read_csv(FIX_CSV)
col_cur   = pick_col(df_fix, ["Current", "current", "I"])
col_t_raw = pick_col(df_fix, ["Time", "t_sec", "time"])

cur_raw = pd.to_numeric(df_fix[col_cur], errors="coerce").dropna().to_numpy(dtype=np.float64)
fs_raw  = float(FS_RAW)

# estimate actual fs from time column if available
if col_t_raw is not None:
    t_raw_col = pd.to_numeric(df_fix[col_t_raw], errors="coerce").dropna().to_numpy(dtype=np.float64)
    if len(t_raw_col) > 10 and np.isfinite(t_raw_col).mean() > 0.8:
        dt_est = np.median(np.diff(t_raw_col[:10000]))
        if 1e-6 < dt_est < 0.01:
            fs_raw = float(round(1.0 / dt_est))

cur_raw -= cur_raw.mean()
cur_bp   = butter_bandpass(cur_raw, fs_raw, BPF_LO, BPF_HI, FILTER_ORD)
t_raw    = np.arange(len(cur_raw)) / fs_raw

# frequency tracking
t_fk, f_pk = track_frequency(cur_bp, t_raw, fs_raw,
                              FTRACK_WIN_SEC, FTRACK_HOP_SEC,
                              BPF_LO, BPF_HI)

# estimate fn_fix from steady region
region = (t_fk >= FN_EST_T0) & (t_fk <= FN_EST_T1) & np.isfinite(f_pk)
fn_fix = float(np.median(f_pk[region])) if region.sum() >= 5 else float(np.nanmedian(f_pk))
print(f"  fn_fix={fn_fix:.3f}Hz  (ref fn_sweep={fn_sweep:.3f}Hz)")

# interpolate f_inst to 1kHz grid
f_inst = np.interp(t, t_fk, np.nan_to_num(f_pk, nan=fn_fix))
f_inst = np.clip(f_inst, BPF_LO, BPF_HI)

G_res   = resonance_gain(f_inst, fn_fix, Qm_ref)
A_phys  = C_SCALE * I_rms * G_res

print(f"  A_phys: mean={A_phys.mean():.5f}  std={A_phys.std():.5f}")

# — plots —
fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(t_fk, f_pk, lw=0.8, label="f_peak (Welch)")
ax.axhline(fn_fix, ls="--", color="r", label=f"fn_fix={fn_fix:.1f}Hz")
ax.axhline(fn_sweep, ls=":", color="g", label=f"fn_sweep={fn_sweep:.1f}Hz")
ax.set(xlabel="Time (s)", ylabel="Freq (Hz)",
       title=f"{BASE} | Welch frequency tracking")
ax.legend(); ax.grid(True, alpha=0.25); fig.tight_layout()
savefig(fig, f"{BASE}_A_f_track.png")

fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
axes[0].plot(t, A_meas,  lw=0.8, color="#E06C75", alpha=0.7, label="A_meas")
axes[0].plot(t, A_clean, lw=1.2, color="#61AFEF", label="A_clean")
axes[0].plot(t, A_phys,  lw=1.2, color="#98C379", label="A_phys (Qm_ref)")
axes[0].set(ylabel="Amplitude (µm)",
            title=f"{BASE} | Amplitude overlay (stepA baseline)")
axes[0].legend(); axes[0].grid(True, alpha=0.25)
axes[1].plot(t, G_res, lw=0.8, color="#E5C07B")
axes[1].set(xlabel="Time (s)", ylabel="G_res (arb.)",
            title="Resonance gain G_res(t)")
axes[1].grid(True, alpha=0.25)
fig.tight_layout()
savefig(fig, f"{BASE}_B_Aphys_overlay.png")


# ──────────────────────────────────────────────────────────────────
#  STEP B — Qm-required bound check
# ──────────────────────────────────────────────────────────────────
print("\n[STEP B] Qm-required bound check ...")

# แก้ — smooth I_rms ก่อนใช้หาร Qm_required
from pandas import Series
I_smooth = Series(I_rms).rolling(500, center=True, min_periods=1).mean().values
I_safe   = np.maximum(I_smooth, EPS)

# G_res_ref ก็ใช้ I_smooth ด้วย
G_res_ref = float(np.median(A_clean / I_safe))
G_req     = A_clean / I_safe
Qm_req    = Qm_ref * (G_req / G_res_ref)

# EMA slow/fast separation
dt_s  = 1.0 / FS_DS
alpha = dt_s / (TAU_S + dt_s)
Qm_slow = ema(Qm_req, alpha)
Qm_fast = Qm_req - Qm_slow

# rebuild A_hat using Qm_slow only
G_hat = G_res_ref * (Qm_slow / max(Qm_ref, EPS))
A_hat = I_safe * G_hat

# fastness metrics
rms_total = float(np.std(Qm_req))
rms_fast  = float(np.std(Qm_fast))
fast_ratio = rms_fast / (rms_total + EPS)
dQ       = np.diff(Qm_req) / dt_s
dQ_slow  = np.diff(Qm_slow) / dt_s
deriv_ratio = float(np.mean(np.abs(dQ_slow)) / (np.mean(np.abs(dQ)) + EPS))

# y-space feasibility (log normalized)
def y_log(A, A_ref):
    return np.log(np.maximum(A, EPS) / max(A_ref, EPS))

A_ref_meas  = float(np.median(A_meas[:REF_N]))
A_ref_clean = float(np.median(A_clean[:REF_N]))
A_ref_phys  = float(np.median(A_phys[:REF_N]))
A_ref_hat   = float(np.median(A_hat[:REF_N]))

y_meas  = y_log(A_meas,  A_ref_meas)
y_clean = y_log(A_clean, A_ref_clean)
y_phys  = y_log(A_phys,  A_ref_phys)
y_hat   = y_log(A_hat,   A_ref_hat)

r_before = y_clean - y_phys     # residual before Qm adaptation
r_after  = y_clean - y_hat      # residual after  Qm_slow

def metrics(a, b):
    d = a - b
    return float(np.sqrt(np.mean(d**2))), float(np.mean(np.abs(d)))

rmse_bf, mae_bf = metrics(y_clean, y_phys)
rmse_af, mae_af = metrics(y_clean, y_hat)
improve_full = 100.0 * (rmse_bf - rmse_af) / (rmse_bf + EPS)

# test-only
n_tr   = int(TRAIN_RATIO * N)
sl_te  = slice(n_tr, None)
rmse_bf_te, mae_bf_te = metrics(y_clean[sl_te], y_phys[sl_te])
rmse_af_te, mae_af_te = metrics(y_clean[sl_te], y_hat[sl_te])
improve_test = 100.0 * (rmse_bf_te - rmse_af_te) / (rmse_bf_te + EPS)

print(f"  fast_ratio={fast_ratio:.3f}  deriv_ratio={deriv_ratio:.3f}")
print(f"  FULL   RMSE {rmse_bf:.5f} → {rmse_af:.5f}  improve={improve_full:.1f}%")
print(f"  TEST   RMSE {rmse_bf_te:.5f} → {rmse_af_te:.5f}  improve={improve_test:.1f}%")

# — plots —
fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
axes[0].plot(t, Qm_req,  lw=0.5, color="#888780", alpha=0.6, label="Qm_required")
axes[0].plot(t, Qm_slow, lw=1.5, color="#E5C07B", label=f"Qm_slow (tau={TAU_S}s)")
axes[0].axhline(Qm_ref,  lw=0.8, ls="--", color="k", label=f"Qm_ref={Qm_ref:.2f}")
axes[0].set(ylabel="Qm (arb.)",
            title=f"{BASE} | Qm_required vs Qm_slow")
axes[0].legend(); axes[0].grid(True, alpha=0.25)
axes[1].plot(t, Qm_fast, lw=0.8, color="#E06C75", alpha=0.8)
axes[1].axhline(0, lw=0.5, ls="--", color="k")
axes[1].set(xlabel="Time (s)", ylabel="Qm_fast",
            title=f"Qm_fast  (RMS fast/total={fast_ratio:.3f})")
axes[1].grid(True, alpha=0.25)
fig.tight_layout()
savefig(fig, f"{BASE}_C_Qm_required_slow.png")

fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
axes[0].plot(t, A_clean, lw=0.8, color="#61AFEF", alpha=0.9, label="A_clean")
axes[0].plot(t, A_phys,  lw=1.0, color="#E06C75", alpha=0.8, label="A_phys (Qm_ref)")
axes[0].plot(t, A_hat,   lw=1.0, color="#98C379", alpha=0.8, label="A_hat (Qm_slow)")
axes[0].set(ylabel="Amplitude (µm)",
            title=f"{BASE} | A_clean vs A_phys vs A_hat")
axes[0].legend(); axes[0].grid(True, alpha=0.25)
axes[1].plot(t, r_before, lw=0.7, color="#E06C75", alpha=0.7,
             label=f"r_before RMSE={rmse_bf:.4f}")
axes[1].plot(t, r_after,  lw=0.7, color="#98C379", alpha=0.7,
             label=f"r_after  RMSE={rmse_af:.4f}")
axes[1].axhline(0, lw=0.5, ls="--", color="k")
axes[1].set(xlabel="Time (s)", ylabel="y-residual",
            title=f"y-residual (FULL) — improve={improve_full:.1f}%")
axes[1].legend(); axes[1].grid(True, alpha=0.25)
fig.tight_layout()
savefig(fig, f"{BASE}_D_residual_before_after_full.png")

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].plot(t[sl_te], r_before[sl_te], lw=0.7, color="#E06C75",
             label=f"before RMSE={rmse_bf_te:.4f}")
axes[0].plot(t[sl_te], r_after[sl_te],  lw=0.7, color="#98C379",
             label=f"after  RMSE={rmse_af_te:.4f}")
axes[0].axhline(0, lw=0.5, ls="--", color="k")
axes[0].set(xlabel="Time (s)", ylabel="y-residual",
            title=f"{BASE} | TEST residual (improve={improve_test:.1f}%)")
axes[0].legend(); axes[0].grid(True, alpha=0.25)

axes[1].hist(r_before, bins=80, alpha=0.5, color="#E06C75", label="before")
axes[1].hist(r_after,  bins=80, alpha=0.5, color="#98C379", label="after")
axes[1].axvline(0, lw=0.8, ls="--", color="k")
axes[1].set(xlabel="y-residual", ylabel="Count",
            title="Residual histogram (FULL)")
axes[1].legend(); axes[1].grid(True, alpha=0.25)
fig.tight_layout()
savefig(fig, f"{BASE}_E_residual_test_hist.png")

# summary text plot
fig, ax = plt.subplots(figsize=(8, 6))
ax.axis("off")
lines = [
    f"BASE: {BASE}",
    f"N={N}  fs_ds={FS_DS:.0f}Hz  TAU_S={TAU_S}s",
    f"Qm_ref={Qm_ref:.4f}  fn_fix={fn_fix:.3f}Hz",
    f"G_res_ref={G_res_ref:.6f}",
    "",
    "Qm_required:",
    f"  min={Qm_req.min():.3f}  max={Qm_req.max():.3f}",
    f"  mean={Qm_req.mean():.3f}  std={Qm_req.std():.3f}",
    f"  fast_ratio={fast_ratio:.3f}  deriv_ratio={deriv_ratio:.3f}",
    "",
    "Feasibility (FULL):",
    f"  RMSE {rmse_bf:.5f} → {rmse_af:.5f}  ({improve_full:.1f}%)",
    f"  MAE  {mae_bf:.5f} → {mae_af:.5f}",
    "",
    "Feasibility (TEST only):",
    f"  RMSE {rmse_bf_te:.5f} → {rmse_af_te:.5f}  ({improve_test:.1f}%)",
    f"  MAE  {mae_bf_te:.5f} → {mae_af_te:.5f}",
]
ax.text(0.02, 0.98, "\n".join(lines), va="top", ha="left",
        fontsize=10, family="monospace")
fig.tight_layout()
savefig(fig, f"{BASE}_F_summary.png")


# ──────────────────────────────────────────────────────────────────
#  STEP C — Merge temperature columns (PZT, Tool_temp)
# ──────────────────────────────────────────────────────────────────
print("\n[STEP C] Merge temperature columns ...")

pzt_1k      = np.full(N, np.nan, dtype=np.float32)
tool_temp_1k = np.full(N, np.nan, dtype=np.float32)

try:
    col_pzt  = pick_col(df_fix, ["PZT", "pzt", "Pzt"])
    col_tool = pick_col(df_fix, ["Tool_temp", "tool_temp", "ToolTemp"])

    if col_pzt is not None:
        pzt_raw  = pd.to_numeric(df_fix[col_pzt],  errors="coerce").to_numpy(dtype=np.float64)
        t_raw_full = np.arange(len(pzt_raw)) / fs_raw
        pzt_1k   = np.interp(t, t_raw_full, pzt_raw).astype(np.float32)
        print(f"  PZT merged  mean={pzt_1k[np.isfinite(pzt_1k)].mean():.3f}")
    else:
        print("  ⚠️ PZT column not found — filled NaN")

    if col_tool is not None:
        tool_raw = pd.to_numeric(df_fix[col_tool], errors="coerce").to_numpy(dtype=np.float64)
        t_raw_full = np.arange(len(tool_raw)) / fs_raw
        tool_temp_1k = np.interp(t, t_raw_full, tool_raw).astype(np.float32)
        print(f"  Tool_temp merged  mean={tool_temp_1k[np.isfinite(tool_temp_1k)].mean():.3f}")
    else:
        print("  ⚠️ Tool_temp column not found — filled NaN")

except Exception as e:
    print(f"  ⚠️ Temp merge error: {e}  — continuing without temp")

# — temp plot —
fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
axes[0].plot(t, pzt_1k, lw=0.8, color="#E5C07B")
axes[0].set(ylabel="PZT temp (°C)",
            title=f"{BASE} | Temperature (merged @1kHz)")
axes[0].grid(True, alpha=0.25)
axes[1].plot(t, tool_temp_1k, lw=0.8, color="#E06C75")
axes[1].set(xlabel="Time (s)", ylabel="Tool temp (°C)")
axes[1].grid(True, alpha=0.25)
fig.tight_layout()
savefig(fig, f"{BASE}_G_temperature.png")


# ──────────────────────────────────────────────────────────────────
#  SAVE  all outputs
# ──────────────────────────────────────────────────────────────────
print("\n[Save] ...")

# A) A_phys CSV + NPZ
df_phys = pd.DataFrame({
    "t_sec":        t.astype(np.float32),
    "I_rms":        I_rms.astype(np.float32),
    "f_inst":       f_inst.astype(np.float32),
    "G_res":        G_res.astype(np.float32),
    "A_phys":       A_phys.astype(np.float32),
    "fn_used":      np.full(N, fn_fix,  dtype=np.float32),
    "Qm_used":      np.full(N, Qm_ref, dtype=np.float32),
})
df_phys.to_csv(os.path.join(DATA_DIR, "A_phys_1k.csv"), index=False)
np.savez_compressed(os.path.join(DATA_DIR, "A_phys_1k.npz"),
                    t=t.astype(np.float32),
                    I_rms=I_rms.astype(np.float32),
                    f_inst=f_inst.astype(np.float32),
                    G_res=G_res.astype(np.float32),
                    A_phys=A_phys.astype(np.float32),
                    fn_used=np.float32(fn_fix),
                    Qm_used=np.float32(Qm_ref))

# B) Qm boundcheck CSV
df_qm = pd.DataFrame({
    "t_sec":    t.astype(np.float32),
    "A_meas":   A_meas.astype(np.float32),
    "A_clean":  A_clean.astype(np.float32),
    "A_phys":   A_phys.astype(np.float32),
    "I_rms":    I_rms.astype(np.float32),
    "Qm_req":   Qm_req.astype(np.float32),
    "Qm_slow":  Qm_slow.astype(np.float32),
    "Qm_fast":  Qm_fast.astype(np.float32),
    "A_hat":    A_hat.astype(np.float32),
    "r_before": r_before.astype(np.float32),
    "r_after":  r_after.astype(np.float32),
})
df_qm.to_csv(os.path.join(DATA_DIR, "qm_boundcheck.csv"), index=False)

# C) Final merged CSV (for stepQm5)
df_merged = pd.DataFrame({
    "t_sec":        t.astype(np.float32),
    "A_meas":       A_meas.astype(np.float32),
    "A_clean":      A_clean.astype(np.float32),
    "I_rms":        I_rms.astype(np.float32),
    "f_inst":       f_inst.astype(np.float32),
    "G_res":        G_res.astype(np.float32),
    "A_phys":       A_phys.astype(np.float32),
    "Qm_req":       Qm_req.astype(np.float32),
    "Qm_slow":      Qm_slow.astype(np.float32),
    "PZT":          pzt_1k,
    "Tool_temp":    tool_temp_1k,
})
df_merged.to_csv(os.path.join(DATA_DIR, "merged_1k.csv"), index=False)
np.savez_compressed(os.path.join(DATA_DIR, "merged_1k.npz"),
                    t=t.astype(np.float32),
                    A_meas=A_meas.astype(np.float32),
                    A_clean=A_clean.astype(np.float32),
                    I_rms=I_rms.astype(np.float32),
                    f_inst=f_inst.astype(np.float32),
                    G_res=G_res.astype(np.float32),
                    A_phys=A_phys.astype(np.float32),
                    Qm_req=Qm_req.astype(np.float32),
                    Qm_slow=Qm_slow.astype(np.float32),
                    PZT=pzt_1k,
                    Tool_temp=tool_temp_1k)

# D) Metrics JSON
metrics_out = {
    "BASE": BASE, "N": int(N), "fs_ds": float(FS_DS),
    "sweep_json": SWEEP_JSON,
    "fn_fix_hz": round(fn_fix, 4), "fn_sweep_hz": round(fn_sweep, 4),
    "Qm_ref": round(Qm_ref, 6), "G_res_ref": round(G_res_ref, 8),
    "C_SCALE": C_SCALE, "TAU_S": TAU_S,
    "Qm_req_stats": {
        "min": round(float(Qm_req.min()), 4),
        "max": round(float(Qm_req.max()), 4),
        "mean": round(float(Qm_req.mean()), 4),
        "std": round(float(Qm_req.std()), 4),
    },
    "fast_ratio": round(fast_ratio, 4),
    "deriv_ratio": round(deriv_ratio, 4),
    "feasibility_full": {
        "rmse_before": round(rmse_bf, 6), "rmse_after": round(rmse_af, 6),
        "mae_before": round(mae_bf, 6),   "mae_after": round(mae_af, 6),
        "improve_pct": round(improve_full, 2),
    },
    "feasibility_test": {
        "rmse_before": round(rmse_bf_te, 6), "rmse_after": round(rmse_af_te, 6),
        "mae_before": round(mae_bf_te, 6),   "mae_after": round(mae_af_te, 6),
        "improve_pct": round(improve_test, 2),
    },
}
with open(os.path.join(DATA_DIR, "metrics.json"), "w") as f:
    json.dump(metrics_out, f, indent=2)

print(f"  → {DATA_DIR}")
print("\n" + "="*60)
print(f"✅  Done  |  BASE={BASE}")
print(f"   fn_fix      = {fn_fix:.3f} Hz")
print(f"   Qm_ref      = {Qm_ref:.4f}")
print(f"   fast_ratio  = {fast_ratio:.3f}  (< 0.3 = slow drift dominates)")
print(f"   FULL improve = {improve_full:.1f}%")
print(f"   TEST improve = {improve_test:.1f}%")
print(f"   merged_1k.csv ready for stepQm5 (MLP training)")
print("="*60)