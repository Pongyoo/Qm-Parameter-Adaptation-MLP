"""
stepQm1b_clean.py  —  Phase 0: Raw CSV → A_meas → A_clean
==========================================================
Pipeline:
  stepQm0 : โหลด raw CSV → select cols → drop NaN → parse Time → float seconds
  stepQm1 : Displacement → BPF → rolling RMS → downsample → A_meas @1kHz
  Phase 0 : A_meas → calibrate T̂ → MA(window=T̂) → A_clean

Diagnostic plots (4 อย่าง):
  diag1_raw_displacement.png      — raw Displacement ก่อน filter (zoom 0.2s)
  diag2_filtered_displacement.png — หลัง BPF (zoom 0.2s)
  diag3_A_meas.png                — RMS envelope @1kHz ทั้ง session
  diag4_A_clean.png               — A_meas vs A_clean overlay
"""

import os, json
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ══════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════
ROOT    = r"E:\raw_data\exp5(2026.3.27)"
BASE    = "1.5_exp5"
IN_CSV  = os.path.join(ROOT, "1.5_fix18.0_exp5.csv")

OUT_DIR  = os.path.join(ROOT, "1.5_fix18.0_processed_Qm_rms10", "stepQm1b_clean")
PLOT_DIR = os.path.join(OUT_DIR, "plots")
DATA_DIR = os.path.join(OUT_DIR, "data")

FS_RAW     = 100_000
BPF_LO     = 15_000
BPF_HI     = 25_000
FILTER_ORD = 4
RMS_WIN_MS = 10
DS_FACTOR  = 100       # → 1 kHz

CALIB_SEC  = 30.0
SMOOTH_SEC = 1.0
PEAK_PROM  = 0.005
CV_WARN    = 0.15

COL_TIME = "Time"
COL_DISP = "Displacement"
COL_CURR = "Current"
USE_COLS = ["Time", "Current", "Displacement", "PZT", "Tool_temp", "Voltage"]

# ══════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════

def parse_time_to_seconds(series):
    """
    แปลง Time column เป็น float seconds (relative จาก 0)
    รองรับ: float, mm:ss.f, datetime string
    """
    sample = str(series.iloc[0]).strip()

    def _is_float(s):
        try: float(s); return True
        except: return False

    if _is_float(sample):
        t = series.astype(float).values
        print(f"  Time format  : float seconds")
    elif ":" in sample and "/" not in sample and len(sample) < 12:
        def mmssf(s):
            p = str(s).strip().split(":")
            return float(p[0]) * 60.0 + float(p[1])
        t = series.apply(mmssf).values.astype(float)
        print(f"  Time format  : mm:ss.f")
    else:
        t_dt = pd.to_datetime(series, infer_datetime_format=True)
        t = (t_dt - t_dt.iloc[0]).dt.total_seconds().values
        print(f"  Time format  : datetime  origin={t_dt.iloc[0]}")

    t = t - t[0]   # relative จาก 0
    print(f"  t range      : {t[0]:.4f} – {t[-1]:.2f} s  (duration={t[-1]:.1f}s)")
    return t


def butter_bandpass(x, fs, lo, hi, order=4):
    nyq = 0.5 * fs
    lo_n, hi_n = lo / nyq, hi / nyq
    if not (0 < lo_n < hi_n < 1):
        raise ValueError(f"BPF range invalid: lo={lo}, hi={hi}, fs={fs}")
    b, a = butter(order, [lo_n, hi_n], btype="bandpass")
    return filtfilt(b, a, x)


def rolling_rms(x, win):
    x = np.asarray(x, dtype=np.float64)
    cs = np.cumsum(np.insert(x**2, 0, 0.0))
    return np.sqrt((cs[win:] - cs[:-win]) / win)


# ══════════════════════════════════════════════════════════════════
#  STEP 0 — Load & clean
# ══════════════════════════════════════════════════════════════════

def step0_load(path):
    print("\n" + "="*55)
    print("STEP 0 — Load raw CSV")

    df = pd.read_csv(path)
    print(f"  Loaded       : {df.shape[0]:,} rows × {df.shape[1]} cols")
    print(f"  Columns      : {df.columns.tolist()}")

    missing = [c for c in USE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    out = df[USE_COLS].copy()
    for c in USE_COLS:
        if c != COL_TIME:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    # parse Time → relative seconds (ทำก่อน drop NaN)
    t = parse_time_to_seconds(out[COL_TIME])
    out[COL_TIME] = t

    # drop NaN
    before = len(out)
    out = out.dropna(subset=[COL_DISP, COL_CURR, "PZT", "Tool_temp", "Voltage"])
    out = out.reset_index(drop=True)
    print(f"  Rows kept    : {len(out):,}/{before:,}")
    print(f"  Displacement : {out[COL_DISP].min():.5f} – {out[COL_DISP].max():.5f}")
    print(f"  Current      : {out[COL_CURR].min():.5f} – {out[COL_CURR].max():.5f}")

    return out


# ══════════════════════════════════════════════════════════════════
#  STEP 1 — Build A_meas  +  Diagnostic plots 1–3
# ══════════════════════════════════════════════════════════════════

def step1_build_Ameas(df, plot_dir):
    print("\n" + "="*55)
    print("STEP 1 — BPF → RMS → Downsample")

    t_raw   = df[COL_TIME].values.astype(np.float64)
    disp    = df[COL_DISP].values.astype(np.float64)
    current = df[COL_CURR].values.astype(np.float64)

    print(f"  n_samples    : {len(disp):,}")
    print(f"  disp raw     : min={disp.min():.5f}  max={disp.max():.5f}  std={disp.std():.5f}")

    # ── detrend ──
    disp    -= np.nanmean(disp)
    current -= np.nanmean(current)

    # ── DIAG 1: raw displacement zoom 0.2s ──
    n_zoom = min(int(0.2 * FS_RAW), len(disp))
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(t_raw[:n_zoom], disp[:n_zoom], lw=0.5, color="#E06C75")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Displacement (µm)")
    ax.set_title(f"DIAG 1 — Raw Displacement (first 0.2s)  |  std={disp.std():.5f}")
    ax.grid(True, alpha=0.3); plt.tight_layout()
    p = os.path.join(plot_dir, "diag1_raw_displacement.png")
    plt.savefig(p, dpi=150); plt.close()
    print(f"  [diag1] saved {p}")

    # ── BPF ──
    print(f"  BPF          : {BPF_LO/1000:.0f}–{BPF_HI/1000:.0f} kHz  order={FILTER_ORD}")
    disp_f    = butter_bandpass(disp,    FS_RAW, BPF_LO, BPF_HI, FILTER_ORD)
    current_f = butter_bandpass(current, FS_RAW, BPF_LO, BPF_HI, FILTER_ORD)
    print(f"  disp_f       : min={disp_f.min():.5f}  max={disp_f.max():.5f}  std={disp_f.std():.5f}")

    # ── DIAG 2: raw vs filtered zoom ──
    fig, axes = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
    axes[0].plot(t_raw[:n_zoom], disp[:n_zoom],   lw=0.5, color="#E06C75", alpha=0.7, label="raw")
    axes[0].plot(t_raw[:n_zoom], disp_f[:n_zoom], lw=0.8, color="#61AFEF", label="BPF")
    axes[0].set_ylabel("µm"); axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[0].set_title(f"DIAG 2 — Displacement raw vs BPF (first 0.2s)  |  disp_f std={disp_f.std():.5f}")
    axes[1].plot(t_raw[:n_zoom], current[:n_zoom],   lw=0.5, color="#E06C75", alpha=0.7, label="raw")
    axes[1].plot(t_raw[:n_zoom], current_f[:n_zoom], lw=0.8, color="#98C379", label="BPF")
    axes[1].set_ylabel("A"); axes[1].set_xlabel("Time (s)"); axes[1].legend(); axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    p = os.path.join(plot_dir, "diag2_filtered_displacement.png")
    plt.savefig(p, dpi=150); plt.close()
    print(f"  [diag2] saved {p}")

    # ── Rolling RMS ──
    RMS_WIN = int(FS_RAW * RMS_WIN_MS / 1000)
    print(f"  RMS window   : {RMS_WIN} samples ({RMS_WIN_MS} ms)")
    A_env = rolling_rms(disp_f,    RMS_WIN)
    I_env = rolling_rms(current_f, RMS_WIN)
    t_env = t_raw[RMS_WIN - 1:]
    print(f"  A_env        : min={A_env.min():.5f}  max={A_env.max():.5f}  mean={A_env.mean():.5f}")

    # ── Downsample ──
    A_meas = A_env[::DS_FACTOR]
    I_rms  = I_env[::DS_FACTOR]
    t_meas = t_env[::DS_FACTOR]
    fs_ds  = FS_RAW / DS_FACTOR
    print(f"  Downsample   : /{DS_FACTOR} → {fs_ds:.0f} Hz  |  n={len(A_meas):,}")
    print(f"  A_meas       : min={A_meas.min():.5f}  max={A_meas.max():.5f}  mean={A_meas.mean():.5f}")
    print(f"  t_meas range : {t_meas[0]:.3f} – {t_meas[-1]:.2f} s")

    # ── DIAG 3: A_meas full session ──
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    axes[0].plot(t_meas, A_meas, lw=0.7, color="#61AFEF")
    axes[0].set_ylabel("Amplitude (µm)")
    axes[0].set_title(f"DIAG 3 — A_meas full session  |  mean={A_meas.mean():.4f} µm")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(t_meas, I_rms, lw=0.7, color="#98C379")
    axes[1].set_ylabel("Current RMS (A)"); axes[1].set_xlabel("Time (s)")
    axes[1].set_title("I_rms full session")
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    p = os.path.join(plot_dir, "diag3_A_meas.png")
    plt.savefig(p, dpi=150); plt.close()
    print(f"  [diag3] saved {p}")

    return t_meas, A_meas, I_rms, fs_ds


# ══════════════════════════════════════════════════════════════════
#  PHASE 0 — T̂ calibration + MA  +  Diagnostic plot 4
# ══════════════════════════════════════════════════════════════════

def calibrate_T_hat_autocorr(A_meas, fs, calib_sec,
                              T_min=2.0, T_max=15.0):
    """
    หา T̂ ด้วย autocorrelation — ทนทานกว่า find_peaks
    เพราะไม่สนใจว่า drop แต่ละอันลึกแค่ไหน
    แค่หา dominant period ของสัญญาณทั้งหมด

    Algorithm:
      1. detrend A_meas (ตัด slow thermal drift ออก)
      2. autocorrelate
      3. หา peak แรกใน lag range [T_min, T_max]
      4. นั่นคือ T̂
    """
    n_calib = int(calib_sec * fs)
    A_seg   = A_meas[:n_calib].copy()

    # detrend: ลบ linear trend เพื่อไม่ให้ thermal drift รบกวน autocorr
    x = np.arange(len(A_seg), dtype=np.float64)
    p = np.polyfit(x, A_seg, 1)
    A_seg -= np.polyval(p, x)
    A_seg -= A_seg.mean()

    # autocorrelation
    corr = np.correlate(A_seg, A_seg, mode="full")
    corr = corr[len(corr) // 2:]      # lag >= 0
    corr /= corr[0]                   # normalize → max=1 at lag=0

    # หา peak ใน lag range [T_min, T_max]
    lag_min = int(T_min * fs)
    lag_max = min(int(T_max * fs), len(corr) - 1)
    corr_seg = corr[lag_min:lag_max]

    peaks, props = find_peaks(corr_seg, prominence=0.05)

    if len(peaks) == 0:
        # ลด prominence แล้วลองใหม่
        peaks, props = find_peaks(corr_seg, prominence=0.01)

    if len(peaks) == 0:
        print(f"  ⚠️  autocorr ไม่เจอ peak → ใช้ T̂ default = 6.25s")
        return 6.25, None

    # เลือก peak ที่มี prominence สูงสุด (dominant period)
    best = peaks[np.argmax(props["prominences"])]
    T_hat = (best + lag_min) / fs
    corr_val = corr_seg[best]
    print(f"  T̂ = {T_hat:.3f} s  (autocorr peak  corr={corr_val:.3f})")
    return T_hat, None


def phase0_clean(t, A_meas, fs, plot_dir):
    print("\n" + "="*55)
    print("PHASE 0 — Calibrate T̂ (autocorr) + MA → A_clean")

    # calibrate T̂ ด้วย autocorrelation
    T_hat, _ = calibrate_T_hat_autocorr(A_meas, fs, CALIB_SEC)

    # cross-check ด้วย find_peaks (เพื่อ report CV ให้รู้ว่า drop สม่ำเสมอแค่ไหน)
    n_calib = int(CALIB_SEC * fs)
    A_seg   = A_meas[:n_calib]
    sw      = max(int(SMOOTH_SEC * fs), 1)
    A_sm    = pd.Series(A_seg).rolling(sw, center=True, min_periods=1).mean().values
    min_dist = int(T_hat * fs * 0.5)
    valleys, _ = find_peaks(-A_sm, distance=min_dist, prominence=PEAK_PROM)
    if len(valleys) < 2:
        valleys, _ = find_peaks(-A_sm, distance=min_dist, prominence=PEAK_PROM * 0.3)
    if len(valleys) >= 2:
        intervals = np.diff(valleys) / fs
        cv = float(np.std(intervals) / np.mean(intervals))
        print(f"  CV (find_peaks cross-check) = {cv*100:.1f}%  ({len(valleys)} valleys)")
        if cv > CV_WARN:
            print(f"  ⚠️  CV > {CV_WARN*100:.0f}% — drop ไม่สม่ำเสมอ (autocorr ยังใช้ได้)")
    else:
        cv = None
        print(f"  CV : ไม่สามารถ cross-check ได้ (valleys < 2)")

    # apply MA
    window  = max(int(round(T_hat * fs)), 1)
    A_clean = pd.Series(A_meas).rolling(window, center=True, min_periods=1).mean().values
    print(f"  MA window    : {window} samples ({T_hat:.3f} s)")

    # ── DIAG 4: A_meas vs A_clean ──
    cv_str = f"  CV={cv*100:.1f}%" if cv is not None else ""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    axes[0].plot(t, A_meas, lw=0.5, color="#E06C75", alpha=0.9, label="A_meas")
    axes[0].set_ylabel("Amplitude (µm)")
    axes[0].set_title(f"DIAG 4 — A_meas  |  T̂={T_hat:.3f}s{cv_str}")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, A_meas,  lw=0.4, color="#E06C75", alpha=0.35, label="A_meas")
    axes[1].plot(t, A_clean, lw=1.2, color="#61AFEF", label="A_clean (MA)")
    axes[1].set_ylabel("Amplitude (µm)"); axes[1].set_xlabel("Time (s)")
    axes[1].set_title("A_clean — drop removed")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    p = os.path.join(plot_dir, "diag4_A_clean.png")
    plt.savefig(p, dpi=150); plt.close()
    print(f"  [diag4] saved {p}")

    return A_clean, T_hat, cv


# ══════════════════════════════════════════════════════════════════
#  SAVE
# ══════════════════════════════════════════════════════════════════

def save_outputs(t, A_meas, A_clean, I_rms, T_hat, cv, fs):
    os.makedirs(DATA_DIR, exist_ok=True)
    np.save(os.path.join(DATA_DIR, "t.npy"),       t.astype(np.float32))
    np.save(os.path.join(DATA_DIR, "A_meas.npy"),  A_meas.astype(np.float32))
    np.save(os.path.join(DATA_DIR, "A_clean.npy"), A_clean.astype(np.float32))
    np.save(os.path.join(DATA_DIR, "I_rms.npy"),   I_rms.astype(np.float32))
    meta = {
        "base": BASE, "fs_hz": fs,
        "n_samples": int(len(t)), "duration_s": round(float(t[-1]), 3),
        "T_hat_s": round(T_hat, 5),
        "CV": round(cv, 5) if cv is not None else None,
        "bpf_lo_hz": BPF_LO, "bpf_hi_hz": BPF_HI,
        "rms_win_ms": RMS_WIN_MS, "ds_factor": DS_FACTOR,
    }
    with open(os.path.join(DATA_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n  Saved → {DATA_DIR}")


# ══════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    os.makedirs(PLOT_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    df                       = step0_load(IN_CSV)
    t, A_meas, I_rms, fs_ds  = step1_build_Ameas(df, PLOT_DIR)
    A_clean, T_hat, cv       = phase0_clean(t, A_meas, fs_ds, PLOT_DIR)
    save_outputs(t, A_meas, A_clean, I_rms, T_hat, cv, fs_ds)

    print("\n" + "="*55)
    print("✅  Done")
    print(f"   t_meas  : {t[0]:.3f} – {t[-1]:.2f} s")
    print(f"   A_meas  : mean={A_meas.mean():.4f}  max={A_meas.max():.4f} µm")
    print(f"   A_clean : mean={A_clean.mean():.4f}  max={A_clean.max():.4f} µm")
    print(f"   T̂       : {T_hat:.3f} s" + (f"   CV={cv*100:.1f}%" if cv else ""))
    print(f"   outputs : {OUT_DIR}")