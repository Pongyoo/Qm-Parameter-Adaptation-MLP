# ============================================================
# stepQm2_build_Aphys.py
# Step 2: Build physics amplitude A_phys(t) for FIX (1kHz aligned)
#
# Concept:
#   A_phys(t) = C * I_rms(t) * G_res(f_inst(t); fn_fix, Qm)
#   - I_rms: RMS envelope of bandpassed Current
#   - f_inst: Welch-peak tracking (robust) in 15-25 kHz band
#   - fn_fix: median f_inst in chosen steady region (from FIX itself)
#   - Qm: loaded from sweep params (reference)
#
# Outputs:
#   processed_Qm/stepQm2_phys/data/A_phys_1k.csv
#   processed_Qm/stepQm2_phys/data/A_phys_1k.npz
#   processed_Qm/stepQm2_phys/plots/*.png
# ============================================================

import os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, welch

# ================= USER CONFIG =================
ROOT = r"C:\Users\ploy\Desktop\ML\GRU_2"
BASE_FIX   = "2.5nl-fix17.37"
SWEEP_NAME = "2.5nl-swp"   # used only for loading Qm/fn reference json

CLEAN_FIX_PATH = os.path.join(ROOT, "processed", "01_clean_fix", f"{BASE_FIX}_clean.csv")
SWEEP_PARAM_PATH = os.path.join(ROOT, "processed_hybrid", "00_params_sweep", f"params_{SWEEP_NAME}.json")

OUT_DIR  = os.path.join(ROOT, "processed_Qm", "stepQm2_phys")
PLOT_DIR = os.path.join(OUT_DIR, "plots")
DATA_DIR = os.path.join(OUT_DIR, "data")

FS_RAW = 100_000
DS_FACTOR = 100
FS_DS = FS_RAW // DS_FACTOR  # 1000 Hz

# Bandpass for ultrasonic carrier (Current)
BPF_LO = 15_000.0
BPF_HI = 25_000.0
FILTER_ORDER = 4

# Welch tracking (robust)
FTRACK_WIN_SEC = 0.20
FTRACK_HOP_SEC = 0.05
FTRACK_FMIN = 15_000.0
FTRACK_FMAX = 25_000.0

# Choose region to estimate fn_fix from FIX (seconds)
FN_EST_T0 = 1.0
FN_EST_T1 = 6.0

# RMS envelope on bandpassed current (ms)
RMS_WIN_MS = 50

# scale constant (keep 1.0 for now)
C_SCALE = 1.0

EPS = 1e-12
# =================================================

os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

def get_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def butter_bandpass(x, fs, lo, hi, order=4):
    nyq = 0.5 * fs
    lo_n = lo / nyq
    hi_n = hi / nyq
    b, a = butter(order, [lo_n, hi_n], btype="bandpass")
    return filtfilt(b, a, x)

def rolling_rms_valid(x, win):
    x = x.astype(np.float64)
    if len(x) < win:
        return None
    x2 = x * x
    cs = np.cumsum(np.insert(x2, 0, 0.0))
    wsum = cs[win:] - cs[:-win]
    return np.sqrt(wsum / win)

def estimate_fs_from_time(t):
    dt = np.diff(t)
    dt = dt[np.isfinite(dt)]
    if len(dt) < 10:
        return None
    dt_med = np.median(dt)
    if dt_med <= 0:
        return None
    return 1.0 / dt_med

def resonance_gain(f, fn, Qm):
    r = f / (fn + EPS)
    zeta = 1.0 / (2.0 * (Qm + EPS))
    return 1.0 / np.sqrt((1.0 - r**2)**2 + (2.0 * zeta * r)**2 + EPS)

def peak_freq_welch(x_seg, fs, fmin, fmax):
    f, Pxx = welch(x_seg, fs=fs, nperseg=min(len(x_seg), 4096))
    m = (f >= fmin) & (f <= fmax)
    if not np.any(m):
        return np.nan
    f2 = f[m]
    p2 = Pxx[m]
    if (not np.all(np.isfinite(p2))) or len(p2) < 3:
        return np.nan
    return float(f2[np.argmax(p2)])

def track_frequency_welch(x_bp, t, fs, win_sec, hop_sec, fmin, fmax):
    win = int(win_sec * fs)
    hop = int(hop_sec * fs)
    win = max(win, 1024)
    hop = max(hop, 1)

    f_list, tc_list = [], []
    idx0, N = 0, len(x_bp)
    while idx0 + win <= N:
        seg = x_bp[idx0:idx0+win]
        fpk = peak_freq_welch(seg, fs, fmin, fmax)
        tc  = float(t[idx0 + win//2])
        f_list.append(fpk)
        tc_list.append(tc)
        idx0 += hop

    return np.array(tc_list, dtype=np.float64), np.array(f_list, dtype=np.float64)

def quick_stats(name, x):
    x = np.asarray(x, dtype=np.float64)
    print(f"[{name}] len={len(x)} min={np.nanmin(x):.6g} max={np.nanmax(x):.6g} mean={np.nanmean(x):.6g} std={np.nanstd(x):.6g}")

def main():
    if not os.path.exists(CLEAN_FIX_PATH):
        raise FileNotFoundError(CLEAN_FIX_PATH)

    df = pd.read_csv(CLEAN_FIX_PATH)

    col_t = get_col(df, ["Time", "t_sec", "time", "t"])
    col_i = get_col(df, ["Current", "current", "I", "i"])

    if col_i is None:
        raise ValueError(f"Missing Current column in FIX clean: columns={list(df.columns)}")

    cur = pd.to_numeric(df[col_i], errors="coerce").to_numpy(dtype=np.float64)
    m = np.isfinite(cur)
    cur = cur[m]

    # time + fs
    fs = float(FS_RAW)
    if col_t is not None:
        t_try = pd.to_numeric(df[col_t], errors="coerce").to_numpy(dtype=np.float64)[m]
        finite_ratio = np.isfinite(t_try).mean() if len(t_try) else 0.0
        print("Time sanity:")
        print("  finite_ratio =", float(finite_ratio))
        if finite_ratio > 0.8:
            fs_est = estimate_fs_from_time(t_try)
            if fs_est is not None and 1000 < fs_est < 1e6:
                fs = float(fs_est)
                t = t_try
            else:
                t = np.arange(len(cur), dtype=np.float64) / fs
        else:
            print("⚠️ Time column looks bad. Rebuild t from FS_RAW.")
            t = np.arange(len(cur), dtype=np.float64) / fs
    else:
        t = np.arange(len(cur), dtype=np.float64) / fs

    # load sweep params for Qm (reference)
    Qm = None
    fn_sweep = None
    if os.path.exists(SWEEP_PARAM_PATH):
        with open(SWEEP_PARAM_PATH, "r", encoding="utf-8") as f:
            params = json.load(f)
        if "Qm" in params:
            Qm = float(params["Qm"])
        if "fn_hz" in params:
            fn_sweep = float(params["fn_hz"])

    if Qm is None:
        Qm = 35.0
        print("⚠️ Sweep params not found or missing Qm. Use default Qm=35.")

    print("\n================ StepQm2: Build A_phys (FIX) ================")
    print("BASE_FIX:", BASE_FIX)
    print("CLEAN_FIX_PATH:", CLEAN_FIX_PATH)
    print("SWEEP_PARAM_PATH:", SWEEP_PARAM_PATH)
    print("FS:", fs, "| DS_FACTOR:", DS_FACTOR, "| FS_DS:", FS_DS)
    print("BPF:", BPF_LO, "-", BPF_HI, "Hz")
    print("FTRACK win/hop:", FTRACK_WIN_SEC, "/", FTRACK_HOP_SEC, "sec")
    print("FN_EST region:", FN_EST_T0, "->", FN_EST_T1, "sec")
    print("Qm_used:", Qm, "| fn_sweep_ref:", fn_sweep)
    print("============================================================\n")

    # 1) bandpass current
    cur0 = cur - float(np.mean(cur))
    cur_bp = butter_bandpass(cur0, fs, BPF_LO, BPF_HI, order=FILTER_ORDER)

    quick_stats("Current(raw)", cur)
    quick_stats("Current(bandpass)", cur_bp)

    # plot current snippet (20 ms)
    Nshow = int(min(len(cur_bp), 0.02 * fs))
    plt.figure(figsize=(10,4))
    plt.plot(t[:Nshow], cur_bp[:Nshow], lw=1.0)
    plt.xlabel("Time (s)")
    plt.ylabel("Current (A)  (bandpassed)")
    plt.title(f"{BASE_FIX} | Current bandpass {BPF_LO/1000:.1f}-{BPF_HI/1000:.1f} kHz")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    p1 = os.path.join(PLOT_DIR, f"{BASE_FIX}_cur_bp.png")
    plt.savefig(p1, dpi=160)
    plt.close()

    # 2) Welch frequency tracking
    t_fk, f_pk = track_frequency_welch(
        cur_bp, t, fs,
        win_sec=FTRACK_WIN_SEC,
        hop_sec=FTRACK_HOP_SEC,
        fmin=FTRACK_FMIN, fmax=FTRACK_FMAX
    )

    # estimate fn_fix
    region = (t_fk >= FN_EST_T0) & (t_fk <= FN_EST_T1) & np.isfinite(f_pk)
    if np.sum(region) < 5:
        fn_fix = float(np.nanmedian(f_pk))
        print("⚠️ Not enough points in FN_EST region; fallback fn_fix = median(all valid peaks).")
    else:
        fn_fix = float(np.median(f_pk[region]))

    print(f"Estimated fn_fix = {fn_fix:.3f} Hz (from FIX)")
    quick_stats("f_peak", f_pk)

    # plot f tracking
    plt.figure(figsize=(10,4))
    plt.plot(t_fk, f_pk, lw=1.0, label="f_peak (Welch)")
    plt.axhline(fn_fix, linestyle="--", label=f"fn_fix={fn_fix:.1f} Hz")
    if fn_sweep is not None:
        plt.axhline(fn_sweep, linestyle=":", label=f"fn_sweep(ref)={fn_sweep:.1f} Hz")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title(f"{BASE_FIX} | Welch peak tracking")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    p2 = os.path.join(PLOT_DIR, f"{BASE_FIX}_f_track.png")
    plt.savefig(p2, dpi=160)
    plt.close()

    # 3) I_rms envelope (on bandpassed current)
    rms_win = int(fs * RMS_WIN_MS / 1000.0)
    rms_win = max(10, rms_win)
    I_rms = rolling_rms_valid(cur_bp, rms_win)
    if I_rms is None:
        raise ValueError("Signal too short for RMS.")
    t_rms = t[rms_win - 1:]

    # downsample to 1kHz (match StepQm1 style)
    I_rms_1k = I_rms[::DS_FACTOR]
    t_1k = t_rms[::DS_FACTOR]

    # interpolate f_inst to t_1k
    f_inst_1k = np.interp(t_1k, t_fk, np.nan_to_num(f_pk, nan=fn_fix))
    f_inst_1k = np.clip(f_inst_1k, FTRACK_FMIN, FTRACK_FMAX)

    # 4) G_res and A_phys
    G_res_1k = resonance_gain(f_inst_1k, fn=fn_fix, Qm=Qm)
    A_phys_1k = (C_SCALE * I_rms_1k * G_res_1k).astype(np.float64)

    quick_stats("I_rms_1k", I_rms_1k)
    quick_stats("G_res_1k", G_res_1k)
    quick_stats("A_phys_1k", A_phys_1k)
    print("t_1k min/max =", float(t_1k[0]), float(t_1k[-1]))

    # save csv + npz
    out_df = pd.DataFrame({
        "t_sec": t_1k.astype(np.float64),
        "I_rms": I_rms_1k.astype(np.float64),
        "f_inst": f_inst_1k.astype(np.float64),
        "G_res": G_res_1k.astype(np.float64),
        "A_phys": A_phys_1k.astype(np.float64),
        "fn_used": np.full(len(t_1k), fn_fix, dtype=np.float64),
        "Qm_used": np.full(len(t_1k), Qm, dtype=np.float64),
        "fn_sweep_ref": np.full(len(t_1k), fn_sweep if fn_sweep is not None else np.nan, dtype=np.float64),
    })

    csv_path = os.path.join(DATA_DIR, "A_phys_1k.csv")
    out_df.to_csv(csv_path, index=False)

    npz_path = os.path.join(DATA_DIR, "A_phys_1k.npz")
    np.savez_compressed(
        npz_path,
        t=out_df["t_sec"].to_numpy(dtype=np.float32),
        I_rms=out_df["I_rms"].to_numpy(dtype=np.float32),
        f_inst=out_df["f_inst"].to_numpy(dtype=np.float32),
        G_res=out_df["G_res"].to_numpy(dtype=np.float32),
        A_phys=out_df["A_phys"].to_numpy(dtype=np.float32),
        fn_used=np.float32(fn_fix),
        Qm_used=np.float32(Qm),
        fn_sweep_ref=np.float32(fn_sweep) if fn_sweep is not None else np.float32(np.nan),
    )

    # plot G_res
    plt.figure(figsize=(10,4))
    plt.plot(out_df["t_sec"], out_df["G_res"], lw=1.0)
    plt.xlabel("Time (s)")
    plt.ylabel("G_res (arb.)")
    plt.title(f"{BASE_FIX} | G_res (fn_fix, Qm_ref)")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    p3 = os.path.join(PLOT_DIR, f"{BASE_FIX}_G_res.png")
    plt.savefig(p3, dpi=160)
    plt.close()

    # plot A_phys
    plt.figure(figsize=(10,4))
    plt.plot(out_df["t_sec"], out_df["A_phys"], lw=1.0)
    plt.xlabel("Time (s)")
    plt.ylabel("A_phys (arb.)")
    plt.title(f"{BASE_FIX} | A_phys = I_rms * G_res")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    p4 = os.path.join(PLOT_DIR, f"{BASE_FIX}_A_phys.png")
    plt.savefig(p4, dpi=160)
    plt.close()

    print("\n✅ StepQm2 done.")
    print("Saved to:", OUT_DIR)
    print("CSV saved:", csv_path)
    print("NPZ saved:", npz_path)
    print("Plots:")
    print(" ", p1)
    print(" ", p2)
    print(" ", p3)
    print(" ", p4)

if __name__ == "__main__":
    main()
