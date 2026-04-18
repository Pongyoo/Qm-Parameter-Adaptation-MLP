# ============================================================
# phase1_period_analysis.py
# วิเคราะห์คาบของ periodic drop ใน A_meas signal
# ใช้ pipeline เดียวกับโค้ดที่มีอยู่:
#   bandpass 15-25kHz → rolling RMS → downsample 1000Hz
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')          # ไม่เด้ง popup — เซฟเป็นไฟล์อย่างเดียว
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, filtfilt

# ================= USER CONFIG =================
CSV_FILES = {
     "exp5_0.5V"   : r"C:\Users\ploy\Desktop\ML\GRU_2\processed_Qm_sessions\exp5_0.5_fix18.5\stepQm0_clean\0.5_fix18.5_exp5_clean.csv",
     "exp5_1V"   : r"C:\Users\ploy\Desktop\ML\GRU_2\processed_Qm_sessions\exp5_1_fix18.2\stepQm0_clean\1.0_fix18.2_exp5_clean.csv",
     "exp5_1.5V"   : r"C:\Users\ploy\Desktop\ML\GRU_2\processed_Qm_sessions\exp5_1.5_fix18.0\stepQm0_clean\1.5_fix18.0_exp5_clean.csv",
     "exp5_2V"   : r"C:\Users\ploy\Desktop\ML\GRU_2\processed_Qm_sessions\exp5_2_fix17.7\stepQm0_clean\2_fix17.7_exp5_clean.csv",
     "exp5_2.5V"   : r"C:\Users\ploy\Desktop\ML\GRU_2\processed_Qm_sessions\exp5_2.5_fix17.3\stepQm0_clean\2.5_fix17.3_exp5_clean.csv",
     "exp5_3V"   : r"C:\Users\ploy\Desktop\ML\GRU_2\processed_Qm_sessions\exp5_3_fix17.0\stepQm0_clean\3_fix17.0_exp5_clean.csv",
     "exp5_3.5V"   : r"C:\Users\ploy\Desktop\ML\GRU_2\processed_Qm_sessions\exp5_3.5_fix16.7\stepQm0_clean\3.5_fix16.7_exp5_clean.csv",
}

OUT_DIR   = r"C:\Users\ploy\Desktop\ML\GRU_2\processed_periodic_drop\Thisyear"

# Pipeline params — เหมือน code ที่มีอยู่เลย
FS_RAW    = 100_000
BPF_LO    = 15_000
BPF_HI    = 25_000
RMS_WIN_MS = 50        # ms
DS_FACTOR  = 100       # 100kHz → 1000Hz
FS_DS      = FS_RAW // DS_FACTOR   # 1000 Hz

# Column names
TIME_COL  = "Time"
DISP_COL  = "Displacement"

# ช่วงเวลาที่วิเคราะห์ (ตัด warm-up)
T_START   = 5.0    # วินาที
T_END     = None   # None = ทั้งหมด

# Peak detection สำหรับหา drop ใน A_meas
# ปรับถ้า detect ไม่ครบ
MIN_DISTANCE_SEC = 1.5   # วินาที — ระยะห่างขั้นต่ำระหว่าง drop
MIN_PROMINENCE   = None  # None = auto

# Smoothing window ก่อน detect — สำคัญมาก!
# ต้องใหญ่พอที่จะตัด ripple เล็กๆ ออก แต่ไม่ใหญ่จนกลืน drop
# ถ้า drop เกิดทุก ~3-5s → ลอง 0.5-1.0s
SMOOTH_SEC = 2   # วินาที — ปรับถ้าจุดยังผิดอยู่

# ถ้า data สั้นกว่านี้หลังตัด warm-up → ข้ามไฟล์ (วินาที)
MIN_DURATION_SEC = 10.0
# ============================================================


def butter_bandpass(x, fs, lo, hi, order=4):
    nyq  = 0.5 * fs
    b, a = butter(order, [lo/nyq, hi/nyq], btype='bandpass')
    return filtfilt(b, a, x)


def rolling_rms(x, win):
    """Rolling RMS — same logic as existing code"""
    x  = np.asarray(x, dtype=np.float64)
    cs = np.cumsum(np.insert(x*x, 0, 0.0))
    return np.sqrt((cs[win:] - cs[:-win]) / win)


def compute_A_meas(disp_raw, fs_raw=FS_RAW):
    """
    bandpass → rolling RMS → downsample
    คืนค่า (A_ds, t_offset_samples)
    t_offset คือจำนวน sample ที่หายไปจาก rolling RMS window
    """
    # detrend
    disp_det = disp_raw - np.mean(disp_raw)
    # bandpass
    disp_bp  = butter_bandpass(disp_det, fs_raw, BPF_LO, BPF_HI)
    # rolling RMS
    win      = max(1, int(round(RMS_WIN_MS * 1e-3 * fs_raw)))
    A_env    = rolling_rms(disp_bp, win)
    # downsample
    A_ds     = A_env[::DS_FACTOR]
    return A_ds, win - 1   # offset in raw samples


def analyze_one(name, path):
    print(f"\n[{name}]")
    if not os.path.isfile(path):
        print(f"  ❌ ไม่พบไฟล์: {path}")
        return None

    df = pd.read_csv(path)
    print(f"  rows={len(df):,}, cols={df.columns.tolist()}")

    if DISP_COL not in df.columns:
        print(f"  ❌ ไม่พบ column '{DISP_COL}'")
        return None

    # Time — แปลงให้เป็น relative seconds เริ่มจาก 0
    if TIME_COL in df.columns:
        t_raw = pd.to_numeric(df[TIME_COL], errors="coerce").values
        # ถ้า NaN เยอะ → reconstruct
        if np.isfinite(t_raw).sum() < 100:
            t_rel = np.arange(len(df), dtype=float) / FS_RAW
        else:
            t_rel = t_raw - t_raw[0]
            # เช็ค dt จาก 1000 rows แรก
            sample_diffs = np.diff(t_rel[1:min(1001, len(t_rel))])
            pos_diffs    = sample_diffs[sample_diffs > 0]
            dt = np.median(pos_diffs) if len(pos_diffs) > 0 else 0
            expected_dt  = 1.0 / FS_RAW   # 1e-5 s สำหรับ 100kHz

            print(f"  Time check: median dt={dt:.3e}s  "
                  f"(expected {expected_dt:.3e}s, ratio={dt/expected_dt:.1f}x)")

            if dt <= 0:
                print(f"  ⚠ dt ≤ 0 → reconstruct")
                t_rel = np.arange(len(df), dtype=float) / FS_RAW
            elif dt / expected_dt > 100:
                # น่าจะเป็น milliseconds → หาร 1000
                print(f"  ⚠ dt ใหญ่กว่า expected {dt/expected_dt:.0f}x "
                      f"→ ลอง interpret เป็น ms")
                t_rel = t_rel / 1000.0
                dt_new = dt / 1000.0
                if abs(dt_new / expected_dt - 1.0) > 5.0:
                    print(f"  ⚠ ยัง mismatch → reconstruct จาก FS_RAW")
                    t_rel = np.arange(len(df), dtype=float) / FS_RAW
            elif dt / expected_dt > 10:
                print(f"  ⚠ dt ใหญ่กว่า expected {dt/expected_dt:.1f}x "
                      f"→ reconstruct จาก FS_RAW")
                t_rel = np.arange(len(df), dtype=float) / FS_RAW
    else:
        print(f"  ⚠ ไม่พบ column '{TIME_COL}' → reconstruct จาก FS_RAW")
        t_rel = np.arange(len(df), dtype=float) / FS_RAW

    print(f"  t_rel range: 0 → {t_rel[-1]:.1f}s")

    # ตัด warm-up
    mask     = t_rel >= T_START
    if T_END is not None:
        mask &= t_rel <= T_END

    n_cut = mask.sum()
    duration_sec = n_cut / FS_RAW
    if duration_sec < MIN_DURATION_SEC:
        print(f"  ⚠ data หลังตัด T_START={T_START}s เหลือแค่ {duration_sec:.1f}s "
              f"(ต้องการ ≥ {MIN_DURATION_SEC}s) → ข้ามไฟล์นี้")
        print(f"  💡 ลอง ลด T_START หรือตรวจว่าไฟล์นี้มีข้อมูลครบไหม")
        return None

    disp_raw = pd.to_numeric(df[DISP_COL], errors="coerce") \
                 .fillna(0).values[mask]
    t_cut    = t_rel[mask] - T_START
    print(f"  duration={t_cut[-1]:.1f}s, samples={len(disp_raw):,}")

    # A_meas
    print("  Computing A_meas (bandpass → RMS → downsample)...")
    A_ds, offset = compute_A_meas(disp_raw)
    t_ds = np.linspace(0, t_cut[-1], len(A_ds))
    print(f"  A_meas length={len(A_ds)}, FS_DS={FS_DS}Hz")

    # === Smooth ก่อน find_peaks ===
    # ใช้ rolling mean เพื่อตัด ripple เล็กๆ ออก
    # เหลือแค่ slow drop ที่เป็นรอบๆ
    smooth_win = int(SMOOTH_SEC * FS_DS)
    smooth_win = max(3, smooth_win if smooth_win % 2 == 1 else smooth_win + 1)
    A_smooth = pd.Series(A_ds).rolling(smooth_win, center=True,
                                        min_periods=1).mean().values

    # หา troughs ใน smooth signal
    prom     = MIN_PROMINENCE if MIN_PROMINENCE else np.std(A_smooth) * 0.4
    min_dist = int(MIN_DISTANCE_SEC * FS_DS)
    troughs, _ = find_peaks(-A_smooth, distance=min_dist, prominence=prom)

    # ถ้าน้อยเกิน ลด prominence ลงเรื่อยๆ
    if len(troughs) < 3:
        for factor in [0.25, 0.15, 0.08]:
            prom_try = np.std(A_smooth) * factor
            troughs, _ = find_peaks(-A_smooth,
                                    distance=min_dist,
                                    prominence=prom_try)
            if len(troughs) >= 3:
                print(f"  ลด prominence → {prom_try:.5f}, พบ {len(troughs)} drops")
                break

    if len(troughs) < 2:
        print(f"  ❌ พบ drop น้อยเกินไป ({len(troughs)}) → ข้ามไฟล์นี้")
        return None

    trough_times = t_ds[troughs]
    periods      = np.diff(trough_times)
    cv           = np.std(periods) / np.mean(periods) * 100

    print(f"  ✅ drops={len(troughs)}, "
          f"period={np.mean(periods):.2f}±{np.std(periods):.2f}s "
          f"(CV={cv:.1f}%)")

    return {
        "name"         : name,
        "n_drops"      : len(troughs),
        "mean_period_s": np.mean(periods),
        "std_period_s" : np.std(periods),
        "cv_percent"   : cv,
        "periods"      : periods,
        "trough_times" : trough_times,
        "A_ds"         : A_ds,
        "A_smooth"     : A_smooth,
        "t_ds"         : t_ds,
        "troughs_idx"  : troughs,
    }


def plot_signals(results, out_dir):
    n   = len(results)
    fig, axes = plt.subplots(n, 1, figsize=(14, 3.5*n), sharex=False)
    if n == 1:
        axes = [axes]

    for ax, r in zip(axes, results):
        # Raw A_meas (faint)
        ax.plot(r["t_ds"], r["A_ds"],
                color='lightsteelblue', lw=0.6, alpha=0.7, label='A_meas raw')
        # Smoothed signal (ที่ใช้ detect)
        ax.plot(r["t_ds"], r["A_smooth"],
                color='steelblue', lw=1.5, label=f'A_meas smoothed ({SMOOTH_SEC}s window)')
        # Detected troughs บน smooth signal
        vals = r["A_smooth"][r["troughs_idx"]]
        ax.scatter(r["trough_times"], vals,
                   color='red', s=50, zorder=5,
                   label=f'Detected drops (n={r["n_drops"]})')
        ax.set_title(f"{r['name']}  —  "
                     f"period={r['mean_period_s']:.2f}±{r['std_period_s']:.2f}s  "
                     f"CV={r['cv_percent']:.1f}%")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("A_meas (µm)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    p = os.path.join(out_dir, "period_signals.png")
    plt.savefig(p, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"💾 บันทึกแล้ว → {p}")


def plot_comparison(results, out_dir):
    names = [r["name"] for r in results]
    means = [r["mean_period_s"] for r in results]
    stds  = [r["std_period_s"] for r in results]
    cvs   = [r["cv_percent"] for r in results]
    x     = np.arange(len(names))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.bar(x, means, yerr=stds, capsize=6,
            color='steelblue', edgecolor='white',
            error_kw=dict(elinewidth=2))
    ax1.axhline(np.mean(means), color='red', ls='--', lw=1.5,
                label=f'Overall mean={np.mean(means):.2f}s')
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=25, ha='right')
    ax1.set_ylabel("Mean Period (s)")
    ax1.set_title("Period ข้าม experiments\n(แท่งใกล้เคียงกัน = consistent)")
    ax1.legend(); ax1.grid(True, axis='y', alpha=0.3)

    colors = ['green' if c < 15 else 'orange' if c < 25 else 'red'
              for c in cvs]
    ax2.bar(x, cvs, color=colors, edgecolor='white')
    ax2.axhline(15, color='green', ls='--', lw=1.5, label='CV=15% (good)')
    ax2.axhline(25, color='orange', ls='--', lw=1.5, label='CV=25% (marginal)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=25, ha='right')
    ax2.set_ylabel("CV (%)")
    ax2.set_title("ความสม่ำเสมอของ period\n(เขียว=ดี, ส้ม=ปานกลาง, แดง=ไม่สม่ำเสมอ)")
    ax2.legend(); ax2.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    p = os.path.join(out_dir, "period_comparison.png")
    plt.savefig(p, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"💾 บันทึกแล้ว → {p}")


def save_period_data(results, out_dir):
    """Save period data ของแต่ละ experiment เป็น CSV"""
    rows = []
    for r in results:
        for i, p in enumerate(r["periods"]):
            rows.append({
                "experiment"   : r["name"],
                "drop_index"   : i,
                "period_s"     : p,
                "trough_time_s": r["trough_times"][i],
            })
    df_out = pd.DataFrame(rows)
    p = os.path.join(out_dir, "period_data.csv")
    df_out.to_csv(p, index=False)
    print(f"💾 Period data → {p}")


def print_summary(results):
    print("\n" + "="*68)
    print(f"{'Experiment':<22} {'Drops':>6} {'Mean(s)':>9} "
          f"{'STD(s)':>8} {'CV(%)':>7}")
    print("-"*68)
    for r in results:
        flag = ("✅" if r["cv_percent"] < 15
                else "⚠️ " if r["cv_percent"] < 25
                else "❌")
        print(f"{r['name']:<22} {r['n_drops']:>6} "
              f"{r['mean_period_s']:>9.2f} "
              f"{r['std_period_s']:>8.2f} "
              f"{r['cv_percent']:>7.1f}  {flag}")
    print("="*68)

    spread  = (max(r["mean_period_s"] for r in results) -
               min(r["mean_period_s"] for r in results))
    avg_cv  = np.mean([r["cv_percent"] for r in results])
    avg_per = np.mean([r["mean_period_s"] for r in results])

    print(f"\nSpread (max-min): {spread:.2f}s  |  Avg CV: {avg_cv:.1f}%")
    print()

    if spread < 1.5 and avg_cv < 15:
        print("✅ VERDICT: Period สม่ำเสมอ → Pattern น่า learnable ด้วย GRU")
        print(f"   Drop freq ≈ {1/avg_per:.3f} Hz  (period ≈ {avg_per:.1f}s)")
        print("   → ไป Track B: GRU residual branch")
    elif spread < 3.0 and avg_cv < 25:
        print("⚠️  VERDICT: ปานกลาง → ลอง GRU ได้ แต่ต้อง validate")
        print("   → เตรียม Track A ไว้ด้วย")
    else:
        print("❌ VERDICT: Period ไม่สม่ำเสมอ → GRU generalize ยาก")
        print("   → Track A: clean target (Savitzky-Golay / adaptive)")


# ============================================================
if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    results = []
    for name, path in CSV_FILES.items():
        r = analyze_one(name, path)
        if r:
            results.append(r)

    if not results:
        print("\n❌ ไม่มีผล → เช็ค path และ column names")
    else:
        print_summary(results)
        save_period_data(results, OUT_DIR)
        plot_signals(results, OUT_DIR)
        if len(results) > 1:
            plot_comparison(results, OUT_DIR)