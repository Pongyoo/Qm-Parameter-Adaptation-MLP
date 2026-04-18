# ============================================================
# stepQm3_build_residual_y.py
# Step 3: Alignment + y-space (log-ratio) residual for feasibility check
#  - Build y_meas, y_phys using log(A[t+H]/A[t]) normalized by A_ref
#  - Residual: r_y = y_meas - y_phys
#  - Save CSV + plots
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

EPS = 1e-8

# ================= USER CONFIG =================
ROOT = r"C:\Users\ploy\Desktop\ML\GRU_2"
BASE = "2.5nl-fix17.37"

IN_MEAS = os.path.join(ROOT, "processed_Qm", "stepQm1_meas", "data", "A_meas_1k.csv")
IN_PHYS = os.path.join(ROOT, "processed_Qm", "stepQm2_phys", "data", "A_phys_1k.csv")

OUT_DIR  = os.path.join(ROOT, "processed_Qm", "stepQm3_residual_y")
PLOT_DIR = os.path.join(OUT_DIR, "plots")
DATA_DIR = os.path.join(OUT_DIR, "data")

H = 5000       # <-- ใช้ค่าเดิมที่เคยใช้
REF_N = 1000   # median over first REF_N samples of A (before shift)
# ============================================================

os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

def detect_time_col(df):
    for c in ["t_sec"]:
        if c in df.columns:
            return c
    return None

def compute_y_logratio(A, H, ref_n=1000):
    A = A.astype(np.float64)
    if len(A) <= H + 1:
        raise ValueError(f"Signal too short for H={H}. len(A)={len(A)}")

    A0 = A[:-H]
    AH = A[H:]
    y0 = np.log((AH + EPS) / (A0 + EPS))

    if len(A0) >= ref_n:
        A_ref = float(np.median(A0[:ref_n]))
    else:
        A_ref = float(np.median(A0))

    y0 = y0 / (A_ref + EPS)
    return y0.astype(np.float32), float(A_ref)

print("\n================ StepQm3: y-space Residual Analysis ================")
print(f"BASE={BASE} | H={H} | REF_N={REF_N}")
print("IN_MEAS:", IN_MEAS)
print("IN_PHYS:", IN_PHYS)

df_meas = pd.read_csv(IN_MEAS)
df_phys = pd.read_csv(IN_PHYS)

tcol_m = detect_time_col(df_meas)
tcol_p = detect_time_col(df_phys)
if tcol_m is None or tcol_p is None:
    raise KeyError(f"Missing time column. meas cols={list(df_meas.columns)} | phys cols={list(df_phys.columns)}")

# รองรับชื่อ amplitude หลายแบบ
Acol_m = "A_meas" if "A_meas" in df_meas.columns else df_meas.columns[-1]
Acol_p = "A_phys" if "A_phys" in df_phys.columns else df_phys.columns[-1]

t_m = df_meas[tcol_m].to_numpy(dtype=np.float64)
t_p = df_phys[tcol_p].to_numpy(dtype=np.float64)
A_m = df_meas[Acol_m].to_numpy(dtype=np.float64)
A_p = df_phys[Acol_p].to_numpy(dtype=np.float64)

# -------- align length (simple + safe) --------
N = min(len(A_m), len(A_p), len(t_m), len(t_p))
t = t_m[:N]  # ใช้ time ของ meas เป็นหลัก
A_m = A_m[:N]
A_p = A_p[:N]

# -------- build y-space series --------
y_meas, Aref_m = compute_y_logratio(A_m, H=H, ref_n=REF_N)
y_phys, Aref_p = compute_y_logratio(A_p, H=H, ref_n=REF_N)

# time for y is t[0:len(y)] corresponding to A[t] used as denominator
t_y = t[:len(y_meas)]

r_y = (y_meas - y_phys).astype(np.float32)

print("\nA_ref:")
print(f"  A_ref_meas = {Aref_m:.6f}")
print(f"  A_ref_phys = {Aref_p:.6f}")

print("\ny stats (meas/phys):")
print(f"  y_meas len={len(y_meas)} min={y_meas.min():.6f} max={y_meas.max():.6f} mean={y_meas.mean():.6f} std={y_meas.std():.6f}")
print(f"  y_phys len={len(y_phys)} min={y_phys.min():.6f} max={y_phys.max():.6f} mean={y_phys.mean():.6f} std={y_phys.std():.6f}")

print("\nResidual r_y stats:")
print(f"  len={len(r_y)} min={r_y.min():.6f} max={r_y.max():.6f} mean={r_y.mean():.6f} std={r_y.std():.6f}")

# -------- save csv --------
out_csv = os.path.join(DATA_DIR, "y_residual_1k.csv")
pd.DataFrame({
    "t": t_y.astype(np.float32),
    "y_meas": y_meas.astype(np.float32),
    "y_phys": y_phys.astype(np.float32),
    "r_y": r_y.astype(np.float32),
}).to_csv(out_csv, index=False)
print("\nCSV saved:", out_csv)

# -------- plots --------
# 1) y overlay
plt.figure(figsize=(11, 4))
plt.plot(t_y, y_meas, label="y_meas", linewidth=1.1)
plt.plot(t_y, y_phys, label="y_phys", linewidth=1.1, alpha=0.85)
plt.title(f"{BASE} | y_meas vs y_phys (H={H})")
plt.xlabel("Time (s)")
plt.ylabel("y (log-ratio, normalized)")
plt.legend()
plt.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, f"{BASE}_y_overlay.png"), dpi=160)
plt.close()

# 2) residual vs time
plt.figure(figsize=(11, 4))
plt.plot(t_y, r_y, linewidth=1.1)
plt.title(f"{BASE} | residual r_y = y_meas - y_phys")
plt.xlabel("Time (s)")
plt.ylabel("r_y")
plt.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, f"{BASE}_r_y_time.png"), dpi=160)
plt.close()

# 3) histogram
plt.figure(figsize=(7, 4.5))
plt.hist(r_y, bins=60)
plt.title(f"{BASE} | r_y histogram")
plt.xlabel("r_y")
plt.ylabel("Count")
plt.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, f"{BASE}_r_y_hist.png"), dpi=160)
plt.close()

# 4) scatter (r_y vs y_phys)
plt.figure(figsize=(6, 6))
plt.scatter(y_phys, r_y, s=8, alpha=0.6)
plt.title(f"{BASE} | r_y vs y_phys")
plt.xlabel("y_phys")
plt.ylabel("r_y")
plt.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, f"{BASE}_r_y_scatter.png"), dpi=160)
plt.close()

print("\nPlots saved to:", PLOT_DIR)
print("✅ StepQm3 (y-space) done.")