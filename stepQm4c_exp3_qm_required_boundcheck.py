# ============================================================
# stepQm4c_exp3_qm_required_boundcheck.py
# Step 4c: Qm-required bound check (no training)
#
# Purpose:
#   - Compute Qm_required(t) from A_meas and I_rms
#   - Separate "slow" vs "fast" part using EMA
#   - Quantify whether required Qm changes too fast
#   - Rebuild A_hat using Qm_slow only
#   - Evaluate y-space improvement:
#       (1) full-sequence
#       (2) test-only segment (last 30%, same split style as Step 5/6)
#
# Input:
#   processed_Qm/stepQm1_meas/data/A_meas_1k.csv
#   processed_Qm/stepQm2_phys/data/A_phys_1k.csv
#
# Output:
#   processed_Qm/stepQm4c_boundcheck/data/<BASE>_qm_required_boundcheck.csv
#   processed_Qm/stepQm4c_boundcheck/data/<BASE>_stepQm4c_metrics.json
#   processed_Qm/stepQm4c_boundcheck/plots/*.png
#
# Notes:
#   - This is a feasibility / upper-bound style check
#   - No learning is involved here
#   - "test-only" means last 30% of timeline, consistent with Step 5/6
# ============================================================

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ================= USER CONFIG =================
ROOT = r"C:\Users\ploy\Desktop" 
BASE = "3nl-fix17.37_exp1"

SESSION_TAG = "exp1_3nl_fix17.37"
SESSION_ROOT = os.path.join(ROOT, "ML", "GRU_2", "processed_Qm_sessions", SESSION_TAG)

MEAS_CSV = os.path.join(SESSION_ROOT, "stepQm1_meas", "data", "A_meas_1k.csv")
PHYS_CSV = os.path.join(SESSION_ROOT, "stepQm2_phys", "data", "A_phys_1k.csv")

# Qm_ref from sweep
Qm_ref = 19.101265822784807

OUT_DIR = os.path.join(SESSION_ROOT, "stepQm4c_boundcheck")
PLOT_DIR = os.path.join(OUT_DIR, "plots")
DATA_DIR = os.path.join(OUT_DIR, "data")

# EMA smoothing time constant (seconds)
TAU_S = 2.0

# Reference length for y-space
REF_N = 1000

# Same split philosophy as Step 5/6
TRAIN_RATIO = 0.70

EPS = 1e-12
# =================================================

os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

def pick_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def ema(x, alpha):
    y = np.empty_like(x, dtype=float)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = alpha * x[i] + (1 - alpha) * y[i-1]
    return y

def safe_align(a, b):
    n = min(len(a), len(b))
    return a[:n], b[:n], n

def rmse(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.sqrt(np.mean((a - b) ** 2)))

def mae(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.mean(np.abs(a - b)))

# ---------------- load ----------------
df_m = pd.read_csv(MEAS_CSV)
df_p = pd.read_csv(PHYS_CSV)

tcol_m = pick_col(df_m, ["t_sec", "t", "Time"])
tcol_p = pick_col(df_p, ["t_sec", "t", "Time"])

if tcol_m is None or tcol_p is None:
    raise KeyError("Cannot find time column (expected t_sec/t/Time) in one of CSVs.")

Acol_m = pick_col(df_m, ["A_meas", "A_rms", "A"])
Acol_p = pick_col(df_p, ["A_phys", "A_phys_1k", "A"])

if Acol_m is None:
    raise KeyError("Cannot find A_meas column in MEAS CSV.")
if Acol_p is None:
    raise KeyError("Cannot find A_phys column in PHYS CSV.")

Icol = pick_col(df_p, ["I_rms", "I_rms_1k", "I_RMS"])
if Icol is None:
    raise KeyError("Cannot find I_rms column in PHYS CSV.")

t_m = df_m[tcol_m].to_numpy(dtype=float)
A_meas = df_m[Acol_m].to_numpy(dtype=float)

t_p = df_p[tcol_p].to_numpy(dtype=float)
A_phys = df_p[Acol_p].to_numpy(dtype=float)
I_rms = df_p[Icol].to_numpy(dtype=float)

# ---------------- align by length ----------------
A_meas, A_phys, _ = safe_align(A_meas, A_phys)
t_m, t_p, _ = safe_align(t_m, t_p)
I_rms = I_rms[:min(len(I_rms), len(A_meas))]
N = min(len(A_meas), len(A_phys), len(I_rms), len(t_m), len(t_p))
A_meas, A_phys, I_rms, t = A_meas[:N], A_phys[:N], I_rms[:N], t_m[:N]

# ---------------- basic checks ----------------
fs_ds = 1.0 / np.median(np.diff(t))
dt = 1.0 / fs_ds

I_rms_safe = np.maximum(np.abs(I_rms), EPS)

# ---------------- compute G_res_ref from original phys ----------------
G_res_series = A_phys / I_rms_safe
G_res_ref = float(np.median(G_res_series))

# ---------------- compute required Qm(t) from measured amplitude ----------------
G_req = A_meas / I_rms_safe
Qm_req = Qm_ref * (G_req / np.maximum(G_res_ref, EPS))

# ---------------- slow/fast split ----------------
alpha = dt / (TAU_S + dt)
Qm_slow = ema(Qm_req, alpha=alpha)
Qm_fast = Qm_req - Qm_slow

# ---------------- quantify "too-fast" behavior ----------------
dQ = np.diff(Qm_req) / dt
dQ_slow = np.diff(Qm_slow) / dt

rms_total = np.sqrt(np.mean((Qm_req - np.mean(Qm_req)) ** 2))
rms_fast = np.sqrt(np.mean((Qm_fast - np.mean(Qm_fast)) ** 2))
fast_ratio = float(rms_fast / (rms_total + EPS))

deriv_ratio = float((np.mean(np.abs(dQ_slow)) + EPS) / (np.mean(np.abs(dQ)) + EPS))

# ---------------- rebuild A_hat using Qm_slow only ----------------
G_hat = G_res_ref * (Qm_slow / Qm_ref)
A_hat = I_rms_safe * G_hat

# ---------------- evaluate in y-space (FULL) ----------------
A_ref_meas = float(np.median(A_meas[:min(REF_N, len(A_meas))]))
A_ref_hat  = float(np.median(A_hat[:min(REF_N, len(A_hat))]))
A_ref_phys = float(np.median(A_phys[:min(REF_N, len(A_phys))]))

y_meas = np.log(np.maximum(A_meas, EPS) / np.maximum(A_ref_meas, EPS))
y_phys = np.log(np.maximum(A_phys, EPS) / np.maximum(A_ref_phys, EPS))
y_hat  = np.log(np.maximum(A_hat,  EPS) / np.maximum(A_ref_hat,  EPS))

r_before = y_meas - y_phys
r_after  = y_meas - y_hat

rmse_before = rmse(y_meas, y_phys)
rmse_after  = rmse(y_meas, y_hat)
mae_before = mae(y_meas, y_phys)
mae_after  = mae(y_meas, y_hat)

if rmse_before > 1e-12:
    improve_percent_full = 100.0 * (rmse_before - rmse_after) / rmse_before
else:
    improve_percent_full = np.nan

# ---------------- evaluate in y-space (TEST ONLY) ----------------
n_train = int(np.floor(TRAIN_RATIO * N))
idx_te = np.arange(n_train, N)

t_te = t[idx_te]
A_meas_te = A_meas[idx_te]
A_phys_te = A_phys[idx_te]
A_hat_te = A_hat[idx_te]

y_meas_te = y_meas[idx_te]
y_phys_te = y_phys[idx_te]
y_hat_te  = y_hat[idx_te]

r_before_te = r_before[idx_te]
r_after_te  = r_after[idx_te]

rmse_before_test = rmse(y_meas_te, y_phys_te)
rmse_after_test  = rmse(y_meas_te, y_hat_te)
mae_before_test = mae(y_meas_te, y_phys_te)
mae_after_test  = mae(y_meas_te, y_hat_te)

if rmse_before_test > 1e-12:
    improve_percent_test = 100.0 * (rmse_before_test - rmse_after_test) / rmse_before_test
else:
    improve_percent_test = np.nan

# ---------------- save CSV ----------------
out_csv = os.path.join(DATA_DIR, f"{BASE}_qm_required_boundcheck.csv")
pd.DataFrame({
    "t_sec": t,
    "A_meas": A_meas,
    "A_phys": A_phys,
    "I_rms": I_rms,
    "G_res_ref": np.full_like(t, G_res_ref, dtype=float),
    "G_req": G_req,
    "Qm_req": Qm_req,
    "Qm_slow": Qm_slow,
    "Qm_fast": Qm_fast,
    "A_hat": A_hat,
    "y_meas": y_meas,
    "y_phys": y_phys,
    "y_hat": y_hat,
    "r_before": r_before,
    "r_after": r_after,
}).to_csv(out_csv, index=False)

# ---------------- save metrics JSON ----------------
out_metrics_json = os.path.join(DATA_DIR, f"{BASE}_stepQm4c_metrics.json")
metrics = {
    "BASE": BASE,
    "MEAS_CSV": MEAS_CSV,
    "PHYS_CSV": PHYS_CSV,
    "Qm_ref": Qm_ref,
    "G_res_ref": G_res_ref,
    "TAU_S": TAU_S,
    "TRAIN_RATIO": TRAIN_RATIO,
    "N": int(N),
    "fs_ds": float(fs_ds),
    "dt": float(dt),

    "Qm_req_min": float(np.min(Qm_req)),
    "Qm_req_max": float(np.max(Qm_req)),
    "Qm_req_mean": float(np.mean(Qm_req)),
    "Qm_req_std": float(np.std(Qm_req)),

    "fast_ratio": fast_ratio,
    "deriv_ratio": deriv_ratio,

    "rmse_before_full": rmse_before,
    "rmse_after_full": rmse_after,
    "mae_before_full": mae_before,
    "mae_after_full": mae_after,
    "improve_percent_full": improve_percent_full,

    "test_start_index": int(n_train),
    "test_samples": int(len(idx_te)),
    "rmse_before_test": rmse_before_test,
    "rmse_after_test": rmse_after_test,
    "mae_before_test": mae_before_test,
    "mae_after_test": mae_after_test,
    "improve_percent_test": improve_percent_test,
}
with open(out_metrics_json, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

# ---------------- plots ----------------
plt.figure(figsize=(12,4))
plt.plot(t, Qm_req, linewidth=1, label="Qm_required")
plt.plot(t, Qm_slow, linewidth=2, label=f"Qm_slow (EMA tau={TAU_S}s)")
plt.title(f"{BASE} | Qm required vs slow")
plt.xlabel("Time (s)")
plt.ylabel("Qm (arb.)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, f"{BASE}_Qm_required_vs_slow.png"), dpi=150)
plt.close()

plt.figure(figsize=(12,4))
plt.plot(t, Qm_fast, linewidth=1)
plt.title(f"{BASE} | Qm_fast = Qm_required - Qm_slow")
plt.xlabel("Time (s)")
plt.ylabel("Qm_fast (arb.)")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, f"{BASE}_Qm_fast.png"), dpi=150)
plt.close()

plt.figure(figsize=(12,4))
plt.plot(t, r_before, linewidth=1, label=f"before (RMSE={rmse_before:.4f})")
plt.plot(t, r_after,  linewidth=1, label=f"after  (RMSE={rmse_after:.4f})")
plt.title(f"{BASE} | y-residual before vs after (Qm_slow only, FULL)")
plt.xlabel("Time (s)")
plt.ylabel("r_y")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, f"{BASE}_r_before_after_QmSlow_FULL.png"), dpi=150)
plt.close()

plt.figure(figsize=(12,4))
plt.plot(t_te, r_before_te, linewidth=1, label=f"before_test (RMSE={rmse_before_test:.4f})")
plt.plot(t_te, r_after_te,  linewidth=1, label=f"after_test  (RMSE={rmse_after_test:.4f})")
plt.title(f"{BASE} | y-residual before vs after (Qm_slow only, TEST)")
plt.xlabel("Time (s)")
plt.ylabel("r_y")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, f"{BASE}_r_before_after_QmSlow_TEST.png"), dpi=150)
plt.close()

plt.figure(figsize=(8,6))
plt.hist(r_before, bins=120, alpha=0.5, label="before")
plt.hist(r_after,  bins=120, alpha=0.5, label="after")
plt.title(f"{BASE} | r_y histogram before/after (FULL)")
plt.xlabel("r_y")
plt.ylabel("count")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, f"{BASE}_r_hist_before_after_QmSlow_FULL.png"), dpi=150)
plt.close()

plt.figure(figsize=(8,6))
plt.hist(r_before_te, bins=80, alpha=0.5, label="before_test")
plt.hist(r_after_te,  bins=80, alpha=0.5, label="after_test")
plt.title(f"{BASE} | r_y histogram before/after (TEST)")
plt.xlabel("r_y")
plt.ylabel("count")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, f"{BASE}_r_hist_before_after_QmSlow_TEST.png"), dpi=150)
plt.close()

# ---------------- amplitude overlay (raw, full) ----------------
plt.figure(figsize=(12,4))
plt.plot(t, A_meas, linewidth=1.2, label="A_meas")
plt.plot(t, A_phys, linewidth=1.2, label="A_phys (orig)")
plt.plot(t, A_hat,  linewidth=1.2, label="A_hat (Qm_slow only)")
plt.title(f"{BASE} | Amplitude overlay (raw, FULL)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (um)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, f"{BASE}_A_overlay_raw_FULL.png"), dpi=150)
plt.close()

# ---------------- amplitude overlay (raw, test) ----------------
plt.figure(figsize=(12,4))
plt.plot(t_te, A_meas_te, linewidth=1.2, label="A_meas_test")
plt.plot(t_te, A_phys_te, linewidth=1.2, label="A_phys_test")
plt.plot(t_te, A_hat_te,  linewidth=1.2, label="A_hat_test")
plt.title(f"{BASE} | Amplitude overlay (raw, TEST)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (um)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, f"{BASE}_A_overlay_raw_TEST.png"), dpi=150)
plt.close()

# ---------------- amplitude overlay (normalized, full) ----------------
A_meas_n = A_meas / max(A_ref_meas, EPS)
A_phys_n = A_phys / max(A_ref_phys, EPS)
A_hat_n  = A_hat  / max(A_ref_hat,  EPS)

plt.figure(figsize=(12,4))
plt.plot(t, A_meas_n, linewidth=1.2, label="A_meas / A_ref_meas")
plt.plot(t, A_phys_n, linewidth=1.2, label="A_phys / A_ref_phys")
plt.plot(t, A_hat_n,  linewidth=1.2, label="A_hat / A_ref_hat")
plt.title(f"{BASE} | Amplitude overlay (normalized, FULL)")
plt.xlabel("Time (s)")
plt.ylabel("Normalized amplitude")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, f"{BASE}_A_overlay_norm_FULL.png"), dpi=150)
plt.close()

# ---------------- amplitude overlay (normalized, test) ----------------
A_meas_te_n = A_meas_te / max(A_ref_meas, EPS)
A_phys_te_n = A_phys_te / max(A_ref_phys, EPS)
A_hat_te_n  = A_hat_te  / max(A_ref_hat, EPS)

plt.figure(figsize=(12,4))
plt.plot(t_te, A_meas_te_n, linewidth=1.2, label="A_meas_test / A_ref_meas")
plt.plot(t_te, A_phys_te_n, linewidth=1.2, label="A_phys_test / A_ref_phys")
plt.plot(t_te, A_hat_te_n,  linewidth=1.2, label="A_hat_test / A_ref_hat")
plt.title(f"{BASE} | Amplitude overlay (normalized, TEST)")
plt.xlabel("Time (s)")
plt.ylabel("Normalized amplitude")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, f"{BASE}_A_overlay_norm_TEST.png"), dpi=150)
plt.close()

# ---------------- summary text plot ----------------
plt.figure(figsize=(10,6))
plt.axis("off")
lines = [
    f"BASE: {BASE}",
    f"N = {N}",
    f"fs_ds = {fs_ds:.3f} Hz",
    f"dt = {dt:.6f} s",
    "",
    f"G_res_ref = {G_res_ref:.6f}",
    f"Qm_ref    = {Qm_ref:.6f}",
    "",
    "Qm_required stats:",
    f"  min/max/mean/std = {np.min(Qm_req):.3f} / {np.max(Qm_req):.3f} / {np.mean(Qm_req):.3f} / {np.std(Qm_req):.3f}",
    "",
    "Fastness indicators:",
    f"  fast_ratio  = {fast_ratio:.3f}",
    f"  deriv_ratio = {deriv_ratio:.3f}",
    "",
    "Feasibility (FULL):",
    f"  RMSE before = {rmse_before:.6f}",
    f"  RMSE after  = {rmse_after:.6f}",
    f"  Improve %   = {improve_percent_full:.2f}",
    "",
    "Feasibility (TEST ONLY):",
    f"  RMSE before = {rmse_before_test:.6f}",
    f"  RMSE after  = {rmse_after_test:.6f}",
    f"  Improve %   = {improve_percent_test:.2f}",
]
plt.text(0.02, 0.98, "\n".join(lines), va="top", ha="left", fontsize=11, family="monospace")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, f"{BASE}_stepQm4c_summary.png"), dpi=150)
plt.close()

# ---------------- prints ----------------
print("\n================ StepQm4c: Qm-required bound check ================")
print(f"BASE={BASE} | N={N} | fs_ds~{fs_ds:.3f} Hz | dt={dt:.6f} s")
print(f"G_res_ref (median A_phys/I_rms) = {G_res_ref:.6f}")
print(f"Qm_ref = {Qm_ref:.6f}")

print("\nQm_required stats:")
print(f"  min/max/mean/std = {np.min(Qm_req):.3f} {np.max(Qm_req):.3f} {np.mean(Qm_req):.3f} {np.std(Qm_req):.3f}")

print("\nFastness indicators (heuristics):")
print(f"  fast_ratio (RMS fast / RMS total) = {fast_ratio:.3f}")
print(f"  deriv_ratio (mean|dQ_slow| / mean|dQ|) = {deriv_ratio:.3f}")
print("  (Interpretation: fast_ratio high => Qm is being forced to explain fast/cyclic stuff)")

print("\nFeasibility (FULL):")
print(f"  RMSE before = {rmse_before:.6f}")
print(f"  RMSE after  = {rmse_after:.6f}")
print(f"  Improve %   = {improve_percent_full:.2f}")

print("\nFeasibility (TEST ONLY):")
print(f"  RMSE before = {rmse_before_test:.6f}")
print(f"  RMSE after  = {rmse_after_test:.6f}")
print(f"  Improve %   = {improve_percent_test:.2f}")

print("\nSaved CSV:", out_csv)
print("Saved metrics JSON:", out_metrics_json)
print("Plots saved to:", PLOT_DIR)
print("===================================================================")