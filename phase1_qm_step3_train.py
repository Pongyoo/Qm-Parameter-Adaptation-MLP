# ============================================================
# phase1_qm_step3_train.py
# Phase 1 Step 3: Train MLP to predict Qm_slow (sensorless)
#
# Core insight:
#   Qm_slow tracks thermal drift → feature ที่ดีต้องบอก
#   "thermal state change" และ dynamics ที่ generalize ได้
#   ไม่ใช่ raw absolute value ที่ extrapolate ออกนอก train range ไม่ได้
#
# Features (causal only, 5 ตัว):
#   I_rms                     electrical level ปัจจุบัน
#   dI/dt                     rate of change ของกระแส
#   roll_mean_I (30s)         slow electrical trend
#   dPZT_from_start           thermal change from session baseline
#   roll_mean_dPZT/dt         smoothed thermal rise rate
#
# สิ่งที่ตัดออกจาก version ก่อน:
#   PZT_raw                   เสี่ยง distribution shift / extrapolation fail
#   roll_mean_PZT             ผูกกับ absolute thermal range เกินไป
#   cumsum_I_norm             session-position leak / extrapolate ไม่ได้
#   rolling_std ทุกตัว        noisy ไม่ตรงกับ smooth Qm target
#   d²I/dt²                   noisy ไม่มี physical meaning สำหรับ thermal
#
# Input:  merged_1k.csv (จาก stepQm_build_phys_and_qmcheck.py)
# Output: data/ pred.csv  metrics.json  model.pt  scaler.json
#         plots/ p1–p5
# ============================================================

import os, json, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ══════════════════════════════════════════════════════════════════
#  CONFIG  ← แก้ตรงนี้เท่านั้น
# ══════════════════════════════════════════════════════════════════
BASE    = "1.5_exp5"
IN_CSV  = r"E:\raw_data\exp5(2026.3.27)\1.5_fix18.0_processed_Qm\stepQm2_build_phys_and_qmcheck_aclean\data\merged_1k.csv"
OUT_DIR = r"E:\raw_data\exp5(2026.3.27)\1.5_fix18.0_processed_Qm\stepQm3_train"

TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15

# Rolling window ใหญ่ขึ้น — จับ thermal trend ไม่ใช่ noise
ROLL_SEC = 30.0

# MLP เล็กลง + regularization แรงขึ้น
HIDDEN       = [32, 16]
DROPOUT      = 0.20
LR           = 1e-3
WEIGHT_DECAY = 1e-3
EPOCHS       = 500
BATCH        = 2048
PATIENCE     = 40
LAMBDA_SMOOTH = 1e-10

SEED   = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPS    = 1e-12
REF_N  = 1000
# ══════════════════════════════════════════════════════════════════

PLOT_DIR = os.path.join(OUT_DIR, "plots")
DATA_DIR = os.path.join(OUT_DIR, "data")
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)


def set_seed(s):
    import random; random.seed(s)
    np.random.seed(s); torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)

def causal_roll_mean(x, win):
    s = pd.Series(x)
    minp = max(5, win // 10)
    return s.rolling(win, center=False, min_periods=minp).mean()\
             .bfill().ffill().values

def savefig(fig, name):
    p = os.path.join(PLOT_DIR, name)
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"  [plot] {p}")

def get_metrics(pred, target):
    d = np.asarray(pred, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    rmse = float(np.sqrt(np.mean(d**2)))
    mae  = float(np.mean(np.abs(d)))
    ss_res = np.sum(d**2)
    ss_tot = np.sum((np.asarray(target) - np.mean(target))**2)
    r2   = float(1 - ss_res / (ss_tot + EPS))
    return rmse, mae, r2


class QmMLP(nn.Module):
    def __init__(self, n_in, hidden, dropout):
        super().__init__()
        layers, prev = [], n_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h),
                       nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class ArrayDS(Dataset):
    def __init__(self, X, y, t):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()
        self.t = torch.from_numpy(t).float()

    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i], self.t[i]


def main():
    set_seed(SEED)
    print("\n" + "="*60)
    print(f"phase1_qm_step3_train  |  BASE={BASE}  |  device={DEVICE}")
    print("="*60)

    # 1. LOAD
    df = pd.read_csv(IN_CSV)
    t_sec   = df["t_sec"].values.astype(np.float64)
    I_rms   = df["I_rms"].values.astype(np.float64)
    Qm_slow = df["Qm_slow"].values.astype(np.float64)
    pzt     = df["PZT"].values.astype(np.float64)
    A_meas  = df["A_meas"].values.astype(np.float64)
    A_clean = df["A_clean"].values.astype(np.float64)
    A_phys  = df["A_phys"].values.astype(np.float64)
    N       = len(t_sec)
    fs      = float(round(1.0 / np.median(np.diff(t_sec[:1000]))))

    print(f"\n[Load] N={N:,}  dur={t_sec[-1]:.1f}s  fs={fs:.0f}Hz")
    print(f"  Qm_slow: {Qm_slow.min():.3f}–{Qm_slow.max():.3f}"
          f"  std={Qm_slow.std():.4f}")
    print(f"  PZT:     {pzt.min():.2f}–{pzt.max():.2f}°C"
          f"  range={pzt.max()-pzt.min():.2f}°C")

    # 2. FEATURES
    roll_win   = max(10, int(round(ROLL_SEC * fs)))
    pzt_delta  = pzt - float(pzt[0])
    pzt_rate   = causal_roll_mean(np.gradient(pzt, 1.0/fs), roll_win)
    dI         = np.gradient(I_rms, 1.0/fs)
    I_mu30     = causal_roll_mean(I_rms, roll_win)

    FEAT_NAMES = ["I_rms", "dI/dt",
                  f"roll_mean_I({ROLL_SEC}s)",
                  "dPZT_from_start",
                  "roll_mean_dPZT/dt"]

    X = np.stack([I_rms, dI, I_mu30, pzt_delta, pzt_rate],
                 axis=1).astype(np.float32)
    y = Qm_slow.astype(np.float32)
    t = t_sec.astype(np.float32)

    print(f"\n[Features] {X.shape[1]} features  roll_win={roll_win}s={ROLL_SEC}s")

    # 3. SPLIT
    n_tr  = int(N * TRAIN_FRAC)
    n_val = int(N * VAL_FRAC)
    n_te  = N - n_tr - n_val
    idx_tr  = np.arange(0, n_tr)
    idx_val = np.arange(n_tr, n_tr+n_val)
    idx_te  = np.arange(n_tr+n_val, N)

    X_tr, y_tr, t_tr = X[idx_tr],  y[idx_tr],  t[idx_tr]
    X_va, y_va, t_va = X[idx_val], y[idx_val], t[idx_val]

    print(f"[Split] train={n_tr:,}  val={n_val:,}  test={n_te:,}")
    print(f"  t ranges: train 0–{t_tr[-1]:.1f}s"
          f" | val {t_va[0]:.1f}–{t_va[-1]:.1f}s"
          f" | test {t[n_tr+n_val]:.1f}–{t[-1]:.1f}s")

    # 4. NORMALIZE
    X_mean = X_tr.mean(0, keepdims=True)
    X_std  = X_tr.std(0,  keepdims=True) + 1e-8
    y_mean = float(y_tr.mean())
    y_std  = float(y_tr.std()) + 1e-8

    nx  = lambda x:  (x - X_mean) / X_std
    ny  = lambda yy: (yy - y_mean) / y_std
    dny = lambda yy:  yy * y_std + y_mean

    # 5. TRAIN
    dl_tr = DataLoader(ArrayDS(nx(X_tr), ny(y_tr), t_tr), BATCH, shuffle=True)
    dl_va = DataLoader(ArrayDS(nx(X_va), ny(y_va), t_va), BATCH)

    model  = QmMLP(X.shape[1], HIDDEN, DROPOUT).to(DEVICE)
    opt    = torch.optim.Adam(model.parameters(), lr=LR,
                               weight_decay=WEIGHT_DECAY)
    mse_fn = nn.MSELoss()

    best_val, best_state, wait = np.inf, None, 0
    hist_tr, hist_va = [], []

    print(f"\n[Train] {X.shape[1]}→{HIDDEN}→1  "
          f"lr={LR}  wd={WEIGHT_DECAY}  dropout={DROPOUT}"
          f"  patience={PATIENCE}")
    t0 = time.time()

    for ep in range(1, EPOCHS+1):
        model.train()
        tr_l = []
        for xb, yb, tb in dl_tr:
            xb, yb, tb = xb.to(DEVICE), yb.to(DEVICE), tb.to(DEVICE)
            pred = model(xb)
            loss = mse_fn(pred, yb)
            if LAMBDA_SMOOTH > 0 and pred.numel() >= 3:
                tb_s, ord_ = torch.sort(tb)
                p  = pred[ord_]
                dp = p[1:] - p[:-1]
                dt1 = torch.clamp(tb_s[1:]-tb_s[:-1], min=1e-6)
                ddp = dp[1:]-dp[:-1]
                dt2 = torch.clamp(0.5*(dt1[1:]+dt1[:-1]), min=1e-6)
                loss = loss + LAMBDA_SMOOTH * torch.mean((ddp/dt2**2)**2)
            opt.zero_grad(); loss.backward(); opt.step()
            tr_l.append(mse_fn(model(xb).detach(), yb).item())

        model.eval()
        va_l = []
        with torch.no_grad():
            for xb, yb, _ in dl_va:
                va_l.append(mse_fn(model(xb.to(DEVICE)),
                                   yb.to(DEVICE)).item())

        trl, val = np.mean(tr_l), np.mean(va_l)
        hist_tr.append(trl); hist_va.append(val)

        if val < best_val:
            best_val  = val
            best_state = {k: v.cpu().clone()
                          for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f"  Early stop ep={ep}")
                break

        if ep % 50 == 0 or ep == 1:
            print(f"  ep{ep:4d}  tr={trl:.5f}  va={val:.5f}"
                  f"{'  ← best' if wait==0 else ''}")

    print(f"  Done in {time.time()-t0:.1f}s  best_val={best_val:.6f}")

    # 6. PREDICT
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        Qm_pred = dny(model(torch.from_numpy(nx(X)).to(DEVICE)).cpu().numpy())

    # 7. AMPLITUDE RECONSTRUCTION
    G_res_ref  = float(np.median(A_clean / (I_rms + EPS)))
    Qm_ref_val = float(np.median(Qm_slow))
    A_hat      = I_rms * G_res_ref * (Qm_pred / (Qm_ref_val + EPS))

    Aref_c = float(np.median(A_clean[:REF_N]))
    Aref_p = float(np.median(A_phys[:REF_N]))
    Aref_h = float(np.median(A_hat[:REF_N]))

    yc = np.log(np.maximum(A_clean, EPS) / max(Aref_c, EPS))
    yp = np.log(np.maximum(A_phys,  EPS) / max(Aref_p, EPS))
    yh = np.log(np.maximum(A_hat,   EPS) / max(Aref_h, EPS))

    r_before = yc - yp
    r_after  = yc - yh

    # 8. METRICS
    m_tr = get_metrics(Qm_pred[idx_tr],  y_tr)
    m_va = get_metrics(Qm_pred[idx_val], y_va)
    m_te = get_metrics(Qm_pred[idx_te],  y[idx_te])

    def amp_imp(sl):
        rb, _, _ = get_metrics(yc[sl], yp[sl])
        ra, _, _ = get_metrics(yc[sl], yh[sl])
        return rb, ra, 100*(rb-ra)/(rb+EPS)

    rb_f, ra_f, imp_f = amp_imp(slice(None))
    rb_t, ra_t, imp_t = amp_imp(slice(n_tr+n_val, None))

    print(f"\n[Qm]  train R²={m_tr[2]:.3f}  val R²={m_va[2]:.3f}"
          f"  test R²={m_te[2]:.3f}")
    print(f"[Amp] FULL {rb_f:.5f}→{ra_f:.5f}  +{imp_f:.1f}%")
    print(f"      TEST {rb_t:.5f}→{ra_t:.5f}  +{imp_t:.1f}%")

    # 9. SAVE
    pd.DataFrame({
        "t_sec": t_sec.astype(np.float32),
        "I_rms": I_rms.astype(np.float32),
        "A_meas": A_meas.astype(np.float32),
        "A_clean": A_clean.astype(np.float32),
        "A_phys": A_phys.astype(np.float32),
        "A_hat": A_hat.astype(np.float32),
        "Qm_slow": Qm_slow.astype(np.float32),
        "Qm_pred": Qm_pred.astype(np.float32),
        "r_before": r_before.astype(np.float32),
        "r_after":  r_after.astype(np.float32),
        "split": np.where(np.isin(np.arange(N), idx_tr), "train",
                 np.where(np.isin(np.arange(N), idx_val), "val", "test")),
    }).to_csv(os.path.join(DATA_DIR, f"{BASE}_pred.csv"), index=False)

    torch.save({
        "state_dict": best_state,
        "X_mean": X_mean, "X_std": X_std,
        "y_mean": y_mean, "y_std": y_std,
        "hidden": HIDDEN, "n_features": X.shape[1],
        "feat_names": FEAT_NAMES,
    }, os.path.join(DATA_DIR, "model.pt"))

    with open(os.path.join(DATA_DIR, "scaler.json"), "w") as f:
        json.dump({"X_mean": X_mean.tolist(), "X_std": X_std.tolist(),
                   "y_mean": y_mean, "y_std": y_std,
                   "feat_names": FEAT_NAMES}, f, indent=2)

    with open(os.path.join(DATA_DIR, "metrics.json"), "w") as f:
        json.dump({
            "BASE": BASE, "N": int(N), "fs": float(fs),
            "features": FEAT_NAMES, "roll_sec": ROLL_SEC,
            "hidden": HIDDEN, "lr": LR, "wd": WEIGHT_DECAY,
            "dropout": DROPOUT,
            "Qm": {"train": {"rmse": round(m_tr[0],6), "r2": round(m_tr[2],4)},
                   "val":   {"rmse": round(m_va[0],6), "r2": round(m_va[2],4)},
                   "test":  {"rmse": round(m_te[0],6), "r2": round(m_te[2],4)}},
            "amp_y_space": {
                "full_improve_pct": round(imp_f, 2),
                "test_improve_pct": round(imp_t, 2),
                "rmse_before_test": round(rb_t, 6),
                "rmse_after_test":  round(ra_t, 6),
            },
        }, f, indent=2)

    # 10. PLOTS
    # P1
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(hist_tr, lw=1.2, color="#61AFEF", label="train")
    ax.plot(hist_va, lw=1.2, color="#E06C75", label="val")
    best_ep = len(hist_tr)-1-wait
    ax.axvline(best_ep, color="k", lw=0.8, ls="--", alpha=0.5,
               label=f"best ep={best_ep}")
    ax.set(xlabel="Epoch", ylabel="MSE (norm)",
           title=f"{BASE} | Training curve")
    ax.legend(); ax.grid(True, alpha=0.25); fig.tight_layout()
    savefig(fig, "p1_train_curve.png")

    # P2
    fig, axes = plt.subplots(2, 1, figsize=(13, 8))
    for i, (sl, lab, r2) in enumerate([
        (idx_tr, "TRAIN", m_tr[2]), (idx_te, "TEST", m_te[2])
    ]):
        axes[i].plot(t_sec[sl], Qm_slow[sl], lw=2.0,
                     color="#E5C07B", alpha=0.9, label="Qm_slow (target)")
        axes[i].plot(t_sec[sl], Qm_pred[sl], lw=1.0,
                     color="#98C379", alpha=0.9,
                     label=f"Qm_pred (R²={r2:.3f})")
        axes[i].set(ylabel="Qm (arb.)",
                    title=f"{BASE} | {lab}  Qm_slow vs Qm_pred")
        axes[i].legend(); axes[i].grid(True, alpha=0.25)
    axes[1].set_xlabel("Time (s)")
    fig.tight_layout()
    savefig(fig, "p2_qm_tracking.png")

    # P3
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    axes[0].plot(t_sec, A_clean, lw=0.8, color="#61AFEF",
                 alpha=0.9, label="A_clean")
    axes[0].plot(t_sec, A_phys, lw=0.8, color="#E06C75",
                 alpha=0.6, label="A_phys(Qm_ref)")
    axes[0].plot(t_sec, A_hat, lw=1.2, color="#98C379",
                 alpha=0.9, label="A_hat(Qm_pred)")
    for xv, lbl in [(t_sec[n_tr], "tr|val"), (t_sec[n_tr+n_val], "val|te")]:
        axes[0].axvline(xv, color="k", lw=0.7, ls="--", alpha=0.3)
    axes[0].set(ylabel="µm", title=f"{BASE} | Amplitude overlay")
    axes[0].legend(); axes[0].grid(True, alpha=0.25)
    axes[1].plot(t_sec, r_before, lw=0.7, color="#E06C75", alpha=0.7,
                 label=f"before RMSE={rb_f:.4f}")
    axes[1].plot(t_sec, r_after, lw=0.7, color="#98C379", alpha=0.7,
                 label=f"after  RMSE={ra_f:.4f}")
    axes[1].axhline(0, color="k", lw=0.5, ls="--")
    axes[1].set(xlabel="Time (s)", ylabel="y-residual",
                title=f"FULL y-residual  improve={imp_f:.1f}%")
    axes[1].legend(); axes[1].grid(True, alpha=0.25)
    fig.tight_layout()
    savefig(fig, "p3_amplitude_overlay.png")

    # P4
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    axes[0].plot(t_sec[idx_te], r_before[idx_te], lw=0.8,
                 color="#E06C75", alpha=0.7, label=f"before {rb_t:.4f}")
    axes[0].plot(t_sec[idx_te], r_after[idx_te], lw=0.8,
                 color="#98C379", alpha=0.7, label=f"after  {ra_t:.4f}")
    axes[0].axhline(0, color="k", lw=0.5, ls="--")
    axes[0].set(xlabel="Time (s)", ylabel="y-residual",
                title=f"TEST residual  +{imp_t:.1f}%")
    axes[0].legend(); axes[0].grid(True, alpha=0.25)
    s = slice(None, None, 5)
    axes[1].scatter(Qm_slow[idx_te][s], Qm_pred[idx_te][s],
                    s=4, alpha=0.3, color="#98C379")
    lims = [min(Qm_slow[idx_te].min(), Qm_pred[idx_te].min()),
            max(Qm_slow[idx_te].max(), Qm_pred[idx_te].max())]
    axes[1].plot(lims, lims, "k--", lw=0.8)
    axes[1].set(xlabel="Qm_slow (true)", ylabel="Qm_pred",
                title=f"Qm scatter  R²={m_te[2]:.3f}")
    axes[1].grid(True, alpha=0.25)
    fig.suptitle(f"{BASE} | TEST", fontsize=11)
    fig.tight_layout()
    savefig(fig, "p4_test_eval.png")

    # P5: feature importance
    fig, ax = plt.subplots(figsize=(9, 4))
    with torch.no_grad():
        w1 = model.net[0].weight.cpu().numpy()
    imp = (X_std[0] * np.abs(w1).mean(0))
    imp /= imp.sum() + EPS
    short = ["I_rms", "dI/dt", f"rollI({ROLL_SEC}s)", "dPZT",
             "roll_dPZT/dt"]
    bars = ax.bar(short, imp, color="#61AFEF", alpha=0.8)
    for bar, v in zip(bars, imp):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003,
                f"{v:.3f}", ha="center", fontsize=9)
    ax.set(ylabel="Relative importance",
           title=f"{BASE} | Feature importance (|W₁| × input_std)")
    ax.grid(True, alpha=0.25, axis="y"); fig.tight_layout()
    savefig(fig, "p5_feature_importance.png")

    print("\n" + "="*60)
    print(f"✅  Done  |  BASE={BASE}")
    print(f"   Qm  train R²={m_tr[2]:.3f}  val R²={m_va[2]:.3f}"
          f"  test R²={m_te[2]:.3f}")
    print(f"   Amp TEST improve={imp_t:.1f}%")
    print(f"   → {DATA_DIR}")
    print("="*60)
    if m_te[2] < 0:
        print("\n⚠️  test R² ยังติดลบ → ดู p5_feature_importance")
        print("   ถ้า feature thermal ยัง dominant แต่ test range ≠ train range")
        print("   → distribution shift จาก thermal — เป็นเรื่องปกติ")
        print("   ดูที่ TEST amp improve แทน R² ของ Qm")


if __name__ == "__main__":
    main()