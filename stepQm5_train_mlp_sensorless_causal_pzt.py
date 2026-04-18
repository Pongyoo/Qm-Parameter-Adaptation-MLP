# ============================================================
# stepQm5_train_mlp_sensorless_causal_pzt.py
# Step 5 (variant): Train MLP to predict Qm_slow from
#                   current + PZT temperature features
#                   (sensorless, no leakage, causal feature version)
#
# Purpose:
#   - Train MLP using current-based features + PZT temperature features
#   - Keep train/test split strictly time-based (70/30)
#   - Fit scaler on train only
#   - Save predictions for ALL samples so Step 6 can evaluate test-only later
#   - Use CAUSAL rolling features only (no centered rolling, no look-ahead)
#   - In this step, diagnostic plots/metrics are TRAIN-ONLY for physics reconstruction
#
# Input:
#   processed_Qm/stepQm4d_merge_temp/data/<BASE>_qm_required_with_temp.csv
#
# Output:
#   processed_Qm/stepQm5_train_mlp_sensorless_lambda1e-15_causal_pzt/data/<BASE>_stepQm5_pred.csv
#   processed_Qm/stepQm5_train_mlp_sensorless_lambda1e-15_causal_pzt/data/<BASE>_stepQm5_metrics.json
#   processed_Qm/stepQm5_train_mlp_sensorless_lambda1e-15_causal_pzt/plots/*.png
#
# Key rules locked:
#   - X uses current-only features + PZT-derived features
#   - y(target) = Qm_slow
#   - time-based split 70/30
#   - scaler fit on train only
#   - smoothness loss uses dt from t_sec (finite difference)
#   - rolling mean/std are CAUSAL (trailing window only)
#   - Step 5 reports TRAIN-STAGE diagnostics only
#   - Final generalization claim must come from Step 6 (test only)
# ============================================================

import os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ================= USER CONFIG =================
ROOT = r"C:\Users\ploy\Desktop\ML\GRU_2"
BASE = "2.5nl-fix17.37"

IN_CSV = os.path.join(
    ROOT, "processed_Qm", "stepQm4d_merge_temp", "data",
    f"{BASE}_qm_required_with_temp.csv"
)

OUT_DIR  = os.path.join(ROOT, "processed_Qm", "stepQm5_train_mlp_sensorless_lambda1e-15_causal_pzt")
PLOT_DIR = os.path.join(OUT_DIR, "plots")
DATA_DIR = os.path.join(OUT_DIR, "data")

# Split (time-based)
TRAIN_RATIO = 0.70

# MLP
HIDDEN = [64, 64]
DROPOUT = 0.10
LR = 1e-3
EPOCHS = 60
BATCH = 2048

# Smoothness penalty
LAMBDA_SMOOTH = 1e-15

# Rolling features
ROLL_SEC = 2.0

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

REF_N = 1000
EPS = 1e-12
# =================================================

os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)


def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def safe_dt_from_time(t):
    dt = np.diff(t)
    dt = dt[np.isfinite(dt)]
    if len(dt) < 10:
        return 0.001
    return float(np.median(dt))


def rolling_mean_std_causal(x, win):
    """
    Causal / trailing rolling statistics:
    feature at time t uses only samples up to t (no future samples).
    """
    s = pd.Series(x)
    minp = max(5, win // 10)

    m = s.rolling(window=win, center=False, min_periods=minp).mean()
    v = s.rolling(window=win, center=False, min_periods=minp).std()

    # practical startup fill for initial edge
    m = m.bfill().ffill().to_numpy()
    v = v.bfill().ffill().to_numpy()

    return m, v


def finite_diff(x, t):
    return np.gradient(x, t)


class StandardScaler1D:
    def __init__(self):
        self.mu = None
        self.sig = None

    def fit(self, X):
        self.mu = np.mean(X, axis=0, keepdims=True)
        self.sig = np.std(X, axis=0, keepdims=True) + 1e-12
        return self

    def transform(self, X):
        return (X - self.mu) / self.sig

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class ArrayDataset(Dataset):
    def __init__(self, X, y, t):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()
        self.t = torch.from_numpy(t).float()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i], self.t[i]


class MLP(nn.Module):
    def __init__(self, in_dim, hidden=[64, 64], dropout=0.1):
        super().__init__()
        layers = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def rmse(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mae(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.mean(np.abs(a - b)))


def corrcoef_safe(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) < 2:
        return np.nan
    sa = np.std(a)
    sb = np.std(b)
    if sa < 1e-12 or sb < 1e-12:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def main():
    set_seed(SEED)

    if not os.path.exists(IN_CSV):
        raise FileNotFoundError(IN_CSV)

    df = pd.read_csv(IN_CSV)

    # ---- required columns ----
    tcol = pick_col(df, ["t_sec", "t", "Time"])
    if tcol is None:
        raise KeyError("Missing time column (t_sec)")

    if "I_rms" not in df.columns:
        raise KeyError("Missing I_rms in Step4d CSV")
    if "Qm_slow" not in df.columns:
        raise KeyError("Missing Qm_slow in Step4d CSV")
    if "PZT" not in df.columns:
        raise KeyError("Missing PZT in Step4d CSV")

    has_A = all(c in df.columns for c in ["A_meas", "A_phys"])

    t = df[tcol].to_numpy(dtype=np.float64)
    I = df["I_rms"].to_numpy(dtype=np.float64)
    y = df["Qm_slow"].to_numpy(dtype=np.float64)
    pzt = df["PZT"].to_numpy(dtype=np.float64)

    # ---- build features (CAUSAL) ----
    dt = safe_dt_from_time(t)
    fs = 1.0 / dt
    roll_win = int(max(10, round(ROLL_SEC * fs)))

    # current features
    dI = finite_diff(I, t)
    d2I = finite_diff(dI, t)
    I_mu, I_sd = rolling_mean_std_causal(I, roll_win)

    # PZT features
    #dPZT = finite_diff(pzt, t)
    PZT_mu, PZT_sd = rolling_mean_std_causal(pzt, roll_win)

    X = np.column_stack([
        I, dI, d2I, I_mu, I_sd,
        pzt, PZT_mu, PZT_sd
    ]).astype(np.float32)

    y = y.astype(np.float32)
    t = t.astype(np.float32)

    # ---- time split 70/30 ----
    N = len(y)
    n_train = int(np.floor(TRAIN_RATIO * N))
    idx_tr = np.arange(0, n_train)
    idx_te = np.arange(n_train, N)

    X_tr, y_tr, t_tr = X[idx_tr], y[idx_tr], t[idx_tr]
    X_te, y_te, t_te = X[idx_te], y[idx_te], t[idx_te]

    print("\n[INFO] Split summary")
    print("  N total   =", N)
    print("  N train   =", len(idx_tr))
    print("  N test    =", len(idx_te))
    print("  train t   =", float(t_tr[0]), "to", float(t_tr[-1]))
    print("  test  t   =", float(t_te[0]), "to", float(t_te[-1]))
    print("  rolling   = CAUSAL / trailing only")
    print("  roll_win  =", int(roll_win), "samples")
    print("  features  = current + PZT")

    # ---- scaler fit TRAIN only ----
    sc = StandardScaler1D()
    X_tr_n = sc.fit_transform(X_tr)
    X_te_n = sc.transform(X_te)

    # ---- loaders ----
    ds_tr = ArrayDataset(X_tr_n, y_tr, t_tr)
    ds_te = ArrayDataset(X_te_n, y_te, t_te)
    dl_tr = DataLoader(ds_tr, batch_size=BATCH, shuffle=True, drop_last=False)
    dl_te = DataLoader(ds_te, batch_size=BATCH, shuffle=False, drop_last=False)

    # ---- model ----
    model = MLP(in_dim=X.shape[1], hidden=HIDDEN, dropout=DROPOUT).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    mse = nn.MSELoss()

    train_losses = []
    test_losses = []

    # ---- train ----
    for ep in range(EPOCHS):
        model.train()
        loss_sum = 0.0
        n_sum = 0

        for xb, yb, tb in dl_tr:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            tb = tb.to(DEVICE)

            pred = model(xb)

            loss_main = mse(pred, yb)

            tb_sorted, order = torch.sort(tb)
            p = pred[order].view(-1)

            if p.numel() >= 3:
                dp = p[1:] - p[:-1]
                dt1 = torch.clamp(tb_sorted[1:] - tb_sorted[:-1], min=1e-6)

                ddp = dp[1:] - dp[:-1]
                dt2 = torch.clamp(0.5 * (dt1[1:] + dt1[:-1]), min=1e-6)
                loss_smooth = torch.mean((ddp / (dt2 ** 2)) ** 2)
            else:
                loss_smooth = torch.zeros((), device=pred.device)

            loss = loss_main + (LAMBDA_SMOOTH * loss_smooth)

            opt.zero_grad()
            loss.backward()
            opt.step()

            loss_sum += float(loss_main.detach().cpu().item()) * len(yb)
            n_sum += len(yb)

        train_losses.append(loss_sum / max(1, n_sum))

        # quick test MSE (diagnostic only)
        model.eval()
        with torch.no_grad():
            preds = []
            ys = []
            for xb, yb, tb in dl_te:
                xb = xb.to(DEVICE)
                pred = model(xb).detach().cpu().numpy()
                preds.append(pred)
                ys.append(yb.numpy())
            preds = np.concatenate(preds)
            ys = np.concatenate(ys)
            test_losses.append(float(np.mean((preds - ys) ** 2)))

    # ---- inference on ALL (for saving; final eval belongs to Step 6) ----
    model.eval()
    with torch.no_grad():
        X_all_n = sc.transform(X).astype(np.float32)
        pred_all = model(torch.from_numpy(X_all_n).to(DEVICE)).detach().cpu().numpy().astype(np.float32)

    pred_tr = pred_all[idx_tr]
    pred_te = pred_all[idx_te]

    # =========================================================
    # Qm metrics
    # =========================================================
    rmse_qm_train = rmse(pred_tr, y_tr)
    mae_qm_train = mae(pred_tr, y_tr)
    corr_qm_train = corrcoef_safe(pred_tr, y_tr)

    rmse_qm_test = rmse(pred_te, y_te)
    mae_qm_test = mae(pred_te, y_te)
    corr_qm_test = corrcoef_safe(pred_te, y_te)

    # =========================================================
    # Rebuild amplitude using Qm_pred
    # =========================================================
    if "G_res_ref" in df.columns:
        G_res_ref = float(np.median(df["G_res_ref"].to_numpy(dtype=np.float64)))
    else:
        if "A_phys" in df.columns:
            G_res_ref = float(np.median(df["A_phys"].to_numpy(dtype=np.float64) / (I + EPS)))
        else:
            G_res_ref = 1.0

    Qm_ref = float(np.median(df["Qm_slow"].to_numpy(dtype=np.float64)))
    A_hat_pred = (I * (G_res_ref * (pred_all / (Qm_ref + EPS)))).astype(np.float32)

    # =========================================================
    # TRAIN-ONLY physics diagnostics
    # =========================================================
    rmse_y_before_train = np.nan
    rmse_y_after_train = np.nan
    improve_percent_y_train = np.nan

    if has_A:
        A_meas = df["A_meas"].to_numpy(dtype=np.float64)
        A_phys = df["A_phys"].to_numpy(dtype=np.float64)

        Aref_meas = float(np.median(A_meas[:min(REF_N, len(A_meas))]))
        Aref_phys = float(np.median(A_phys[:min(REF_N, len(A_phys))]))
        Aref_hat  = float(np.median(A_hat_pred[:min(REF_N, len(A_hat_pred))]))

        y_meas = np.log(np.maximum(A_meas, EPS) / np.maximum(Aref_meas, EPS))
        y_phys = np.log(np.maximum(A_phys, EPS) / np.maximum(Aref_phys, EPS))
        y_hat  = np.log(np.maximum(A_hat_pred, EPS) / np.maximum(Aref_hat, EPS))

        r_before = y_meas - y_phys
        r_after  = y_meas - y_hat

        y_meas_tr = y_meas[idx_tr]
        y_phys_tr = y_phys[idx_tr]
        y_hat_tr  = y_hat[idx_tr]
        r_before_tr = r_before[idx_tr]
        r_after_tr  = r_after[idx_tr]

        rmse_y_before_train = rmse(y_meas_tr, y_phys_tr)
        rmse_y_after_train  = rmse(y_meas_tr, y_hat_tr)

        if rmse_y_before_train > 1e-12:
            improve_percent_y_train = 100.0 * (rmse_y_before_train - rmse_y_after_train) / rmse_y_before_train
    else:
        A_meas = A_phys = None
        y_meas = y_phys = y_hat = None
        r_before = r_after = None
        r_before_tr = r_after_tr = None

    # =========================================================
    # Save pred CSV
    # =========================================================
    out_pred_csv = os.path.join(DATA_DIR, f"{BASE}_stepQm5_pred.csv")
    out = pd.DataFrame({
        "t_sec": t,
        "I_rms": I.astype(np.float32),
        "Qm_slow": df["Qm_slow"].to_numpy(dtype=np.float32),
        "Qm_pred": pred_all.astype(np.float32),
        "A_hat_pred": A_hat_pred.astype(np.float32),
        "is_train": np.isin(np.arange(N), idx_tr).astype(np.int32),
        "is_test": np.isin(np.arange(N), idx_te).astype(np.int32),
    })

    # carry useful original columns
    for c in ["A_meas", "A_phys", "A_hat", "y_meas", "y_phys", "y_hat", "r_before", "r_after", "PZT", "Room_temp", "Tool_temp"]:
        if c in df.columns:
            out[c] = df[c].to_numpy()

    out.to_csv(out_pred_csv, index=False)

    # =========================================================
    # Save metrics JSON
    # =========================================================
    metrics = {
        "BASE": BASE,
        "IN_CSV": IN_CSV,
        "TRAIN_RATIO": TRAIN_RATIO,
        "features": [
            "I_rms",
            "dI_dt",
            "d2I_dt2",
            f"roll_mean_{ROLL_SEC}s_causal_I",
            f"roll_std_{ROLL_SEC}s_causal_I",
            "PZT",
            f"roll_mean_{ROLL_SEC}s_causal_PZT",
            f"roll_std_{ROLL_SEC}s_causal_PZT"
        ],
        "target": "Qm_slow",
        "lambda_smooth": LAMBDA_SMOOTH,
        "rolling_mode": "causal_trailing_only",
        "roll_sec": ROLL_SEC,
        "roll_win_samples": int(roll_win),

        "rmse_qm_train": rmse_qm_train,
        "mae_qm_train": mae_qm_train,
        "corr_qm_train": corr_qm_train,

        "rmse_qm_test": rmse_qm_test,
        "mae_qm_test": mae_qm_test,
        "corr_qm_test": corr_qm_test,

        "rmse_y_before_train": rmse_y_before_train,
        "rmse_y_after_train": rmse_y_after_train,
        "improve_percent_y_train": improve_percent_y_train,

        "device": DEVICE,
        "epochs": EPOCHS,
        "lr": LR,
        "hidden": HIDDEN,
        "dropout": DROPOUT,

        "note": "Step 5 PZT variant. Final test-only claim must come from Step 6."
    }

    out_metrics = os.path.join(DATA_DIR, f"{BASE}_stepQm5_metrics.json")
    with open(out_metrics, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # =========================================================
    # Plots
    # =========================================================

    # 1) training curve
    plt.figure(figsize=(10, 4))
    plt.plot(train_losses, label="train_mse")
    plt.plot(test_losses, label="test_mse")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title(f"{BASE} | training curve (current + PZT)")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{BASE}_train_curve.png"), dpi=160)
    plt.close()

    # 2) Qm overlay - TRAIN only diagnostic
    plt.figure(figsize=(12, 4))
    plt.plot(t_tr, y_tr, lw=2, label="Qm_slow (target, train)")
    plt.plot(t_tr, pred_tr, lw=1, alpha=0.85, label=f"Qm_pred (MLP, train, lambda={LAMBDA_SMOOTH})")
    plt.xlabel("Time (s)")
    plt.ylabel("Qm (arb.)")
    plt.title(f"{BASE} | TRAIN ONLY | Qm_slow vs Qm_pred (current + PZT)")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{BASE}_TRAIN_Qm_overlay.png"), dpi=160)
    plt.close()

    if has_A:
        # 3) amplitude overlay - TRAIN only
        plt.figure(figsize=(12, 4))
        plt.plot(t_tr, A_meas[idx_tr], lw=1.5, label="A_meas")
        plt.plot(t_tr, A_phys[idx_tr], lw=1.5, label="A_phys_ref")
        plt.plot(t_tr, A_hat_pred[idx_tr], lw=1.5, label="A_hat(Qm_pred)")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude (arb./um)")
        plt.title(f"{BASE} | TRAIN ONLY | Amplitude overlay (current + PZT)")
        plt.grid(True, alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, f"{BASE}_TRAIN_A_overlay.png"), dpi=160)
        plt.close()

        # 4) y-space overlay - TRAIN only
        plt.figure(figsize=(12, 4))
        plt.plot(t_tr, y_meas[idx_tr], lw=1.5, label="y_meas")
        plt.plot(t_tr, y_phys[idx_tr], lw=1.5, label="y_ref (phys)")
        plt.plot(t_tr, y_hat[idx_tr], lw=1.5, label="y_hat (Qm_pred)")
        plt.xlabel("Time (s)")
        plt.ylabel("y = log(A/Aref)")
        plt.title(f"{BASE} | TRAIN ONLY | y-space overlay (current + PZT)")
        plt.grid(True, alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, f"{BASE}_TRAIN_y_overlay.png"), dpi=160)
        plt.close()

        # 5) residual before/after - TRAIN only
        plt.figure(figsize=(12, 4))
        plt.plot(t_tr, r_before_tr, lw=1.2, label=f"before (RMSE={rmse_y_before_train:.4f})")
        plt.plot(t_tr, r_after_tr,  lw=1.2, label=f"after  (RMSE={rmse_y_after_train:.4f})")
        plt.xlabel("Time (s)")
        plt.ylabel("r_y")
        plt.title(f"{BASE} | TRAIN ONLY | y-residual before vs after (current + PZT)")
        plt.grid(True, alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, f"{BASE}_TRAIN_r_before_after.png"), dpi=160)
        plt.close()

    print("\n================ StepQm5 DONE (sensorless MLP, current + PZT) ================")
    print("Input CSV:", IN_CSV)
    print("Saved pred CSV:", out_pred_csv)
    print("Saved metrics:", out_metrics)
    print("Plots:", PLOT_DIR)

    print("\nTraining-stage diagnostics:")
    print("  RMSE_qm_train      =", rmse_qm_train)
    print("  MAE_qm_train       =", mae_qm_train)
    print("  Corr_qm_train      =", corr_qm_train)
    print("  RMSE_qm_test(diag) =", rmse_qm_test)
    print("  MAE_qm_test(diag)  =", mae_qm_test)
    print("  Corr_qm_test(diag) =", corr_qm_test)

    if has_A:
        print("  RMSE_y_before_train =", rmse_y_before_train)
        print("  RMSE_y_after_train  =", rmse_y_after_train)
        print(f"  improvement_y_train = {improve_percent_y_train:.2f}%")

    print("\nNote: causal rolling features used; PZT added as thermal feature.")
    print("      Final test-only result must be reported from Step 6.")
    print("===============================================================================")


if __name__ == "__main__":
    main()