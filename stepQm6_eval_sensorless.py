# ============================================================
# stepQm6_eval_sensorless.py
# Step 6: Test-only evaluation for sensorless Qm learning
#
# Purpose:
#   Evaluate ONLY the unseen test segment (last 30%) from Step 5 results.
#   Compare actual learned performance against Step 4c feasibility bound.
#
# Input:
#   processed_Qm/stepQm5_train_mlp_sensorless_lambda*/data/<BASE>_stepQm5_pred.csv
#   processed_Qm/stepQm5_train_mlp_sensorless_lambda*/data/<BASE>_stepQm5_metrics.json   (optional)
#   processed_Qm/stepQm4c_boundcheck/data/<BASE>_stepQm4c_metrics.json                    (optional but recommended)
#
# Output:
#   processed_Qm/stepQm6_eval_sensorless/data/<BASE>_stepQm6_test_eval.csv
#   processed_Qm/stepQm6_eval_sensorless/data/<BASE>_stepQm6_test_metrics.json
#   processed_Qm/stepQm6_eval_sensorless/plots/*.png
#
# Key rules:
#   - Evaluate TEST segment only
#   - Recompute y-space and residual on test segment
#   - Keep plotting/reporting style similar to Step 5
#   - Compare actual result vs Step 4c test-only upper bound
# ============================================================

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ================= USER CONFIG =================
ROOT = r"C:\Users\ploy\Desktop\ML\GRU_2"
BASE = "2.5nl-fix17.37"

# ---- Step 5 output folder to evaluate ----
STEP5_DIR = os.path.join(ROOT, "processed_Qm", "stepQm5_pzt_rise")

IN_PRED_CSV = os.path.join(
    STEP5_DIR, "data", f"{BASE}_stepQm5_pred.csv"
)

IN_STEP5_METRICS_JSON = os.path.join(
    STEP5_DIR, "data", f"{BASE}_stepQm5_metrics.json"
)

# ---- Step 4c feasibility metrics ----
IN_STEP4C_METRICS_JSON = os.path.join(
    ROOT, "processed_Qm", "stepQm4c_boundcheck", "data", f"{BASE}_stepQm4c_metrics.json"
)

OUT_DIR  = os.path.join(ROOT, "processed_Qm", "stepQm6_pzt_rise_lamda1e-12_eval")
PLOT_DIR = os.path.join(OUT_DIR, "plots")
DATA_DIR = os.path.join(OUT_DIR, "data")

# Same split ratio as Step 5 / Step 4c
TRAIN_RATIO = 0.70

# Reference window for y = log(A/Aref)
REF_N = 1000
EPS = 1e-12
# =================================================

os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)


def rmse(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mae(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.mean(np.abs(a - b)))


def r2_score_np(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot < 1e-12:
        return np.nan
    return float(1.0 - ss_res / ss_tot)


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


def load_json_if_exists(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def main():
    if not os.path.exists(IN_PRED_CSV):
        raise FileNotFoundError(f"Missing Step 5 prediction CSV:\n{IN_PRED_CSV}")

    df = pd.read_csv(IN_PRED_CSV)

    # ---- optional external metrics ----
    step5_metrics = load_json_if_exists(IN_STEP5_METRICS_JSON)
    step4c_metrics = load_json_if_exists(IN_STEP4C_METRICS_JSON)

    # ---- required columns from Step 5 ----
    required_cols = ["t_sec", "I_rms", "Qm_slow", "Qm_pred"]
    for c in required_cols:
        if c not in df.columns:
            raise KeyError(f"Missing required column in Step 5 pred CSV: {c}")

    has_amp = all(c in df.columns for c in ["A_meas", "A_phys", "A_hat_pred"])

    t = df["t_sec"].to_numpy(dtype=np.float64)
    I = df["I_rms"].to_numpy(dtype=np.float64)
    qm_true = df["Qm_slow"].to_numpy(dtype=np.float64)
    qm_pred = df["Qm_pred"].to_numpy(dtype=np.float64)

    # ---- split same as Step 5 ----
    N = len(df)
    n_train = int(np.floor(TRAIN_RATIO * N))
    idx_te = np.arange(n_train, N)

    t_te = t[idx_te]
    I_te = I[idx_te]
    qm_true_te = qm_true[idx_te]
    qm_pred_te = qm_pred[idx_te]

    # =========================================================
    # Qm-space metrics (test only)
    # =========================================================
    rmse_qm_test = rmse(qm_pred_te, qm_true_te)
    mae_qm_test = mae(qm_pred_te, qm_true_te)
    r2_qm_test = r2_score_np(qm_true_te, qm_pred_te)
    corr_qm_test = corrcoef_safe(qm_pred_te, qm_true_te)
    bias_qm_test = float(np.mean(qm_pred_te - qm_true_te))
    std_err_qm_test = float(np.std(qm_pred_te - qm_true_te))

    # =========================================================
    # y-space / residual recompute (test only)
    # =========================================================
    rmse_y_before_test = np.nan
    rmse_y_after_test = np.nan
    mae_y_before_test = np.nan
    mae_y_after_test = np.nan
    improve_percent_y_test = np.nan
    bias_r_before_test = np.nan
    bias_r_after_test = np.nan
    std_r_before_test = np.nan
    std_r_after_test = np.nan

    y_meas_all = y_phys_all = y_hat_all = None
    r_before_all = r_after_all = None

    if has_amp:
        A_meas = df["A_meas"].to_numpy(dtype=np.float64)
        A_phys = df["A_phys"].to_numpy(dtype=np.float64)
        A_hat_pred = df["A_hat_pred"].to_numpy(dtype=np.float64)

        # same normalization philosophy as Step 5 / Step 4c
        Aref_meas = float(np.median(A_meas[:min(REF_N, len(A_meas))]))
        Aref_phys = float(np.median(A_phys[:min(REF_N, len(A_phys))]))
        Aref_hat  = float(np.median(A_hat_pred[:min(REF_N, len(A_hat_pred))]))

        y_meas_all = np.log(np.maximum(A_meas, EPS) / np.maximum(Aref_meas, EPS))
        y_phys_all = np.log(np.maximum(A_phys, EPS) / np.maximum(Aref_phys, EPS))
        y_hat_all  = np.log(np.maximum(A_hat_pred, EPS) / np.maximum(Aref_hat, EPS))

        r_before_all = y_meas_all - y_phys_all
        r_after_all  = y_meas_all - y_hat_all

        y_meas_te = y_meas_all[idx_te]
        y_phys_te = y_phys_all[idx_te]
        y_hat_te  = y_hat_all[idx_te]
        r_before_te = r_before_all[idx_te]
        r_after_te  = r_after_all[idx_te]

        rmse_y_before_test = rmse(y_meas_te, y_phys_te)
        rmse_y_after_test  = rmse(y_meas_te, y_hat_te)
        mae_y_before_test = mae(y_meas_te, y_phys_te)
        mae_y_after_test  = mae(y_meas_te, y_hat_te)

        if np.isfinite(rmse_y_before_test) and rmse_y_before_test > 1e-12:
            improve_percent_y_test = 100.0 * (rmse_y_before_test - rmse_y_after_test) / rmse_y_before_test

        bias_r_before_test = float(np.mean(r_before_te))
        bias_r_after_test = float(np.mean(r_after_te))
        std_r_before_test = float(np.std(r_before_te))
        std_r_after_test = float(np.std(r_after_te))
    else:
        A_meas = A_phys = A_hat_pred = None

    # =========================================================
    # Compare against Step 4c test-only upper bound
    # =========================================================
    upper_bound_rmse_before_test = np.nan
    upper_bound_rmse_after_test = np.nan
    upper_bound_improve_test = np.nan

    improve_gap_points = np.nan
    rmse_after_gap_to_bound = np.nan
    attainment_ratio_percent = np.nan

    if step4c_metrics:
        upper_bound_rmse_before_test = float(step4c_metrics.get("rmse_before_test", np.nan))
        upper_bound_rmse_after_test = float(step4c_metrics.get("rmse_after_test", np.nan))
        upper_bound_improve_test = float(step4c_metrics.get("improve_percent_test", np.nan))

        if np.isfinite(upper_bound_improve_test) and np.isfinite(improve_percent_y_test):
            improve_gap_points = upper_bound_improve_test - improve_percent_y_test

        if np.isfinite(upper_bound_rmse_after_test) and np.isfinite(rmse_y_after_test):
            rmse_after_gap_to_bound = rmse_y_after_test - upper_bound_rmse_after_test

        if np.isfinite(upper_bound_improve_test) and abs(upper_bound_improve_test) > 1e-12 and np.isfinite(improve_percent_y_test):
            attainment_ratio_percent = 100.0 * improve_percent_y_test / upper_bound_improve_test

    # =========================================================
    # Save test-only eval csv
    # =========================================================
    out_eval_csv = os.path.join(DATA_DIR, f"{BASE}_stepQm6_test_eval.csv")

    out_df = pd.DataFrame({
        "t_sec": t_te,
        "I_rms": I_te,
        "Qm_slow_test": qm_true_te,
        "Qm_pred_test": qm_pred_te,
        "Qm_error_test": (qm_pred_te - qm_true_te),
    })

    if has_amp:
        out_df["A_meas_test"] = A_meas[idx_te]
        out_df["A_phys_test"] = A_phys[idx_te]
        out_df["A_hat_pred_test"] = A_hat_pred[idx_te]
        out_df["y_meas_test"] = y_meas_all[idx_te]
        out_df["y_phys_test"] = y_phys_all[idx_te]
        out_df["y_hat_test"] = y_hat_all[idx_te]
        out_df["r_before_test"] = r_before_all[idx_te]
        out_df["r_after_test"] = r_after_all[idx_te]

    out_df.to_csv(out_eval_csv, index=False)

    # =========================================================
    # Save metrics json
    # =========================================================
    out_metrics = {
        "BASE": BASE,
        "STEP5_DIR": STEP5_DIR,
        "IN_PRED_CSV": IN_PRED_CSV,
        "IN_STEP5_METRICS_JSON": IN_STEP5_METRICS_JSON,
        "IN_STEP4C_METRICS_JSON": IN_STEP4C_METRICS_JSON,
        "TRAIN_RATIO": TRAIN_RATIO,
        "eval_scope": "test_only",
        "test_start_index": int(n_train),
        "test_samples": int(len(idx_te)),

        "rmse_qm_test": rmse_qm_test,
        "mae_qm_test": mae_qm_test,
        "r2_qm_test": r2_qm_test,
        "corr_qm_test": corr_qm_test,
        "bias_qm_test": bias_qm_test,
        "std_err_qm_test": std_err_qm_test,

        "rmse_y_before_test": rmse_y_before_test,
        "rmse_y_after_test": rmse_y_after_test,
        "mae_y_before_test": mae_y_before_test,
        "mae_y_after_test": mae_y_after_test,
        "improve_percent_y_test": improve_percent_y_test,

        "bias_r_before_test": bias_r_before_test,
        "bias_r_after_test": bias_r_after_test,
        "std_r_before_test": std_r_before_test,
        "std_r_after_test": std_r_after_test,

        # Step 4c upper bound comparison
        "upper_bound_rmse_before_test": upper_bound_rmse_before_test,
        "upper_bound_rmse_after_test": upper_bound_rmse_after_test,
        "upper_bound_improve_test": upper_bound_improve_test,
        "improve_gap_points": improve_gap_points,
        "rmse_after_gap_to_bound": rmse_after_gap_to_bound,
        "attainment_ratio_percent": attainment_ratio_percent,

        "step5_metrics_reference": step5_metrics,
        "step4c_metrics_reference": step4c_metrics,
    }

    out_metrics_json = os.path.join(DATA_DIR, f"{BASE}_stepQm6_test_metrics.json")
    with open(out_metrics_json, "w", encoding="utf-8") as f:
        json.dump(out_metrics, f, indent=2)

    # =========================================================
    # PLOTS (test only, similar style to Step 5)
    # =========================================================

    # 1) Qm overlay (test only)
    plt.figure(figsize=(12, 4))
    plt.plot(t_te, qm_true_te, lw=2, label="Qm_slow (target, test)")
    plt.plot(t_te, qm_pred_te, lw=1.2, alpha=0.9, label="Qm_pred (MLP, test)")
    plt.xlabel("Time (s)")
    plt.ylabel("Qm (arb.)")
    plt.title(f"{BASE} | TEST ONLY | Qm_slow vs Qm_pred")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{BASE}_TEST_Qm_overlay.png"), dpi=160)
    plt.close()

    # 2) Qm error
    plt.figure(figsize=(12, 4))
    plt.plot(t_te, qm_pred_te - qm_true_te, lw=1.2, label="Qm_pred - Qm_slow")
    plt.axhline(0.0, color="k", lw=1.0, alpha=0.7)
    plt.xlabel("Time (s)")
    plt.ylabel("Qm error")
    plt.title(f"{BASE} | TEST ONLY | Qm prediction error")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{BASE}_TEST_Qm_error.png"), dpi=160)
    plt.close()

    if has_amp:
        y_meas_te = y_meas_all[idx_te]
        y_phys_te = y_phys_all[idx_te]
        y_hat_te  = y_hat_all[idx_te]
        r_before_te = r_before_all[idx_te]
        r_after_te  = r_after_all[idx_te]

        # 3) amplitude overlay (test only)
        plt.figure(figsize=(12, 4))
        plt.plot(t_te, A_meas[idx_te], lw=1.5, label="A_meas")
        plt.plot(t_te, A_phys[idx_te], lw=1.5, label="A_phys_ref")
        plt.plot(t_te, A_hat_pred[idx_te], lw=1.5, label="A_hat(Qm_pred)")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude (arb./um)")
        plt.title(f"{BASE} | TEST ONLY | Amplitude overlay")
        plt.grid(True, alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, f"{BASE}_TEST_A_overlay.png"), dpi=160)
        plt.close()

        # 4) y-space overlay (test only)
        plt.figure(figsize=(12, 4))
        plt.plot(t_te, y_meas_te, lw=1.5, label="y_meas")
        plt.plot(t_te, y_phys_te, lw=1.5, label="y_ref (phys)")
        plt.plot(t_te, y_hat_te, lw=1.5, label="y_hat (Qm_pred)")
        plt.xlabel("Time (s)")
        plt.ylabel("y = log(A/Aref)")
        plt.title(f"{BASE} | TEST ONLY | y-space overlay")
        plt.grid(True, alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, f"{BASE}_TEST_y_overlay.png"), dpi=160)
        plt.close()

        # 5) residual before vs after (test only)
        plt.figure(figsize=(12, 4))
        plt.plot(t_te, r_before_te, lw=1.2, label=f"before (RMSE={rmse_y_before_test:.4f})")
        plt.plot(t_te, r_after_te, lw=1.2, label=f"after  (RMSE={rmse_y_after_test:.4f})")
        plt.xlabel("Time (s)")
        plt.ylabel("r_y")
        plt.title(f"{BASE} | TEST ONLY | y-residual before vs after")
        plt.grid(True, alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, f"{BASE}_TEST_r_before_after.png"), dpi=160)
        plt.close()

        # 6) residual histogram (test only)
        plt.figure(figsize=(8, 6))
        plt.hist(r_before_te, bins=80, alpha=0.5, label="before")
        plt.hist(r_after_te, bins=80, alpha=0.5, label="after")
        plt.xlabel("r_y")
        plt.ylabel("count")
        plt.title(f"{BASE} | TEST ONLY | residual histogram")
        plt.grid(True, alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, f"{BASE}_TEST_r_hist.png"), dpi=160)
        plt.close()

        # 7) scatter residual vs A_hat_pred (test only)
        plt.figure(figsize=(7, 7))
        plt.scatter(A_hat_pred[idx_te], r_after_te, s=8, alpha=0.5)
        plt.xlabel("A_hat_pred")
        plt.ylabel("r_after")
        plt.title(f"{BASE} | TEST ONLY | r_after vs A_hat_pred")
        plt.grid(True, alpha=0.25)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, f"{BASE}_TEST_r_scatter.png"), dpi=160)
        plt.close()

    # 8) summary text figure
    plt.figure(figsize=(10, 7))
    plt.axis("off")

    lines = [
        f"BASE: {BASE}",
        f"Evaluation scope: TEST ONLY",
        f"Train/Test split: {int(TRAIN_RATIO*100)}/{int((1-TRAIN_RATIO)*100)}",
        f"Test samples: {len(idx_te)}",
        "",
        "Qm metrics:",
        f"  RMSE(Qm_test)  = {rmse_qm_test:.6f}",
        f"  MAE(Qm_test)   = {mae_qm_test:.6f}",
        f"  R2(Qm_test)    = {r2_qm_test:.6f}" if np.isfinite(r2_qm_test) else "  R2(Qm_test)    = nan",
        f"  Corr(Qm_test)  = {corr_qm_test:.6f}" if np.isfinite(corr_qm_test) else "  Corr(Qm_test)  = nan",
        f"  Bias(Qm_test)  = {bias_qm_test:.6f}",
        "",
    ]

    if has_amp:
        lines += [
            "y-space / residual metrics:",
            f"  RMSE_y_before_test = {rmse_y_before_test:.6f}",
            f"  RMSE_y_after_test  = {rmse_y_after_test:.6f}",
            f"  MAE_y_before_test  = {mae_y_before_test:.6f}",
            f"  MAE_y_after_test   = {mae_y_after_test:.6f}",
            f"  Improvement (%)    = {improve_percent_y_test:.2f}",
            "",
        ]
    else:
        lines += ["Amplitude columns not found -> y-space evaluation skipped", ""]

    if step4c_metrics:
        lines += [
            "Step 4c upper bound (TEST ONLY):",
            f"  RMSE_before_bound  = {upper_bound_rmse_before_test:.6f}",
            f"  RMSE_after_bound   = {upper_bound_rmse_after_test:.6f}",
            f"  Improve_bound (%)  = {upper_bound_improve_test:.2f}",
            "",
            "Actual vs Bound:",
            f"  Improve gap (pt)   = {improve_gap_points:.2f}" if np.isfinite(improve_gap_points) else "  Improve gap (pt)   = nan",
            f"  RMSE gap to bound  = {rmse_after_gap_to_bound:.6f}" if np.isfinite(rmse_after_gap_to_bound) else "  RMSE gap to bound  = nan",
            f"  Attainment (%)     = {attainment_ratio_percent:.2f}" if np.isfinite(attainment_ratio_percent) else "  Attainment (%)     = nan",
        ]
    else:
        lines += ["Step 4c metrics not found -> upper bound comparison skipped"]

    plt.text(
        0.02, 0.98,
        "\n".join(lines),
        va="top", ha="left",
        fontsize=11,
        family="monospace"
    )
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{BASE}_TEST_summary.png"), dpi=160)
    plt.close()

    # =========================================================
    # Print summary
    # =========================================================
    print("\n================ StepQm6 DONE (TEST-ONLY EVAL) ================")
    print("Input Step5 pred CSV :", IN_PRED_CSV)
    print("Input Step4c metrics :", IN_STEP4C_METRICS_JSON if os.path.exists(IN_STEP4C_METRICS_JSON) else "NOT FOUND")
    print("Output eval CSV      :", out_eval_csv)
    print("Output metrics JSON  :", out_metrics_json)
    print("Plots saved to       :", PLOT_DIR)

    print("\nKey TEST results:")
    print("  RMSE_qm_test       =", rmse_qm_test)
    print("  MAE_qm_test        =", mae_qm_test)
    print("  R2_qm_test         =", r2_qm_test)
    print("  Corr_qm_test       =", corr_qm_test)

    if has_amp:
        print("  RMSE_y_before_test =", rmse_y_before_test)
        print("  RMSE_y_after_test  =", rmse_y_after_test)
        print(f"  improvement_y_test = {improve_percent_y_test:.2f}%")
    else:
        print("  y-space evaluation skipped (missing A_meas/A_phys/A_hat_pred)")

    if step4c_metrics:
        print("\nStep 4c upper bound (TEST ONLY):")
        print("  RMSE_before_bound  =", upper_bound_rmse_before_test)
        print("  RMSE_after_bound   =", upper_bound_rmse_after_test)
        print(f"  Improve_bound      = {upper_bound_improve_test:.2f}%")

        print("\nActual vs Bound:")
        print(f"  Improve gap        = {improve_gap_points:.2f} points" if np.isfinite(improve_gap_points) else "  Improve gap        = nan")
        print("  RMSE gap to bound  =", rmse_after_gap_to_bound)
        print(f"  Attainment ratio   = {attainment_ratio_percent:.2f}%" if np.isfinite(attainment_ratio_percent) else "  Attainment ratio   = nan")
    else:
        print("\nStep 4c upper bound comparison skipped (metrics json not found).")

    print("===============================================================")


if __name__ == "__main__":
    main()