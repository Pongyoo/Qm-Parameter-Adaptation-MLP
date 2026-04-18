# ============================================================
# stepQm1p_exp3_build_params_from_sweep_fft.py
# Build fn, Qm from EXP3 sweep using Welch H1 + coherence
# Output json will be used by StepQm2
# ============================================================

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch, csd
from nptdms import TdmsFile

# ================= CONFIG =================
ROOT = r"C:\Users\ploy\Desktop"

SWEEP_TDMS_PATH = r"C:\Users\ploy\Desktop\UTHwithT\data\1.5_swp30s_exp5.tdms"
SESSION_TAG = "exp5_1.5_fix18.0"
SESSION_ROOT = os.path.join(ROOT, "ML", "GRU_2", "processed_Qm_sessions", SESSION_TAG)

OUT_DIR = os.path.join(SESSION_ROOT, "stepQm1p_params_sweep")
os.makedirs(OUT_DIR, exist_ok=True)

LEVEL_NAME = "1.5-swp30s"
GROUP_NAME = "Log"

FS = 100000
F_MIN, F_MAX = 15000, 25000
NFFT = 16384
NSEG = 8192
COH_TH = 0.80
EPS = 1e-12
# ==========================================


def find_3db_bandwidth(f, mag, i_pk):
    peak = float(mag[i_pk])
    thr = peak / np.sqrt(2.0)

    iL = i_pk
    while iL > 0 and mag[iL] >= thr:
        iL -= 1

    iR = i_pk
    while iR < len(mag) - 1 and mag[iR] >= thr:
        iR += 1

    fL = float(f[iL])
    fR = float(f[iR])
    BW = float(max(EPS, fR - fL))
    return fL, fR, BW, thr, peak


print("\n================ StepQm1p EXP3: Sweep Params =================")
print("SWEEP_TDMS_PATH =", SWEEP_TDMS_PATH)
print("EXISTS?         =", os.path.isfile(SWEEP_TDMS_PATH))

if not os.path.isfile(SWEEP_TDMS_PATH):
    raise FileNotFoundError(SWEEP_TDMS_PATH)

# ---- read TDMS directly ----
tdms = TdmsFile.read(SWEEP_TDMS_PATH)
group = tdms[GROUP_NAME]

cur = np.asarray(group["Current"][:], dtype=np.float64)
disp = np.asarray(group["Displacement"][:], dtype=np.float64)

m = np.isfinite(cur) & np.isfinite(disp)
cur, disp = cur[m], disp[m]

print("Samples =", len(cur), "| Duration =", len(cur) / FS, "sec")

# ---- Welch FRF (H1) ----
f, Sxx = welch(cur, fs=FS, nperseg=NSEG, nfft=NFFT)
_, Syy = welch(disp, fs=FS, nperseg=NSEG, nfft=NFFT)
_, Sxy = csd(cur, disp, fs=FS, nperseg=NSEG, nfft=NFFT)

H1 = Sxy / (Sxx + EPS)
coh = np.abs(Sxy) ** 2 / ((Sxx + EPS) * (Syy + EPS))
mag = np.abs(H1)

# ---- frequency band ----
band = (f >= F_MIN) & (f <= F_MAX)
f_b = f[band]
mag_b = mag[band]
coh_b = coh[band]

ok = coh_b >= COH_TH
if ok.sum() < 10:
    raise RuntimeError("Not enough points pass coherence threshold.")

f_ok = f_b[ok]
mag_ok = mag_b[ok]

idx = np.argsort(f_ok)
f_s = f_ok[idx]
m_s = mag_ok[idx]

# ---- peak and Qm ----
i_pk = int(np.argmax(m_s))
fn = float(f_s[i_pk])

fL, fR, BW, thr, peak = find_3db_bandwidth(f_s, m_s, i_pk)
Qm = float(fn / BW)

params = {
    "level_name": LEVEL_NAME,
    "fs_hz": FS,
    "method": "Welch_H1",
    "f_range_hz": [F_MIN, F_MAX],
    "nfft": NFFT,
    "nperseg": NSEG,
    "coh_threshold": COH_TH,
    "fn_hz": fn,
    "bw_hz": BW,
    "Qm": Qm,
    "fL_hz": fL,
    "fR_hz": fR,
    "peak_mag": peak,
    "minus3db_level": thr
}

json_path = os.path.join(OUT_DIR, f"params_{LEVEL_NAME}.json")
with open(json_path, "w", encoding="utf-8") as fp:
    json.dump(params, fp, indent=2)

print(f"fn = {fn:.3f} Hz | BW = {BW:.3f} Hz | Qm = {Qm:.3f}")
print("Saved json:", json_path)

# ---- plot ----
fig = plt.figure(figsize=(10, 6), dpi=140)

ax1 = fig.add_subplot(2, 1, 1)
ax1.plot(f_s, m_s, linewidth=1.5, color="blue")
ax1.axhline(thr, color="gray", linestyle="--", linewidth=1.2, label="-3 dB")
ax1.axvline(fn, color="red", linestyle="--", linewidth=1.6, label="fn")
ax1.axvline(fL, color="green", linestyle="--", linewidth=1.4, label="f1")
ax1.axvline(fR, color="green", linestyle="--", linewidth=1.4, label="f2")
ax1.set_title("FRF Magnitude (coherence-passed region)")
ax1.set_ylabel("|H1|")
ax1.grid(True, alpha=0.3)
txt = f"fn = {fn:.2f} Hz\nBW = {BW:.2f} Hz\nQm = {Qm:.2f}"
ax1.text(0.02, 0.95, txt, transform=ax1.transAxes, va="top", ha="left",
         bbox=dict(facecolor="white", alpha=0.7))
xpad = 1500
ax1.set_xlim(max(F_MIN, fn - xpad), min(F_MAX, fn + xpad))

ax2 = fig.add_subplot(2, 1, 2, sharex=ax1)
ax2.plot(f_b, coh_b, linewidth=1.2, color="tab:blue")
ax2.axhline(COH_TH, color="red", linestyle="--", linewidth=1.4, label="coh threshold")
ax2.set_xlabel("Frequency (Hz)")
ax2.set_ylabel("Coherence")
ax2.set_ylim(0, 1.05)
ax2.grid(True, alpha=0.3)

fig.tight_layout()
plot_path = os.path.join(OUT_DIR, f"plot_frfcoh_{LEVEL_NAME}.png")
fig.savefig(plot_path)
plt.close(fig)

print("Saved plot:", plot_path)
print("Done StepQm1p EXP3")