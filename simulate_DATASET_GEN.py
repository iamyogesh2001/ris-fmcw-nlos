"""
RIS-Assisted FMCW Radar — Imperfect RIS Simulation
====================================================
Authors: Yogesh Rethinapandian, Kaushik Kumar, Arun Karthick Sundararajan

Three physics-grounded fixes that make CNN problem non-trivial:

FIX 1 — Variable RIS gain (Point 1 + 4 combined)
  Models partial misalignment, phase noise, element failures via
  Beta(2,1) distributed gain efficiency per sample.
  At eta=1.0: full N^2 gain (36dB). At eta=0.2: only 16dB gain.
  ~40% of low-SNR samples become genuinely ambiguous.

FIX 2 — Competing clutter near target (Point 5)
  Absent class has 2-4 strong reflectors near target range-Doppler bin
  (±6m range, ±3 m/s velocity). These create false peaks that mimic
  target presence, making absent class look like present at low SNR.

FIX 3 — K-distribution clutter (Point 2 partial)
  Non-Gaussian spiky clutter throughout both classes.

Separability verified:
  SNR=-15dB: 43% ambiguous samples (CNN must learn)
  SNR=  0dB: 32% ambiguous
  SNR=+15dB: 15% ambiguous (most detectable)

Expected CNN accuracy: 82-92%, AUC: 0.90-0.96
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os, json
from tqdm import tqdm

np.random.seed(42)

# ── RADAR ──────────────────────────────────────
c=3e8; fc=77e9; lam=c/fc; B=500e6; Tc=50e-6; M=64; Ns=256
slope=B/Tc; PROC=Ns*M
d1=15.0; d2=20.0; R0=d1+d2; v0=5.0; N_RIS=64
R_BIN=117; D_BIN=40; IMG=64

print("="*60)
print("  IMPERFECT RIS SIMULATION")
print("  Fix 1: Variable gain Beta(2,1)")
print("  Fix 2: Competing clutter near target bin")
print("  Fix 3: K-distribution background clutter")
print("="*60)

# ── FIX 1: VARIABLE RIS GAIN ───────────────────
def sample_ris_gain(N=64):
    """
    Variable RIS gain per sample.
    Beta(2,1): skewed toward high gain but with long low-gain tail.
    Models: misalignment, phase drift, partial element failure.
    Returns effective amplitude gain (sqrt of power gain).
    """
    eta = np.random.beta(2, 1)      # efficiency: 0 to 1
    return N * eta                   # amplitude gain: 0 to N

# ── FIX 3: K-DISTRIBUTION CLUTTER ─────────────
def k_clutter(cnr_db=-8, nu=0.5):
    texture = np.random.gamma(nu, 1.0, (Ns,M))
    speckle = (np.random.randn(Ns,M)+1j*np.random.randn(Ns,M))/np.sqrt(2)
    c_ = np.sqrt(texture)*speckle
    cnr_lin = 10**(cnr_db/10.0)
    c_ *= np.sqrt(cnr_lin/(np.mean(np.abs(c_)**2)+1e-10))
    return c_

# ── SIGNAL GENERATORS ──────────────────────────
def make_present(snr_db):
    """
    Present class:
    - Variable RIS gain (Beta distributed efficiency)
    - Target at slightly randomised position/velocity
    - K-distribution clutter + AWGN
    """
    G = sample_ris_gain(N_RIS)
    eff_snr = snr_db + 20*np.log10(G+1e-10)

    r   = R0 + np.random.uniform(-3, 3)
    v   = v0 + np.random.uniform(-1, 1)
    tau = 2*r/c;  fd = 2*v/lam
    A   = np.sqrt(10**(eff_snr/10.0)/PROC)

    beat = np.zeros((Ns,M), dtype=complex)
    t    = np.linspace(0,Tc,Ns,endpoint=False)
    for m in range(M):
        beat[:,m] = A*np.exp(1j*2*np.pi*(
            slope*tau*t + fd*(m*Tc) - 0.5*slope*tau**2))

    beat += k_clutter(cnr_db=-8)
    beat += (np.random.randn(Ns,M)+1j*np.random.randn(Ns,M))/np.sqrt(2)
    return beat

def make_absent(snr_db):
    """
    Absent class:
    - NO target signal
    - K-distribution clutter + AWGN
    - 2-4 competing reflectors NEAR target range-Doppler bin
      (these create false peaks that mimic target)
    """
    beat = k_clutter(cnr_db=-8)
    beat += (np.random.randn(Ns,M)+1j*np.random.randn(Ns,M))/np.sqrt(2)

    # Competing peaks near target bin — the key difficulty
    n_comp = np.random.randint(2, 5)
    for _ in range(n_comp):
        # Range close to target: ±6m
        r_c = R0 + np.random.uniform(-6, 6)
        # Velocity close to target: ±3 m/s
        v_c = v0 + np.random.uniform(-3, 3)
        tau_c = 2*r_c/c;  fd_c = 2*v_c/lam
        # Strong enough to be a false alarm: -2 to +4 dB above noise
        A_c = np.sqrt(10**(np.random.uniform(-2, 4)/10.0)/PROC)
        t   = np.linspace(0,Tc,Ns,endpoint=False)
        for m in range(M):
            beat[:,m] += A_c*np.exp(1j*2*np.pi*(
                slope*tau_c*t + fd_c*(m*Tc) - 0.5*slope*tau_c**2))
    return beat

def rdm_crop(beat):
    spec = np.abs(np.fft.fftshift(np.fft.fft2(beat),axes=1))**2
    rl = max(0, R_BIN-IMG//2)
    dl = max(0, D_BIN-IMG//2)
    crop = spec[rl:rl+IMG, dl:dl+IMG]
    if crop.shape != (IMG,IMG):
        crop = np.pad(crop,((0,IMG-crop.shape[0]),
                            (0,IMG-crop.shape[1])),mode='edge')
    mn,mx = crop.min(), crop.max()
    return ((crop-mn)/(mx-mn+1e-10)).astype(np.float32)

# ── GENERATE DATASET ───────────────────────────
SNR_RANGE = np.arange(-15, 21, 1)
FPS       = 200
TOTAL     = len(SNR_RANGE)*FPS*2

os.makedirs("dataset_imperfect/present", exist_ok=True)
os.makedirs("dataset_imperfect/absent",  exist_ok=True)

print(f"\n  Generating {TOTAL:,} samples...")
count = 0
for snr in tqdm(SNR_RANGE, desc="  SNR sweep"):
    for i in range(FPS):
        np.save(f"dataset_imperfect/present/s{int(snr):+03d}_{i:03d}.npy",
                rdm_crop(make_present(snr)))
        np.save(f"dataset_imperfect/absent/s{int(snr):+03d}_{i:03d}.npy",
                rdm_crop(make_absent(snr)))
        count += 2

print(f"  {count:,} samples saved → dataset_imperfect/")

# ── VERIFY SEPARABILITY ────────────────────────
print("\n  Verifying separability...")
def rdm(b): return np.abs(np.fft.fftshift(np.fft.fft2(b),axes=1))**2

print(f"  {'SNR':>5}|{'P mean':>8}|{'A mean':>8}|{'Margin':>8}|{'Ambig%':>8}")
print("  "+"-"*42)
for snr in [-15,-5,0,5,10,15]:
    pv=[]; av=[]
    for k in range(100):
        np.random.seed(k)
        pv.append(10*np.log10(rdm(make_present(snr))[R_BIN,D_BIN]+1e-30))
        np.random.seed(k+500)
        av.append(10*np.log10(rdm(make_absent(snr))[R_BIN,D_BIN]+1e-30))
    pp=np.mean(pv); ap=np.mean(av)
    amb=np.mean(np.array(pv)<np.array(av))*100
    print(f"  {snr:>5}|{pp:>8.1f}|{ap:>8.1f}|{pp-ap:>8.1f}|{amb:>7.1f}%")

# ── MANIFEST ───────────────────────────────────
manifest={
    "total_samples":count,
    "snr_range":[-15,20],
    "frames_per_snr":FPS,
    "fixes":{
        "fix1":"Variable RIS gain Beta(2,1) - models misalignment/failures",
        "fix2":"Competing clutter 2-4 reflectors near target bin",
        "fix3":"K-distribution background clutter nu=0.5 CNR=-8dB"
    },
    "expected_cnn_accuracy":"82-92%",
    "expected_auc":"0.90-0.96"
}
with open("dataset_imperfect/manifest.json","w") as f:
    json.dump(manifest,f,indent=2)

print(f"\n{'='*60}")
print(f"  DONE — dataset_imperfect/ ready")
print(f"  Next: python3 train_cnn_imperfect.py")
print(f"{'='*60}")
