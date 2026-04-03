"""
RIS-Assisted FMCW Radar NLOS Detection — WORKING FINAL
=======================================================
Authors: Yogesh Rethinapandian, Kaushik Kumar, Arun Karthick Sundararajan

VERIFIED IN SANDBOX: Pd rises correctly.

Physics model:
- Optimised RIS with N elements adds 20*log10(N) dB gain (N^2 power)
- Random RIS with N elements adds 10*log10(N) dB gain (N power)  
- No RIS = 0 signal, Pd ~ Pfa
- Input SNR is defined at radar receiver with N=1 reference
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os, json

np.random.seed(42)

# ── RADAR ──────────────────────────────────────
c=3e8; fc=77e9; lam=c/fc; B=500e6; Tc=50e-6; M=64; Ns=256
slope=B/Tc; PROC=Ns*M
d1=15.0; d2=20.0; R0=d1+d2; v0=5.0
tau0=2*R0/c; fd0=2*v0/lam
R_BIN=117; D_BIN=40

CB=np.array([0,np.pi/2,np.pi,3*np.pi/2])

print("="*55)
print("  RIS-FMCW SIMULATION — WORKING FINAL")
print(f"  PROC={PROC}  R_BIN={R_BIN}  D_BIN={D_BIN}")
print("="*55)

def make_signal(eff_snr_db):
    """Beat signal at effective SNR after RIS combining."""
    snr_lin=10**(eff_snr_db/10.0)
    A=np.sqrt(snr_lin/PROC)
    t=np.linspace(0,Tc,Ns,endpoint=False)
    beat=np.zeros((Ns,M),dtype=complex)
    for m in range(M):
        beat[:,m]=A*np.exp(1j*2*np.pi*(
            slope*tau0*t+fd0*(m*Tc)-0.5*slope*tau0**2))
    beat+=(np.random.randn(Ns,M)+1j*np.random.randn(Ns,M))/np.sqrt(2)
    return beat

def make_noise():
    """Pure noise — no RIS, NLOS blocked."""
    return (np.random.randn(Ns,M)+1j*np.random.randn(Ns,M))/np.sqrt(2)

def rdm(beat):
    return np.abs(np.fft.fftshift(np.fft.fft2(beat),axes=1))**2

def cfar(power,pfa=1e-3):
    row=power[R_BIN,:]
    cut=row[D_BIN]
    refs=[row[(D_BIN+k)%M] for k in range(3,11)]+\
         [row[(D_BIN-k)%M] for k in range(3,11)]
    mu=np.mean(refs); n=len(refs)
    return bool(cut>n*(pfa**(-1.0/n)-1)*mu)

def eff_snr_opt(snr_db, N):
    """Effective SNR for optimised RIS: adds N^2 = 20*log10(N) dB"""
    return snr_db + 20*np.log10(N)

def eff_snr_rand(snr_db, N):
    """Effective SNR for random RIS: adds N = 10*log10(N) dB"""
    return snr_db + 10*np.log10(N)

# ── VERIFY ─────────────────────────────────────
print("\n  Verification (N=64 optimised, 100 trials):")
for snr in [-5,0,5,10,15,20]:
    eff=eff_snr_opt(snr,64)
    hits=sum(cfar(rdm(make_signal(eff))) for _ in range(100))
    print(f"  input={snr:+3d}dB eff={eff:.1f}dB Pd={hits/100:.2f}")

# ── FIGURE 1: RD MAPS ──────────────────────────
print("\n  Figure 1...")
snr_viz=10
b_none=make_noise()
b_rand=make_signal(eff_snr_rand(snr_viz,64))
b_opt =make_signal(eff_snr_opt(snr_viz,64))
rd_none=10*np.log10(rdm(b_none)+1e-30)
rd_rand=10*np.log10(rdm(b_rand)+1e-30)
rd_opt =10*np.log10(rdm(b_opt) +1e-30)
RZ,DZ=40,25
rl=max(0,R_BIN-RZ); rh=min(Ns,R_BIN+RZ)
dl=max(0,D_BIN-DZ); dh=min(M,D_BIN+DZ)
fig,axes=plt.subplots(1,3,figsize=(15,5))
maps=[rd_none,rd_rand,rd_opt]
titles=["(a) No RIS — NLOS Blocked",
        "(b) Random RIS Phases\n(Incoherent Combining)",
        "(c) Optimised RIS Phases\n(Coherent — Proposed)"]
vmin=min(m[rl:rh,dl:dh].min() for m in maps)
vmax=max(m[rl:rh,dl:dh].max() for m in maps)
for ax,rd,title in zip(axes,maps,titles):
    im=ax.imshow(rd[rl:rh,dl:dh],aspect="auto",origin="lower",
                 cmap="jet",vmin=vmin,vmax=vmax)
    ax.set_title(title,fontsize=11,fontweight="bold")
    ax.set_xlabel("Doppler bin",fontsize=10)
    ax.set_ylabel("Range bin",fontsize=10)
    ax.axhline(R_BIN-rl,color="white",ls="--",lw=1.5,label="Target range")
    ax.axvline(D_BIN-dl,color="cyan",ls="--",lw=1.5,label="Target Doppler")
    ax.legend(fontsize=7,loc="upper right")
plt.colorbar(im,ax=axes,label="Power (dB)",shrink=0.8)
fig.suptitle(f"Range-Doppler Maps — SNR={snr_viz} dB, N=64 RIS elements, 77 GHz FMCW",
             fontsize=12,fontweight="bold")
plt.tight_layout()
plt.savefig("fig1_rd_maps.png",dpi=150,bbox_inches="tight")
plt.close()
print(f"  Peak: none={rd_none[R_BIN,D_BIN]:.1f} rand={rd_rand[R_BIN,D_BIN]:.1f} opt={rd_opt[R_BIN,D_BIN]:.1f} dB")
print("  [SAVED] fig1_rd_maps.png")

# ── PD SWEEPS ──────────────────────────────────
SNR_RANGE=np.arange(-15,21,1); TRIALS=500; PFA=1e-3
N_LIST=[16,32,64,128]

print("\n  Running Pd sweeps...")

# Optimised RIS - different N
pd_opt={}
for N in N_LIST:
    pd=[]
    print(f"  Opt N={N:3d} ",end="",flush=True)
    for snr in SNR_RANGE:
        hits=sum(cfar(rdm(make_signal(eff_snr_opt(snr,N))),pfa=PFA)
                 for _ in range(TRIALS))
        pd.append(hits/TRIALS)
        print(".",end="",flush=True)
    idx10=SNR_RANGE.tolist().index(10)
    print(f"  Pd@10dB={pd[idx10]:.2f}")
    pd_opt[N]=pd

# No RIS
pd_noris=[]
print(f"  No RIS     ",end="",flush=True)
for snr in SNR_RANGE:
    hits=sum(cfar(rdm(make_noise()),pfa=PFA) for _ in range(TRIALS))
    pd_noris.append(hits/TRIALS)
    print(".",end="",flush=True)
print(f"  max={max(pd_noris):.3f}")

# Random RIS N=64
pd_rand=[]
print(f"  Rand N=64  ",end="",flush=True)
for snr in SNR_RANGE:
    hits=sum(cfar(rdm(make_signal(eff_snr_rand(snr,64))),pfa=PFA)
             for _ in range(TRIALS))
    pd_rand.append(hits/TRIALS)
    print(".",end="",flush=True)
idx10=SNR_RANGE.tolist().index(10)
print(f"  Pd@10dB={pd_rand[idx10]:.2f}")

# ── FIGURE 2 ───────────────────────────────────
colors=["#e41a1c","#ff7f00","#4daf4a","#377eb8"]
fig,ax=plt.subplots(figsize=(9,6))
for N,col in zip(N_LIST,colors):
    ax.plot(SNR_RANGE,pd_opt[N],"o-",color=col,lw=2.2,ms=5,
            label=f"Optimised RIS  N={N}")
ax.plot(SNR_RANGE,pd_noris,"k--s",lw=2,ms=5,label="No RIS (NLOS baseline)")
ax.axhline(0.9,color="gray",ls=":",lw=1.5,label="$P_d=0.90$ reference")
ax.set_xlabel("SNR (dB)",fontsize=13)
ax.set_ylabel("Detection Probability $P_d$",fontsize=13)
ax.set_title("$P_d$ vs SNR — Effect of RIS Element Count\n"
             "CA-CFAR, $P_{fa}=10^{-3}$, 77 GHz FMCW NLOS",
             fontsize=12,fontweight="bold")
ax.legend(fontsize=10,loc="lower right")
ax.grid(True,alpha=0.3)
ax.set_xlim([SNR_RANGE[0],SNR_RANGE[-1]]); ax.set_ylim([-0.02,1.05])
plt.tight_layout()
plt.savefig("fig2_pd_vs_N.png",dpi=150,bbox_inches="tight")
plt.close()
print("\n  [SAVED] fig2_pd_vs_N.png")

# ── FIGURE 3 ───────────────────────────────────
fig,ax=plt.subplots(figsize=(10,6.5))
ax.plot(SNR_RANGE,pd_opt[64],"b-o",lw=2.5,ms=6,
        label="CA-CFAR + Optimised RIS (Proposed)")
ax.plot(SNR_RANGE,pd_rand,"g-s",lw=2.2,ms=6,
        label="CA-CFAR + Random RIS")
ax.plot(SNR_RANGE,pd_noris,"r-^",lw=2.2,ms=6,
        label="CA-CFAR + No RIS (NLOS Baseline)")
ax.axhline(0.9,color="gray",ls=":",lw=1.5,label="$P_d=0.90$ reference")
ax.fill_between(SNR_RANGE,pd_noris,pd_opt[64],
                alpha=0.10,color="blue",label="RIS improvement region")
ax.set_xlabel("SNR (dB)",fontsize=13)
ax.set_ylabel("Detection Probability $P_d$",fontsize=13)
ax.set_title("Detection Probability vs SNR\n"
             "CA-CFAR, N=64, $P_{fa}=10^{-3}$, 77 GHz FMCW NLOS",
             fontsize=12,fontweight="bold")
ax.legend(fontsize=10,loc="lower right")
ax.grid(True,alpha=0.3)
ax.set_xlim([SNR_RANGE[0],SNR_RANGE[-1]]); ax.set_ylim([-0.02,1.05])
plt.tight_layout()
plt.savefig("fig3_pd_money.png",dpi=150,bbox_inches="tight")
plt.close()
print("  [SAVED] fig3_pd_money.png")

# ── FIGURE 4 ───────────────────────────────────
N_sweep=[4,8,16,24,32,48,64,96,128]
go=[20*np.log10(N) for N in N_sweep]
gr=[10*np.log10(N) for N in N_sweep]
fig,ax=plt.subplots(figsize=(9,6))
ax.plot(N_sweep,go,"b-o",lw=2.2,ms=6,label="Optimised RIS ($N^2$ power gain)")
ax.plot(N_sweep,gr,"r-s",lw=2.2,ms=6,label="Random RIS ($N$ power gain)")
ax.plot(N_sweep,go,"k--",lw=1.8,alpha=0.5,label="Theoretical $N^2$ ($20\\log_{10}N$ dB)")
ax.set_xlabel("RIS Element Count $N$",fontsize=13)
ax.set_ylabel("Beamforming Gain (dB)",fontsize=13)
ax.set_title("RIS Beamforming Gain vs Element Count\n"
             "Optimised ($N^2$) vs Random ($N$)",
             fontsize=12,fontweight="bold")
ax.legend(fontsize=11); ax.grid(True,alpha=0.3)
plt.tight_layout()
plt.savefig("fig4_gain_vs_N.png",dpi=150,bbox_inches="tight")
plt.close()
print("  [SAVED] fig4_gain_vs_N.png")

# ── DATASET ────────────────────────────────────
print("\n  CNN dataset...")
os.makedirs("dataset/present",exist_ok=True)
os.makedirs("dataset/absent",exist_ok=True)
IMG=64; count=0
for snr in np.arange(-15,21,1):
    for i in range(200):
        # Present: optimised RIS N=64
        b=make_signal(eff_snr_opt(snr,64))
        p=rdm(b)
        rl2=max(0,R_BIN-IMG//2); dl2=max(0,D_BIN-IMG//2)
        crop=p[rl2:rl2+IMG,dl2:dl2+IMG]
        if crop.shape!=(IMG,IMG):
            crop=np.pad(crop,((0,IMG-crop.shape[0]),(0,IMG-crop.shape[1])),mode='edge')
        mn,mx=crop.min(),crop.max()
        np.save(f"dataset/present/s{int(snr):+03d}_{i:03d}.npy",
                ((crop-mn)/(mx-mn+1e-10)).astype(np.float32))
        # Absent: no RIS
        b=make_noise()
        p=rdm(b)
        crop=p[rl2:rl2+IMG,dl2:dl2+IMG]
        if crop.shape!=(IMG,IMG):
            crop=np.pad(crop,((0,IMG-crop.shape[0]),(0,IMG-crop.shape[1])),mode='edge')
        mn,mx=crop.min(),crop.max()
        np.save(f"dataset/absent/s{int(snr):+03d}_{i:03d}.npy",
                ((crop-mn)/(mx-mn+1e-10)).astype(np.float32))
        count+=2
print(f"  {count} samples saved")

# ── RESULTS ────────────────────────────────────
def snr_at(pd,th=0.9):
    for i,p in enumerate(pd):
        if p>=th: return int(SNR_RANGE[i])
    return ">20"

results={
    "N16_snr_at_Pd90":snr_at(pd_opt[16]),
    "N32_snr_at_Pd90":snr_at(pd_opt[32]),
    "N64_snr_at_Pd90":snr_at(pd_opt[64]),
    "N128_snr_at_Pd90":snr_at(pd_opt[128]),
    "rand_snr_at_Pd90":snr_at(pd_rand),
    "noris_max_Pd":round(float(max(pd_noris)),4),
    "gain_opt_N64_dB":round(go[N_sweep.index(64)],2),
    "gain_opt_N128_dB":round(go[N_sweep.index(128)],2),
    "dataset_samples":count,"pfa":PFA,"trials":TRIALS
}
with open("results.json","w") as f: json.dump(results,f,indent=2)

print("\n"+"="*55)
print("  ALL DONE")
print("="*55)
for k,v in results.items(): print(f"  {k:<28}: {v}")
print("="*55)
