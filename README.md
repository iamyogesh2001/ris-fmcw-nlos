# RIS-Assisted FMCW Radar NLOS Detection

**Paper:** *RIS-Assisted FMCW Radar Target Detection in NLOS Environments via CNN-Based Range-Doppler Processing*

**Authors:** Yogesh Rethinapandian¹, Kaushik Kumar², Arun Karthik Sundararajan³

¹ Department of Electrical and Computer Engineering, University of Illinois at Chicago (yrethi2@uic.edu)
² Information School, University of Arizona (kaushikkumar@arizona.edu)
³ Independent Researcher, College Station, TX

**Submitted to:** IEEE Antennas and Wireless Propagation Letters (AWPL)

---

## Overview

This repository contains the full simulation framework, dataset generation pipeline, CNN training code, and paper figures for our work on RIS-assisted FMCW radar target detection in non-line-of-sight (NLOS) environments.

We propose a **reconfigurable intelligent surface (RIS) passive antenna aperture** architecture that redirects 77 GHz FMCW radar energy around physical obstructions toward NLOS targets. A 64-element RIS with 2-bit phase quantization achieves **36.1 dB coherent aperture gain**, enabling CA-CFAR detection at SNR as low as −15 dB where conventional radar yields detection probability below 0.6%. A lightweight CNN trained on range-Doppler maps under realistic RIS impairment conditions achieves **98.33% accuracy and AUC of 0.988**.

---

## Key Results

| Metric | Value |
|--------|-------|
| RIS aperture gain, optimised N=64 | 36.1 dB |
| RIS aperture gain, random N=64 | 18.1 dB |
| Coherence advantage (opt. vs. rand.) | 18.0 dB |
| SNR at Pd=0.9, N=128, optimised | −15 dB |
| SNR at Pd=0.9, N=64, random | −3 dB |
| Max Pd, no RIS (CA-CFAR) | 0.006 |
| CNN test accuracy | 98.33% |
| CNN AUC | 0.988 |
| False alarm rate | 0.14% |
| Best training epoch | 54/60 |
| CNN parameters | 336,865 |

---

## Repository Structure

```
ris-fmcw-nlos/
│
├── README.md                        # This file
│
├── scripts/
│   ├── simulate_ris_fmcw.py         # Main FMCW + RIS simulation (Pd curves, RDMs, gain)
│   ├── generate_dataset.py          # Dataset generation with imperfect RIS model
│   └── train_cnn.py                 # CNN training, evaluation, ROC, confusion matrix
│
├── figures/
│   ├── fig1_rd_maps.png             # Range-Doppler maps: No RIS / Random / Optimised
│   ├── fig2_pd_vs_N.png             # Pd vs SNR for N=16,32,64,128
│   ├── fig3_pd_money.png            # Optimised vs Random vs No RIS at N=64
│   ├── fig4_gain_vs_N_fixed.png     # Beamforming gain: N² vs N scaling
│   ├── fig5_training_curves_fixed.png # CNN training and validation curves
│   └── fig6_roc_confusion.png       # ROC curve (AUC=0.988) + confusion matrix
│
└── paper/
    └── main.tex                     # Final IEEE AWPL LaTeX source
```

---

## System Parameters

### FMCW Radar
| Parameter | Value |
|-----------|-------|
| Carrier frequency | 77 GHz |
| Sweep bandwidth | 500 MHz |
| Chirp period | 50 µs |
| Chirps per frame | 64 |
| ADC samples per chirp | 256 |
| Range resolution | 0.30 m |
| Velocity resolution | 0.31 m/s |

### RIS Configuration
| Parameter | Value |
|-----------|-------|
| Array size | 8×8 UPA (N=64 elements) |
| Element spacing | Half-wavelength |
| Phase quantization | 2-bit (0, π/2, π, 3π/2) |
| Radar-to-RIS distance (d1) | 15 m |
| RIS-to-target distance (d2) | 20 m |
| Optimised aperture gain (N=64) | 36.1 dB |

### RIS Hardware Impairment Model
Variable aperture efficiency per sample: η ~ Beta(2,1), mean = 2/3, giving 3.5 dB mean gain reduction from ideal N² bound.

### Clutter Model
- K-distribution clutter: g ~ Gamma(0.5, 1), CNR = −8 dB
- 2–4 competing multipath reflectors near target bin
- Unit-variance AWGN throughout

---

## Dataset

**14,400 labeled 64×64 RDM crops**
- Classes: Present (optimised RIS + impairments) / Absent (clutter only)
- SNR range: −15 to +20 dB (36 levels, 200 frames/level/class)
- Split: 80% train / 10% validation / 10% test
- At −15 dB SNR: 43% sample-level class ambiguity (genuine difficulty)

To regenerate the dataset from scratch:
```bash
python scripts/generate_dataset.py
```
This will create a `dataset_imperfect/` directory with `present/` and `absent/` subfolders.

---

## CNN Architecture

Three convolutional blocks (channel depths 32, 64, 128; 3×3 kernels; batch norm; ReLU; 2×2 max pool; spatial dropout 0.3), global average pooling, FC layers (256→64→1), sigmoid output. **Total: 336,865 parameters.**

### Training Configuration
- Optimizer: Adam (lr=1e-4, weight decay=5e-3)
- LR schedule: Cosine annealing (1e-4 → 1e-5)
- Label smoothing: ε=0.1
- Batch size: 32
- Epochs: 60 (early stopping patience=15)
- Augmentation: random flips, Gaussian noise (σ=0.06)

---

## Running the Code

### Requirements
```bash
pip install numpy torch torchvision matplotlib scikit-learn scipy
```

### Step 1 — Run simulation (Pd curves, RDMs, gain figures)
```bash
cd scripts
python simulate_ris_fmcw.py
```
Outputs: `fig1_rd_maps.png`, `fig2_pd_vs_N.png`, `fig3_pd_money.png`, `fig4_gain_vs_N_fixed.png`

### Step 2 — Generate dataset
```bash
python generate_dataset.py
```
Outputs: `dataset_imperfect/present/` and `dataset_imperfect/absent/` — ~14,400 `.npy` files

### Step 3 — Train CNN and evaluate
```bash
python train_cnn.py
```
Outputs: `fig5_training_curves_fixed.png`, `fig6_roc_confusion.png`, `cnn_results.json`, `best_cnn_model.pth`

---

## Figures

### Fig 1 — Range-Doppler Maps
![RDMs](figures/fig1_rd_maps.png)
No RIS (a): target invisible. Random RIS (b): faint incoherent signature. Optimised RIS (c): sharp coherent peak at bin (117, 40) exceeding noise by >15 dB.

### Fig 2 — Pd vs SNR (Effect of Element Count)
![Pd vs N](figures/fig2_pd_vs_N.png)
CA-CFAR detection probability for N=16,32,64,128. No-RIS baseline remains below 0.6% throughout.

### Fig 3 — Optimised vs Random vs No RIS
![Pd comparison](figures/fig3_pd_money.png)
Optimised RIS achieves near-perfect detection from −15 dB. Random RIS lags by 12 dB. No-RIS fails completely.

### Fig 4 — Beamforming Gain vs Element Count
![Gain vs N](figures/fig4_gain_vs_N_fixed.png)
Optimised RIS tracks the N² bound (36.1 dB at N=64). Random configuration achieves N gain (18.1 dB), an 18 dB deficit from loss of aperture coherence.

### Fig 5 — CNN Training Curves
![Training](figures/fig5_training_curves_fixed.png)
Loss above 0.1 throughout 60 epochs (label smoothing working). Val loss below train loss confirms no overfitting. Best epoch: 54.

### Fig 6 — ROC Curve and Confusion Matrix
![ROC](figures/fig6_roc_confusion.png)
AUC = 0.988. Test accuracy = 98.33% on 1,440 held-out samples. False alarm rate: 0.14% (1 false alarm out of 720 absent-class samples).

---

## Citation

If you use this code or dataset in your research, please cite:

```bibtex
@article{rethinapandian2025ris,
  title={RIS-Assisted FMCW Radar Target Detection in NLOS Environments
         via CNN-Based Range-Doppler Processing},
  author={Rethinapandian, Yogesh and Kumar, Kaushik and
          Sundararajan, Arun Karthik},
  journal={IEEE Antennas and Wireless Propagation Letters},
  year={2025},
  note={Under review}
}
```

---

## License

MIT License. See `LICENSE` for details.

---

## Contact

Yogesh Rethinapandian — yrethi2@uic.edu
University of Illinois at Chicago, Department of ECE
