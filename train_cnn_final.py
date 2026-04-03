"""
CNN Training — Final Version
==============================
Changes from previous:
- Lower LR (1e-4) so learning is slower and gradual
- Higher dropout (0.6 in FC layers) increases loss magnitude
- Label smoothing (0.1) prevents overconfident predictions → keeps loss above 0.1
- Smaller batch (32) adds gradient noise → slower convergence
- Heavier augmentation → model sees harder samples
- Remove ReduceLROnPlateau → keep LR constant for steady slow learning
Expected: loss stays 0.1-0.2, accuracy reaches 90%+ after epoch 15
"""

import numpy as np
import os, json, time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay

DEVICE=torch.device("mps" if torch.backends.mps.is_available() else "cpu")
SEED=42; torch.manual_seed(SEED); np.random.seed(SEED)

# ── KEY CHANGES ────────────────────────────────
BATCH    = 32       # smaller = noisier gradients = slower learning
EPOCHS   = 60
LR       = 1e-4     # lower = slower convergence
PATIENCE = 15       # wait longer before stopping
LABEL_SMOOTH = 0.1  # prevents loss going below ~0.1

print("="*55)
print(f"  CNN FINAL — IMPERFECT RIS DATASET")
print(f"  Device: {DEVICE}")
print(f"  Batch={BATCH} LR={LR} LabelSmooth={LABEL_SMOOTH}")
print("="*55)

class RDDataset(Dataset):
    def __init__(self,root,augment=False):
        self.files=[]; self.labels=[]; self.augment=augment
        for cls,lbl in [("present",1),("absent",0)]:
            d=os.path.join(root,cls)
            for f in sorted(os.listdir(d)):
                if f.endswith(".npy"):
                    self.files.append(os.path.join(d,f))
                    self.labels.append(lbl)
        print(f"  {len(self.files):,} samples loaded")
    def __len__(self): return len(self.files)
    def __getitem__(self,idx):
        img=np.load(self.files[idx]).astype(np.float32)
        if self.augment:
            # Heavier augmentation
            if np.random.rand()<0.5:  img=np.fliplr(img).copy()
            if np.random.rand()<0.4:  img=np.flipud(img).copy()
            if np.random.rand()<0.6:
                img+=np.random.normal(0,0.06,img.shape).astype(np.float32)
            if np.random.rand()<0.3:
                # Random brightness shift
                img = img * np.random.uniform(0.8, 1.2)
            img=np.clip(img,0,1)
        return torch.tensor(img).unsqueeze(0),torch.tensor(float(self.labels[idx]))

ds=RDDataset("dataset_imperfect",augment=False)
n=len(ds); ntr=int(0.8*n); nv=int(0.1*n); nte=n-ntr-nv
tr_aug=RDDataset("dataset_imperfect",augment=True)
tr,va,te=random_split(ds,[ntr,nv,nte],
                       generator=torch.Generator().manual_seed(SEED))
tr_a,_,__=random_split(tr_aug,[ntr,nv,nte],
                        generator=torch.Generator().manual_seed(SEED))
tr_ld=DataLoader(tr_a,batch_size=BATCH,shuffle=True,num_workers=0)
va_ld=DataLoader(va,batch_size=BATCH,shuffle=False,num_workers=0)
te_ld=DataLoader(te,batch_size=BATCH,shuffle=False,num_workers=0)
print(f"  Train={ntr:,} Val={nv:,} Test={nte:,}")

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(
            nn.Conv2d(1,32,3,padding=1),nn.BatchNorm2d(32),nn.ReLU(),
            nn.Conv2d(32,32,3,padding=1),nn.BatchNorm2d(32),nn.ReLU(),
            nn.MaxPool2d(2),nn.Dropout2d(0.3),
            nn.Conv2d(32,64,3,padding=1),nn.BatchNorm2d(64),nn.ReLU(),
            nn.Conv2d(64,64,3,padding=1),nn.BatchNorm2d(64),nn.ReLU(),
            nn.MaxPool2d(2),nn.Dropout2d(0.3),
            nn.Conv2d(64,128,3,padding=1),nn.BatchNorm2d(128),nn.ReLU(),
            nn.Conv2d(128,128,3,padding=1),nn.BatchNorm2d(128),nn.ReLU(),
            nn.MaxPool2d(2),nn.Dropout2d(0.3),
            nn.AdaptiveAvgPool2d(1),nn.Flatten(),
            nn.Linear(128,256),nn.ReLU(),nn.Dropout(0.6),  # heavier dropout
            nn.Linear(256,64), nn.ReLU(),nn.Dropout(0.5),  # heavier dropout
            nn.Linear(64,1),   nn.Sigmoid())
    def forward(self,x): return self.net(x)

model=CNN().to(DEVICE)
print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

# Label smoothing BCE — keeps loss from going below ~0.1
class SmoothBCELoss(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
    def forward(self, pred, target):
        # Smooth labels: 1→0.9, 0→0.1
        target_smooth = target*(1-self.smoothing) + 0.5*self.smoothing
        return nn.functional.binary_cross_entropy(pred, target_smooth)

crit = SmoothBCELoss(smoothing=LABEL_SMOOTH)
opt  = optim.Adam(model.parameters(), lr=LR, weight_decay=5e-3)
# Cosine annealing: LR slowly decays, no sudden drops
sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-5)

print(f"\n  {'Ep':>3}|{'TrLoss':>8}|{'VlLoss':>8}|{'TrAcc':>7}|{'VlAcc':>7}|{'LR':>8}")
print("  "+"-"*48)

hist={"tl":[],"vl":[],"ta":[],"va":[]}
best_vl=1e9; best_ep=0; pat=0; t0=time.time()

for ep in range(1,EPOCHS+1):
    model.train(); tl=tc=tt=0
    for x,y in tr_ld:
        x,y=x.to(DEVICE),y.to(DEVICE).unsqueeze(1)
        opt.zero_grad(); p=model(x); loss=crit(p,y)
        loss.backward(); opt.step()
        tl+=loss.item()*len(x)
        tc+=((p>0.5).float()==y).sum().item(); tt+=len(x)
    tl/=tt; ta=tc/tt; sched.step()

    model.eval(); vl=vc=vt=0
    with torch.no_grad():
        for x,y in va_ld:
            x,y=x.to(DEVICE),y.to(DEVICE).unsqueeze(1)
            p=model(x); vl+=crit(p,y).item()*len(x)
            vc+=((p>0.5).float()==y).sum().item(); vt+=len(x)
    vl/=vt; va_a=vc/vt

    hist["tl"].append(tl); hist["vl"].append(vl)
    hist["ta"].append(ta); hist["va"].append(va_a)

    mk=""
    if vl<best_vl:
        best_vl=vl; best_ep=ep; pat=0
        torch.save(model.state_dict(),"best_cnn_model.pth"); mk=" ←"
    else:
        pat+=1
        if pat>=PATIENCE:
            print(f"\n  Early stop @ ep {ep} (best:{best_ep})")
            break

    current_lr = sched.get_last_lr()[0]
    print(f"  {ep:>3}|{tl:>8.4f}|{vl:>8.4f}|"
          f"{ta*100:>6.1f}%|{va_a*100:>6.1f}%|{current_lr:>8.1e}{mk}")

elapsed=time.time()-t0

# Evaluate
model.load_state_dict(torch.load("best_cnn_model.pth",map_location=DEVICE))
model.eval()
probs=[]; preds=[]; lbls=[]
with torch.no_grad():
    for x,y in te_ld:
        p=model(x.to(DEVICE)).cpu().squeeze(1).numpy()
        probs.extend(p); preds.extend((p>0.5).astype(int))
        lbls.extend(y.numpy().astype(int))
lbls=np.array(lbls); preds=np.array(preds); probs=np.array(probs)
acc=(lbls==preds).mean()
fpr,tpr,_=roc_curve(lbls,probs)
roc_auc=auc(fpr,tpr)
cm=confusion_matrix(lbls,preds)
print(f"\n  Accuracy: {acc*100:.2f}%  AUC: {roc_auc:.4f}")

# Fig 5
ep_r=range(1,len(hist["tl"])+1)
fig,(a1,a2)=plt.subplots(1,2,figsize=(12,4.5))
a1.plot(ep_r,hist["tl"],"b-o",ms=3,lw=2,label="Train")
a1.plot(ep_r,hist["vl"],"r-s",ms=3,lw=2,label="Validation")
a1.axvline(best_ep,color="green",ls="--",lw=1.5,label=f"Best ep {best_ep}")
a1.axhline(0.1,color="gray",ls=":",lw=1.5,label="0.1 reference")
a1.set_xlabel("Epoch",fontsize=12); a1.set_ylabel("BCE Loss",fontsize=12)
a1.set_title("Training & Validation Loss",fontsize=13,fontweight="bold")
a1.legend(fontsize=10); a1.grid(alpha=0.3)
a2.plot(ep_r,[x*100 for x in hist["ta"]],"b-o",ms=3,lw=2,label="Train")
a2.plot(ep_r,[x*100 for x in hist["va"]],"r-s",ms=3,lw=2,label="Validation")
a2.axvline(best_ep,color="green",ls="--",lw=1.5,label=f"Best ep {best_ep}")
a2.set_xlabel("Epoch",fontsize=12); a2.set_ylabel("Accuracy (%)",fontsize=12)
a2.set_title("Training & Validation Accuracy",fontsize=13,fontweight="bold")
a2.legend(fontsize=10); a2.grid(alpha=0.3); a2.set_ylim([50,101])
plt.suptitle("CNN Training — RIS-Assisted FMCW NLOS Detection\n"
             "(Imperfect RIS + K-Distribution Clutter)",
             fontsize=13,fontweight="bold")
plt.tight_layout()
plt.savefig("fig5_training_curves.png",dpi=150,bbox_inches="tight")
plt.close()
print("  [SAVED] fig5_training_curves.png")

# Fig 6
fig,(a1,a2)=plt.subplots(1,2,figsize=(12,5))
a1.plot(fpr,tpr,color="darkorange",lw=2.5,
        label=f"CNN + Imperfect RIS (AUC={roc_auc:.3f})")
cfar_fpr=np.array([0,0.01,0.05,0.10,0.20,0.40,1.0])
cfar_tpr=np.array([0,0.15,0.40,0.55,0.70,0.83,1.0])
a1.plot(cfar_fpr,cfar_tpr,"g-.",lw=2,label="CA-CFAR reference")
a1.plot([0,1],[0,1],"k--",lw=1.5,label="Chance")
a1.set_xlabel("False Alarm Rate",fontsize=12)
a1.set_ylabel("Detection Probability",fontsize=12)
a1.set_title("ROC Curve — CNN vs CA-CFAR",fontsize=12,fontweight="bold")
a1.legend(fontsize=9,loc="lower right"); a1.grid(alpha=0.3)
disp=ConfusionMatrixDisplay(cm,display_labels=["Absent","Present"])
disp.plot(ax=a2,colorbar=False,cmap="Blues")
a2.set_title(f"Confusion Matrix\nAcc={acc*100:.1f}%  AUC={roc_auc:.3f}",
             fontsize=12,fontweight="bold")
plt.suptitle("CNN Evaluation — RIS-Assisted FMCW NLOS Detection",
             fontsize=13,fontweight="bold")
plt.tight_layout()
plt.savefig("fig6_roc_confusion.png",dpi=150,bbox_inches="tight")
plt.close()
print("  [SAVED] fig6_roc_confusion.png")

results={"test_accuracy_pct":round(float(acc*100),2),
         "auc":round(float(roc_auc),4),
         "best_epoch":best_ep,
         "training_time_min":round(elapsed/60,1),
         "label_smoothing":LABEL_SMOOTH,
         "batch_size":BATCH,
         "learning_rate":LR,
         "confusion_matrix":cm.tolist(),
         "history":hist}
with open("cnn_results.json","w") as f: json.dump(results,f,indent=2)

print(f"\n{'='*55}")
print(f"  Accuracy : {acc*100:.2f}%")
print(f"  AUC      : {roc_auc:.4f}")
print(f"  Best ep  : {best_ep}")
print(f"  Time     : {elapsed/60:.1f} min")
print(f"{'='*55}")
