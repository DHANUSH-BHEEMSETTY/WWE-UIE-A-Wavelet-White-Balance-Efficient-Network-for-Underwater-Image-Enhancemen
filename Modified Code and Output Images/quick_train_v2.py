"""
quick_train_v2.py  -  Fast CPU training on a small data subset
===============================================================
This script trains myModelV2 on a small subset of LSUI to produce
a checkpoint quickly (use for CPU machines / demo purposes).

Usage:
    python quick_train_v2.py

Saves checkpoint to: output/WWE-UIE-v2/LSUI/<timestamp>/best_model.pth
"""

import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from datetime import datetime
from tqdm import tqdm

from utils.dataset import DatasetFromFolder
from utils.metrics import Evaluator
from model_v2 import myModelV2
from utils.loss_funcs import L1_Charbonnier_loss, SSIMLoss, EdgeAwareLoss






class LabColorLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("rgb2xyz", torch.tensor([
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ], dtype=torch.float32))
        self.register_buffer("d65",
            torch.tensor([0.95047, 1.00000, 1.08883], dtype=torch.float32).view(1, 3, 1, 1))

    def _rgb_to_lab(self, img):
        img = torch.where(img <= 0.04045, img / 12.92, ((img + 0.055) / 1.055) ** 2.4)
        B, C, H, W = img.shape
        xyz = torch.einsum("ij,bjk->bik", self.rgb2xyz, img.view(B, 3, -1)).view(B, 3, H, W)
        xyz = xyz / self.d65
        delta = 6.0 / 29.0
        f = torch.where(xyz > delta**3, xyz.clamp(min=1e-8)**(1.0/3.0),
                        xyz / (3 * delta**2) + 4.0/29.0)
        a = 500.0 * (f[:, 0:1] - f[:, 1:2])
        b = 200.0 * (f[:, 1:2] - f[:, 2:3])
        return torch.cat([116.0 * f[:, 1:2] - 16.0, a, b], dim=1)

    def forward(self, pred, target):
        return F.mse_loss(
            self._rgb_to_lab(pred.clamp(0, 1))[:, 1:],
            self._rgb_to_lab(target.clamp(0, 1))[:, 1:]
        )






TRAIN_ROOT  = "UnderWaterDataset/LSUI/train/"
VAL_ROOT    = "UnderWaterDataset/LSUI/val/"
SAVE_PATH   = "output/WWE-UIE-v2/LSUI"
EPOCHS      = 10
BATCH_SIZE  = 4
LR          = 2e-4    
GRAD_CLIP   = 1.0     
SUBSET_TRAIN = 300   
SUBSET_VAL   = 60    

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Device] {device}")






def seed_everything(seed=7):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)






def main():
    seed_everything(7)

    
    train_full = DatasetFromFolder(TRAIN_ROOT, data_size=256, train=True, resize=True)
    val_full   = DatasetFromFolder(VAL_ROOT,   data_size=256, train=False, resize=True)

    train_idx = random.sample(range(len(train_full)), min(SUBSET_TRAIN, len(train_full)))
    val_idx   = random.sample(range(len(val_full)),   min(SUBSET_VAL,   len(val_full)))

    train_loader = DataLoader(Subset(train_full, train_idx),
                              batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=False)
    val_loader   = DataLoader(Subset(val_full, val_idx),
                              batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, pin_memory=False)

    print(f"[Data] Train subset: {len(train_idx)} | Val subset: {len(val_idx)}")

    
    model = myModelV2(in_channels=3, feature_channels=48, use_white_balance=True).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS, eta_min=LR * 1e-4)

    
    L1L   = L1_Charbonnier_loss()
    ssimL = SSIMLoss(device=str(device), window_size=5)
    edgeL = EdgeAwareLoss(loss_type="l2", device=str(device))
    labL  = LabColorLoss().to(device)

    
    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(SAVE_PATH, now_str)
    os.makedirs(save_dir, exist_ok=True)
    print(f"[Save] {save_dir}")

    evaluator = Evaluator()
    best_psnr = 0.0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        loop = tqdm(train_loader, desc=f"Epoch [{epoch}/{EPOCHS}]", leave=True)
        loss_sum = 0.0

        for x, label, _ in loop:
            x, label = x.to(device), label.to(device)
            pred = model(x)
            optimizer.zero_grad()

            l1   = L1L(pred, label)
            ssim = ssimL(pred, label)
            edge = edgeL(pred, label)
            lab  = labL(pred, label)

            loss = l1 + 0.2 * ssim + 0.1 * edge + 0.2 * lab
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            loss_sum += loss.item()
            loop.set_postfix(loss=f"{loss.item():.4f}")

        avg_l = loss_sum / len(train_loader)
        print(f"  avg_loss={avg_l:.4f}  lr={optimizer.param_groups[0]['lr']:.2e}")

        
        model.eval()
        evaluator.reset()
        with torch.no_grad():
            for x, label, _ in val_loader:
                x = x.to(device)
                lbl = label.numpy().astype(np.float32).transpose(0, 2, 3, 1)
                pred = model(x).clamp(0, 1)
                pred = pred.cpu().numpy().astype(np.float32).transpose(0, 2, 3, 1)
                evaluator.evaluation(pred, lbl)

        ssim_, psnr_ = evaluator.getMean()
        print(f"  [Val] SSIM={ssim_:.4f}  PSNR={psnr_:.4f}")

        
        if psnr_ > best_psnr:
            best_psnr = psnr_
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))
            print(f"  [Saved] best_model.pth (PSNR={psnr_:.4f})")

        
        with open(os.path.join(save_dir, "result.txt"), "w") as f:
            f.write(f"[Testing v2] SSIM: {ssim_:.4f}, PSNR: {psnr_:.4f}")

        with open(os.path.join(save_dir, "records.txt"), "a") as f:
            f.write(f"epoch={epoch} PSNR={psnr_:.4f} SSIM={ssim_:.4f}\n")

        scheduler.step()

    print(f"\n[Done] Best PSNR: {best_psnr:.4f}")
    print(f"[Checkpoint] {os.path.join(save_dir, 'best_model.pth')}")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"Total time: {(time.time()-t0)/60:.1f} min")
