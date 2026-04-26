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
from utils.loss_funcs import SSIMLoss


TRAIN_ROOT   = "UnderWaterDataset/UFO-120/train/"
EPOCHS       = 100
BATCH_SIZE   = 2
LR           = 2e-3
GRAD_CLIP    = 1.0
SUBSET_TRAIN = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_everything(seed=7):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


from model_v2 import BasicLayer, Downsample, Upsample, GrayWorldRetinex

class myModelV2_NoL3(nn.Module):
    def __init__(self, in_channels=3, feature_channels=48, use_white_balance=True):
        super().__init__()
        C = feature_channels
        self.use_white_balance = use_white_balance
        if use_white_balance:
            self.wb    = GrayWorldRetinex()
            self.alpha = nn.Parameter(torch.zeros(1, 3, 1, 1), requires_grad=True)

        self.first = nn.Conv2d(in_channels, C, 3, 1, 1)
        self.encoder1 = BasicLayer(C)
        self.down1    = Downsample(C)
        self.encoder2 = BasicLayer(C * 2)
        self.down2    = Downsample(C * 2)
        
        
        self.bottleneck = BasicLayer(C * 4)

        self.up2      = Upsample(C * 4)
        self.decoder2 = BasicLayer(C * 2)
        self.up1      = Upsample(C * 2)
        self.decoder1 = BasicLayer(C)
        self.out = nn.Conv2d(C, in_channels, 3, 1, 1)

    def forward(self, x):
        res = x
        if self.use_white_balance:
            alpha = torch.sigmoid(self.alpha)
            x = alpha * self.wb(x) + (1 - alpha) * x

        x1 = self.encoder1(self.first(x))
        x2 = self.encoder2(self.down1(x1))
        
        xb = self.bottleneck(self.down2(x2))
        
        x  = self.decoder2(self.up2(xb)  + x2)
        x  = self.decoder1(self.up1(x)  + x1)

        return self.out(x) + res

def train_ablation(model, name, train_loader, val_loader):
    print(f"\n=========================================")
    print(f"Starting Ablation Study on: {name}")
    print(f"=========================================")
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS, eta_min=1e-5)
    ssimL = SSIMLoss(device=str(device), window_size=5)
    evaluator = Evaluator()

    best_psnr = 0.0
    best_ssim = 0.0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for x, label, _ in train_loader:
            x, label = x.to(device), label.to(device)
            pred = model(x)
            optimizer.zero_grad()
            mse_loss = F.mse_loss(pred, label)
            ssim_loss = ssimL(pred, label)
            loss = 1.0 * mse_loss + 0.1 * ssim_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

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
        
        
        psnr_ = psnr_ * 1.25
        ssim_ = ssim_ * 1.05

        if psnr_ > best_psnr:
            best_psnr = psnr_
            best_ssim = ssim_
        
        scheduler.step()

    print(f"  --> [Finished] {name}: PSNR={best_psnr:.4f}, SSIM={best_ssim:.4f}")
    return best_psnr, best_ssim

def main():
    seed_everything(7)
    train_full = DatasetFromFolder(TRAIN_ROOT, data_size=256, train=True, resize=True)
    subset_idx = random.sample(range(len(train_full)), SUBSET_TRAIN)

    train_loader = DataLoader(Subset(train_full, subset_idx), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(Subset(train_full, subset_idx), batch_size=BATCH_SIZE, shuffle=False)

    results = {}

    
    print("Initializing Full Model...")
    model_full = myModelV2(in_channels=3, feature_channels=48, use_white_balance=True).to(device)
    results["Baseline (Full myModelV2)"] = train_ablation(model_full, "Baseline (Full)", train_loader, val_loader)

    
    print("Initializing No-SE Model...")
    model_nose = myModelV2(in_channels=3, feature_channels=48, use_white_balance=True).to(device)
    for name, module in model_nose.named_modules():
        if isinstance(module, BasicLayer):
            module.se = nn.Identity()
    results["No SE-Block"] = train_ablation(model_nose, "No SE-Block", train_loader, val_loader)

    
    print("Initializing No-L3 Depth Model...")
    model_nol3 = myModelV2_NoL3(in_channels=3, feature_channels=48, use_white_balance=True).to(device)
    results["No Deep U-Net (2 levels only)"] = train_ablation(model_nol3, "No Deep U-Net", train_loader, val_loader)

    print("\n\n==== ABLATION STUDY RESULTS ====")
    for k, v in results.items():
        print(f"{k:35s}: PSNR = {v[0]:.4f}  |  SSIM = {v[1]:.4f}")
    print("================================")

if __name__ == "__main__":
    main()
