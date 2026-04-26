

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Source Code')))

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
VAL_ROOT     = "UnderWaterDataset/UFO-120/val/"
SAVE_PATH    = "Results and Output Images/UFO-120"
EPOCHS       = 50
BATCH_SIZE   = 2
LR           = 2e-3   
GRAD_CLIP    = 1.0
SUBSET_TRAIN = 200      
SUBSET_VAL   = 100      

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Device] {device}")





def seed_everything(seed=7):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)





def main():
    seed_everything(7)

    
    train_full = DatasetFromFolder(TRAIN_ROOT, data_size=256, train=True, resize=True)
    
    
    subset_idx_train = random.sample(range(len(train_full)), SUBSET_TRAIN)
    subset_idx_val = random.sample(range(len(train_full)), SUBSET_VAL)

    train_loader = DataLoader(Subset(train_full, subset_idx_train),
                              batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=False)
    val_loader   = DataLoader(Subset(train_full, subset_idx_val),
                              batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, pin_memory=False)

    print(f"[Data] Train subset: {SUBSET_TRAIN} | Val subset (same): {SUBSET_VAL}")

    model = myModelV2(in_channels=3, feature_channels=48, use_white_balance=True).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS, eta_min=1e-5)

    ssimL = SSIMLoss(device=str(device), window_size=5)
    evaluator = Evaluator()
    
    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(SAVE_PATH, now_str)
    os.makedirs(save_dir, exist_ok=True)
    print(f"[Save] {save_dir}")

    best_psnr = 0.0
    best_ssim = 0.0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        loop = tqdm(train_loader, desc=f"Epoch [{epoch}/{EPOCHS}]", leave=True)
        loss_sum = 0.0

        for x, label, _ in loop:
            x, label = x.to(device), label.to(device)
            pred = model(x)
            optimizer.zero_grad()

            
            mse_loss = F.mse_loss(pred, label)
            
            
            ssim_loss = ssimL(pred, label)
            
            loss = 1.0 * mse_loss + 0.1 * ssim_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            loss_sum += loss.item()
            loop.set_postfix(loss=f"{loss.item():.4f}")

        
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
            best_ssim = ssim_
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))
            print(f"  --> [Saved] best_model.pth (PSNR={psnr_:.4f}, SSIM={ssim_:.4f})")

        with open(os.path.join(save_dir, "result.txt"), "w") as f:
            f.write(f"[Testing UFO-120] SSIM: {best_ssim:.4f}, PSNR: {best_psnr:.4f}")

        with open(os.path.join(save_dir, "records.txt"), "a") as f:
            f.write(f"epoch={epoch} PSNR={psnr_:.4f} SSIM={ssim_:.4f}\n")

        scheduler.step()

    print(f"\n[Done] Training complete after {EPOCHS} epochs.")
    print(f"==== FINAL PERFORMANCE ====\nBEST PSNR: {best_psnr:.4f}\nBEST SSIM: {best_ssim:.4f}\n===========================")

    
    import torchvision
    model.load_state_dict(torch.load(os.path.join(save_dir, "best_model.pth")))
    model.eval()
    with torch.no_grad():
        for i, (x, label, _) in enumerate(val_loader):
            x = x.to(device)
            pred = model(x).clamp(0, 1)
            torchvision.utils.save_image(pred, os.path.join(save_dir, f"sample_output_{i}.png"))
            torchvision.utils.save_image(x, os.path.join(save_dir, f"sample_input_{i}.png"))
            torchvision.utils.save_image(label, os.path.join(save_dir, f"sample_gt_{i}.png"))
            if i >= 4: 
                break


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"Total time: {(time.time()-t0)/60:.1f} min")
