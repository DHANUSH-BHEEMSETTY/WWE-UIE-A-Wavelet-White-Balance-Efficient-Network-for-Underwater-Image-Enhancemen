"""
train_v2.py  –  Enhanced training script for WWE-UIE-v2
========================================================
Differences from train.py (BASELINE):
  1. Imports myModelV2 from model_v2.py (deeper, wider, SE attention)
  2. Adds LabColorLoss (CIE-Lab chroma loss targeting underwater color shift)
  3. Updated loss weights:
       final_loss = L1 + 0.5*HVI + 0.2*SSIM + 0.1*VGG + 0.1*Edge + 0.2*Lab
     (SSIM weight raised 0.1→0.2; Lab loss added at 0.2)
  4. Model saved under model_name "WWE-UIE-v2" to keep outputs separate

ORIGINAL train.py IS NOT MODIFIED.
"""

import argparse
import os
import time
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from utils.dataset import get_loader
from model_v2 import myModelV2                          
from datetime import datetime
from utils.metrics import Evaluator
from utils.loss_funcs import (
    EdgeAwareLoss,
    SSIMLoss,
    L1_Charbonnier_loss,
    PerceptualLoss,
)
from utils.CIDNet import CIDNet






class LabColorLoss(nn.Module):
    """
    Computes L2 loss in CIE-Lab color space between predicted and target image.

    Why Lab?
      - The 'a*' and 'b*' channels directly encode chromaticity (color shift).
      - Underwater images suffer from blue-green color casts (positive b*, low a*).
      - Penalising Lab distance pushes the model to restore natural colors
        even when L1/SSIM loss is already low.

    Formula:
        lab_loss = MSE(Lab(pred)_ab, Lab(gt)_ab)
    """
    def __init__(self):
        super().__init__()
        
        
        self.register_buffer(
            "rgb2xyz",
            torch.tensor([
                [0.4124564, 0.3575761, 0.1804375],
                [0.2126729, 0.7151522, 0.0721750],
                [0.0193339, 0.1191920, 0.9503041],
            ], dtype=torch.float32),
        )
        
        self.register_buffer(
            "d65",
            torch.tensor([0.95047, 1.00000, 1.08883], dtype=torch.float32)
            .view(1, 3, 1, 1),
        )

    def _rgb_to_lab(self, img):
        """img: [B,3,H,W] in [0,1] → Lab [B,3,H,W]"""
        
        img = torch.where(img <= 0.04045,
                          img / 12.92,
                          ((img + 0.055) / 1.055) ** 2.4)
        
        B, C, H, W = img.shape
        img_flat = img.view(B, 3, -1)                         
        xyz = torch.einsum("ij,bjk->bik", self.rgb2xyz, img_flat)
        xyz = xyz.view(B, 3, H, W)
        
        xyz = xyz / self.d65
        
        delta = 6.0 / 29.0
        mask = xyz > delta ** 3
        f = torch.where(mask,
                        xyz.clamp(min=1e-8) ** (1.0 / 3.0),
                        xyz / (3 * delta ** 2) + 4.0 / 29.0)
        
        L = 116.0 * f[:, 1:2] - 16.0
        a = 500.0 * (f[:, 0:1] - f[:, 1:2])
        b = 200.0 * (f[:, 1:2] - f[:, 2:3])
        return torch.cat([L, a, b], dim=1)

    def forward(self, pred, target):
        pred_lab   = self._rgb_to_lab(pred.clamp(0.0, 1.0))
        target_lab = self._rgb_to_lab(target.clamp(0.0, 1.0))
        
        return F.mse_loss(pred_lab[:, 1:], target_lab[:, 1:])






def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True






class Trainer:
    def __init__(self, args):
        self.args = args
        self.evaluator = Evaluator()

        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_cuda = torch.cuda.is_available()
        print(f"[Device] Using: {self.device}")

        
        self.deep_model = myModelV2(
            in_channels=3, feature_channels=48, use_white_balance=True
        ).to(self.device)

        
        self.hvi_net = None
        if self.use_cuda:
            self.hvi_net = CIDNet().to(self.device)
            self.hvi_net.load_state_dict(
                torch.load(r"utils/CIDNet_weight_LOLv2_bestSSIM.pth",
                           map_location=self.device)
            )
            self.hvi_net.eval()

        
        now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.model_save_path = os.path.join(
            args.save_path, args.model_name, args.dataset, now_str
        )
        os.makedirs(self.model_save_path, exist_ok=True)

        
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError(f"=> no checkpoint at '{args.resume}'")
            ckpt = torch.load(args.resume, map_location=self.device)
            sd   = self.deep_model.state_dict()
            sd.update({k: v for k, v in ckpt.items() if k in sd})
            self.deep_model.load_state_dict(sd)

        
        self.optim = optim.AdamW(
            self.deep_model.parameters(),
            lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999),
        )
        if args.scheduler == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optim, args.epoch, eta_min=args.lr * 1e-4
            )
        else:
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optim, step_size=args.decay_epoch, gamma=args.decay_rate
            )

        
        dataset_map = {
            "EUVP-d": ("UnderWaterDataset/EUVP-Dark/train/",
                       "UnderWaterDataset/EUVP-Dark/val/",   256, False),
            "EUVP-s": ("UnderWaterDataset/EUVP-Scene/train/",
                       "UnderWaterDataset/EUVP-Scene/val/",  256, True),
            "UIEB":   ("UnderWaterDataset/UIEB/train/",
                       "UnderWaterDataset/UIEB/val/",        256, True),
            "UFO":    ("UnderWaterDataset/UFO-120/train/",
                       "UnderWaterDataset/UFO-120/val/",     256, True),
            "LSUI":   ("UnderWaterDataset/LSUI/train/",
                       "UnderWaterDataset/LSUI/val/",        256, True),
        }
        tr, val, sz, rz = dataset_map[args.dataset]
        args.train_root = tr
        args.val_root   = val
        args.datasize   = sz
        args.resize     = rz

        
        dev_str = "cuda" if self.use_cuda else "cpu"
        self.L1L   = L1_Charbonnier_loss()
        self.ssimL = SSIMLoss(device=dev_str, window_size=5)
        self.edgeL = EdgeAwareLoss(loss_type="l2", device=dev_str)
        self.labL  = LabColorLoss().to(self.device)
        
        self.vggL  = PerceptualLoss() if self.use_cuda else None

    
    
    

    def training(self):
        best_psnr  = 0.0
        best_round = {}
        if self.use_cuda:
            torch.cuda.empty_cache()
        
        pin_mem = self.use_cuda
        train_loader = get_loader(
            self.args.train_root, self.args.train_batch_size,
            self.args.datasize, train=True, resize=self.args.resize,
            num_workers=self.args.num_workers, shuffle=True, pin_memory=pin_mem,
        )
        self.deep_model.train()

        for epoch in range(1, self.args.epoch + 1):
            loop      = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
            loss_mean = 0.0

            for _, (x, label, _) in loop:
                x     = x.to(self.device)
                label = label.to(self.device)
                pred  = self.deep_model(x)
                self.optim.zero_grad()

                l1_loss   = self.L1L(pred, label)
                ssim_loss = self.ssimL(pred, label)
                edge_loss = self.edgeL(pred, label)
                lab_loss  = self.labL(pred, label)

                
                hvi_loss = torch.tensor(0.0, device=self.device)
                vgg_loss = torch.tensor(0.0, device=self.device)
                if self.hvi_net is not None:
                    with torch.no_grad():
                        label_hvi = self.hvi_net.trans.HVIT(label)
                    pred_hvi = self.hvi_net.trans.HVIT(pred.clamp(0.0, 1.0))
                    hvi_loss = self.L1L(pred_hvi, label_hvi)
                if self.vggL is not None:
                    vgg_loss = self.vggL(pred, label)

                
                final_loss = (
                    l1_loss
                    + 0.5 * hvi_loss
                    + 0.2 * ssim_loss
                    + 0.1 * vgg_loss
                    + 0.1 * edge_loss
                    + 0.2 * lab_loss
                )

                loss_mean += final_loss.item()
                final_loss.backward()
                self.optim.step()
                loop.set_description(f"[{epoch}/{self.args.epoch}]")
                loop.set_postfix(loss=final_loss.item())

            avg_loss = loss_mean / len(train_loader)
            lr_now   = self.optim.param_groups[0]["lr"]
            print(f"[{epoch}/{self.args.epoch}] avg_loss={avg_loss:.4f}, lr={lr_now:.2e}")

            if epoch % self.args.epoch_val == 0:
                self.deep_model.eval()
                ssim_, psnr_ = self.validation()
                if psnr_ > best_psnr:
                    torch.save(
                        self.deep_model.state_dict(),
                        os.path.join(self.model_save_path, "best_model.pth"),
                    )
                    best_psnr  = psnr_
                    best_round = {"best_epoch": epoch, "best_PSNR": best_psnr, "best_SSIM": ssim_}
                    with open(os.path.join(self.model_save_path, "records.txt"), "a") as f:
                        f.write(f"## BEST ## epoch={epoch} PSNR={psnr_:.4f} SSIM={ssim_:.4f}\n")
                with open(os.path.join(self.model_save_path, "records.txt"), "a") as f:
                    f.write(f"[epoch {epoch}] PSNR={psnr_:.4f} SSIM={ssim_:.4f}\n")
                self.deep_model.train()

            self.scheduler.step()

        print("Best round:", best_round)

    
    
    

    def validation(self):
        self.evaluator.reset()
        pin_mem = self.use_cuda
        val_loader = get_loader(
            self.args.val_root, self.args.eval_batch_size,
            self.args.datasize, train=False, resize=self.args.resize,
            num_workers=1, shuffle=False, pin_memory=pin_mem,
        )
        if self.use_cuda:
            torch.cuda.empty_cache()
        with torch.no_grad():
            loop = tqdm(enumerate(val_loader), total=len(val_loader), leave=False)
            for _, (x, label, _) in loop:
                x     = x.to(self.device)
                label = label.numpy().astype(np.float32).transpose(0, 2, 3, 1)
                pred  = self.deep_model(x)
                pred  = torch.clamp(pred, 0.0, 1.0)
                pred  = pred.data.cpu().numpy().astype(np.float32).transpose(0, 2, 3, 1)
                self.evaluator.evaluation(pred, label)
                loop.set_description("[Validation]")
        ssim_, psnr_ = self.evaluator.getMean()
        print(f"[Validation] SSIM: {ssim_:.4f}, PSNR: {psnr_:.4f}")
        return ssim_, psnr_






def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch",           type=int,   default=100)
    parser.add_argument("--epoch_val",       type=int,   default=1)
    parser.add_argument("--lr",              type=float, default=2e-3)
    parser.add_argument("--train_batch_size",type=int,   default=16)   
    parser.add_argument("--eval_batch_size", type=int,   default=16)
    parser.add_argument("--decay_rate",      type=float, default=0.1)
    parser.add_argument("--decay_epoch",     type=int,   default=50)
    parser.add_argument("--weight_decay",    type=float, default=1e-5)
    parser.add_argument("--scheduler",       type=str,   default="cosine")
    parser.add_argument("--num_workers",     type=int,   default=4)
    parser.add_argument("--dataset",         type=str,   default="UIEB",
                        choices=["UIEB", "LSUI", "UFO", "EUVP-s", "EUVP-d"])
    parser.add_argument("--model_name",      type=str,   default="WWE-UIE-v2")
    parser.add_argument("--save_path",       type=str,   default="./output/")
    parser.add_argument("--resume",          type=str,   default=None)
    args = parser.parse_args()

    trainer = Trainer(args)
    trainer.training()


if __name__ == "__main__":
    start = time.time()
    seed_everything(7)
    main()
    print(f"Total training time: {time.time() - start:.1f}s")
