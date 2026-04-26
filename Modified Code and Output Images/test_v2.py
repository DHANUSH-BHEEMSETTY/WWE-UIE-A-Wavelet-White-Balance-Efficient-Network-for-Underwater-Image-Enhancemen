"""
test_v2.py  –  Evaluation script for the enhanced WWE-UIE-v2 model
===================================================================
Usage (paired datasets):
    python test_v2.py --ckpt output/WWE-UIE-v2/UIEB/<ts>/best_model.pth --dataset UIEB
    python test_v2.py --ckpt output/WWE-UIE-v2/LSUI/<ts>/best_model.pth --dataset LSUI
    python test_v2.py --ckpt output/WWE-UIE-v2/EUVP-d/<ts>/best_model.pth --dataset EUVP-d
    python test_v2.py --ckpt output/WWE-UIE-v2/EUVP-s/<ts>/best_model.pth --dataset EUVP-s
    python test_v2.py --ckpt output/WWE-UIE-v2/UFO/<ts>/best_model.pth --dataset UFO

Saves results to:
    <ckpt_dir>/result.txt        ← picked up by compare_results.py
    <ckpt_dir>/pred/             ← side-by-side prediction images

IMPORTANT: Uses myModelV2 (model_v2.py). For the baseline model use test.py.
"""

import argparse
import os
import time
import numpy as np
from datetime import datetime

import torch
from tqdm import tqdm
from thop import profile, clever_format
from PIL import Image

from utils.dataset import get_loader
from model_v2 import myModelV2                
from utils.metrics import Evaluator
from utils.utils import store_restored


class Tester:
    def __init__(self, args):
        self.args = args
        self.evaluator = Evaluator()

        
        self.deep_model = myModelV2(
            in_channels=3, feature_channels=48, use_white_balance=True
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if os.path.isfile(args.ckpt):
            checkpoint = torch.load(args.ckpt, map_location=self.device)
            sd = self.deep_model.state_dict()
            sd.update({k: v for k, v in checkpoint.items() if k in sd})
            self.deep_model.load_state_dict(sd)
        else:
            raise RuntimeError(f"=> no checkpoint at '{args.ckpt}'")

        self.deep_model = self.deep_model.to(self.device).eval()

        
        if args.dataset == "EUVP-d":
            args.test_root = "UnderWaterDataset/EUVP-Dark/test/"
            args.datasize  = 256
            args.resize    = False
        elif args.dataset == "EUVP-s":
            args.test_root = "UnderWaterDataset/EUVP-Scene/test/"
            args.datasize  = 256
            args.resize    = False
        elif args.dataset == "UIEB":
            args.test_root = "UnderWaterDataset/UIEB/test/"
            args.datasize  = 256
            args.resize    = True
        elif args.dataset == "UFO":
            args.test_root = "UnderWaterDataset/UFO-120/test/"
            args.datasize  = 256
            args.resize    = False
        elif args.dataset == "LSUI":
            args.test_root = "UnderWaterDataset/LSUI/test/"
            args.datasize  = 256
            args.resize    = True

        self.dataloader = get_loader(
            self.args.test_root, self.args.test_batch_size, self.args.datasize,
            train=False, resize=args.resize, num_workers=1, shuffle=False,
            pin_memory=self.device.type == "cuda",
        )

    def testing(self):
        self.evaluator.reset()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        ckpt_dir = os.path.dirname(self.args.ckpt)
        now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_root = os.path.join("output/WWE-UIE-v2", self.args.dataset, "eval_" + now_str)
        pred_dir = os.path.join(out_root, "pred")
        os.makedirs(pred_dir, exist_ok=True)

        with torch.no_grad():
            loop = tqdm(enumerate(self.dataloader), total=len(self.dataloader), leave=False)
            for _, (x, label, fn) in loop:
                x     = x.to(self.device)
                label = label.numpy().astype(np.float32).transpose(0, 2, 3, 1)

                pred = self.deep_model(x)
                pred = torch.clamp(pred, 0.0, 1.0)
                pred = pred.data.cpu().numpy().astype(np.float32).transpose(0, 2, 3, 1)

                self.evaluator.evaluation(pred, label)
                store_restored(pred, label, fn, out_root)
                loop.set_description("[Testing v2]")

        ssim_, psnr_ = self.evaluator.getMean()
        result_str = f"[Testing v2] SSIM: {ssim_:.4f}, PSNR: {psnr_:.4f}"
        print(result_str)

        
        with open(os.path.join(out_root, "result.txt"), "w") as f:
            f.write(result_str)

        dummy       = torch.randn(1, 3, self.args.datasize, self.args.datasize).to(self.device)
        flops, params = profile(self.deep_model, inputs=(dummy,))
        flops, params = clever_format([flops, params], "%.3f")
        print(f"Params: {params}, FLOPs: {flops}")

        return {"ssim": ssim_, "psnr": psnr_, "params": params, "flops": flops}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",            type=str, required=True)
    parser.add_argument("--dataset",         type=str, default="UIEB",
                        choices=["UIEB", "LSUI", "UFO", "EUVP-d", "EUVP-s"])
    parser.add_argument("--test_batch_size", type=int, default=4)
    args = parser.parse_args()

    tester = Tester(args)
    start  = time.time()
    tester.testing()
    print(f"Testing time: {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
