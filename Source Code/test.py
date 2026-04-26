import argparse
import os
import time
import numpy as np

import torch
from tqdm import tqdm
from thop import profile, clever_format
from PIL import Image

from utils.dataset import get_loader
from model import myModel
from utils.metrics import Evaluator
from utils.utils import store_restored


class Tester(object):
    def __init__(self, args):
        self.args = args

        self.evaluator = Evaluator()

        self.deep_model = myModel(
            in_channels=3, feature_channels=32, use_white_balance=True
        )

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if os.path.isfile(args.ckpt):
            checkpoint = torch.load(args.ckpt, map_location=self.device)
            model_dict = {}
            state_dict = self.deep_model.state_dict()
            for k, v in checkpoint.items():
                if k in state_dict:
                    model_dict[k] = v
            state_dict.update(model_dict)
            self.deep_model.load_state_dict(state_dict)
        else:
            raise RuntimeError("=> no checkpoint found at '{}'".format(args.ckpt))

        self.deep_model = self.deep_model.to(self.device)
        self.deep_model.eval()

        if args.dataset == "EUVP-d":  
            args.test_root = "UnderWaterDataset/EUVP-Dark/test/"
            args.datasize = 256
            args.resize = False
        elif args.dataset == "EUVP-s":  
            args.test_root = "UnderWaterDataset/EUVP-Scene/test/"
            args.datasize = 256
            args.resize = False
        elif args.dataset == "UIEB":  
            args.test_root = "UnderWaterDataset/UIEB/test/"
            args.datasize = 256
            args.resize = True
        elif args.dataset == "UFO":  
            args.test_root = "UnderWaterDataset/UFO-120/test/"
            args.datasize = 256
            args.resize = False
        elif args.dataset == "LSUI":  
            args.test_root = "UnderWaterDataset/LSUI/test/"
            args.datasize = 256
            args.resize = True

        self.dataloader = get_loader(
            self.args.test_root,
            self.args.test_batch_size,
            self.args.datasize,
            train=False,
            resize=args.resize,
            num_workers=1,
            shuffle=False,
            pin_memory=True,
        )

    def testing(self):
        self.evaluator.reset()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        with torch.no_grad():
            loop = tqdm(
                enumerate(self.dataloader), total=len(self.dataloader), leave=False
            )
            for _, (x, label, fn) in loop:
                x = x.to(self.device)
                label = (
                    label.numpy().astype(np.float32).transpose(0, 2, 3, 1)
                )  

                pred = self.deep_model(x)
                pred = torch.clamp(pred, 0.0, 1.0)
                pred = (
                    pred.data.cpu().numpy().astype(np.float32).transpose(0, 2, 3, 1)
                )  

                print(f"Prediction type: {type(pred)}, shape: {pred.shape}")
                print(f"Label type: {type(label)}, shape: {label.shape}")
                
                try:
                    self.evaluator.evaluation(pred, label)
                    print("Evaluated successfully")
                except Exception as e:
                    print(f"Evaluation crashed: {e}")
                    raise e
                    
                ckpt_dir = os.path.dirname(self.args.ckpt)
                if not os.path.exists(
                    os.path.join(ckpt_dir, "pred")
                ):
                    os.makedirs(os.path.join(ckpt_dir, "pred"))
                store_restored(
                    pred, label, fn, ckpt_dir
                )
                loop.set_description("[Testing]")
        
        ssim_, psnr_ = self.evaluator.getMean()

        print(
            
            "[Testing] SSIM: %.4f, PSNR: %.4f"
            % (ssim_, psnr_)
            
        )
        with open(os.path.join(os.path.dirname(self.args.ckpt), "result.txt"), "w") as f:
            f.write("[Testing] SSIM: %.4f, PSNR: %.4f" % (ssim_, psnr_))
        
        dummy = torch.randn(1, 3, self.args.datasize, self.args.datasize).to(self.device)
        flops, params = profile(self.deep_model, inputs=(dummy,))
        flops, params = clever_format([flops, params], "%.3f")
        model_info = {
            "params": params,
            "flops": flops,
            "ssim": "%.4f" % ssim_,
            "psnr": "%.4f" % psnr_,
        }
        print(f"Params: {params}, FLOPs: {flops}")

        return model_info


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ckpt",
        type=str,
    )
    parser.add_argument("--dataset", type=str, default="UIEB")
    parser.add_argument("--test_batch_size", type=int, default=4)

    args = parser.parse_args()

    tester = Tester(args)

    start = time.time()

    model_info = tester.testing()

    end = time.time()
    print("Testing time:", end - start, "sec")
    model_info["time"] = end - start


if __name__ == "__main__":
    main()
