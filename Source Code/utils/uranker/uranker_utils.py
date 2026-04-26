import torch
import os
import yaml
from .uranker_model import URanker

def get_option(opt_path):
    with open(opt_path, 'r') as f:
        option = yaml.safe_load(f)
    
    option.setdefault('seed', 2022)

    return option

def build_model(opt):
    model_class = URanker

    
    all_args = list(opt.keys())
    model_args = {}
    for i in range(len(all_args) - 4):
        model_args[all_args[i + 4]] = opt.get(all_args[i + 4])
    model = model_class(**model_args)

    if opt['cuda'] and torch.cuda.is_available():
        model = model.cuda()
    if opt['parallel'] and torch.cuda.is_available():
        model = torch.nn.DataParallel(model)

    
        device = torch.device('cuda' if opt['cuda'] and torch.cuda.is_available() else 'cpu')
        ckpt_dict = torch.load(opt['resume_ckpt_path'], map_location=device)['net']
        model.load_state_dict(ckpt_dict)

    return model