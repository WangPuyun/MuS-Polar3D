# utils_window.py
import torch
from math import pi

PATCH   = 256          # Patch size
OVERLAP = 64           # Recommended 32~64, fine-tune as needed
STRIDE  = PATCH - OVERLAP

def hann2d(size: int, device):
    """生成 2-D Hann window，范围 0~1"""
    w = torch.hann_window(size, periodic=False, device=device)   # (size,)
    window2d = torch.outer(w, w)                                 # (size,size)
    return window2d / window2d.max()                             # Normalized
