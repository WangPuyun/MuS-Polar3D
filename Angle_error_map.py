import argparse
import math
import os
from torch.nn.functional import normalize
from torchvision.transforms.functional import affine
import DCC
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torchvision
from torch.backends import cudnn
import UPIE
import Unet
import config as config
from math import pi
from Datasets import RandomMove, unfold_image, concat_image
from torchvision.utils import save_image
# SfP Baseline
from AttentionU2Net.U2Net import AttentionU2Net
from DeepSfP_Net import DeepSfP
from TransUNet import TransUnet # SfPW
from TransSfP.TransSfPNet.TransSfPNet import TransSfPNet

from utils_window import PATCH, OVERLAP, STRIDE, hann2d
import cv2
import matplotlib.pyplot as plt
# Distributed inference can be used, but it should be noted that the number of GPUs needs to be equal to the number of test_batch_size
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Network Testing')
    # Set the model name (checkpoint file) to load
    parser.add_argument("--model_name", type=str, default=None,
                    help="Path to the pre-trained model file to load, e.g., 'xxx.pth'")
    # Test batch size
    parser.add_argument("--test_batch_size", type=int, default=1, help="Batch size for testing, it should be equal to the number of GPUs")
    # Test data directory or other parameters (add as needed)
    parser.add_argument("--ckpt_path", type=str, default='./pt/300.pth', help="Path to the model weights file, e.g., './pt/100.pt'")
    parser.add_argument("--results_dir", type=str, default='./results_sfp', help="Directory to save prediction maps, e.g., './results_sfp_100'")
    parser.add_argument("--error_maps_dir", type=str, default='./error_maps', help="Directory to save error maps, e.g., './error_maps_100'")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    # Get the number of available GPUs
    args.nprocs = torch.cuda.device_count()
    # For multi-GPU testing, you can use the same multi-process approach as training
    mp.spawn(main_worker, nprocs=args.nprocs, args=(args.nprocs, args))


def main_worker(local_rank, nprocs, args):
    # Initialize the distributed environment
    args.local_rank = local_rank
    config.init_distributed(local_rank=args.local_rank, nprocs=args.nprocs)

    # Build model and optimizer
    model = AttentionU2Net()
    model = model.cuda(args.local_rank)

    # Load the specified checkpoint
    checkpoint = torch.load(args.ckpt_path)
    model.load_state_dict(checkpoint['model'])

    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.error_maps_dir, exist_ok=True)

    # Synchronized BatchNorm (SyncBN): ensures consistent BN statistics across GPUs to prevent inconsistent results during multi-GPU testing
    model = config.wrap_model_distributed(model, local_rank=local_rank)
    model.eval()  # Set the model to evaluation mode

    # Create loss function (can be kept if loss calculation is needed during testing)
    criterion = nn.CosineSimilarity().cuda(args.local_rank)

    # Create the test dataset data loader
    test_loader, _ = config.test_dataloaders(args)

    # For potential speedup
    # cudnn.benchmark = True

    # Test on each GPU in a loop (or execute only on the main process)
    device = torch.device(f'cuda:{local_rank}')
    window = hann2d(PATCH, device).unsqueeze(0).unsqueeze(0)

    total_loss, total_samples = 0., 0
    
    with torch.no_grad():
        for i, sample in enumerate(test_loader):
            inputs   = sample['input'].cuda(device)
            image   = sample['image'].cuda(device)
            gt       = sample['ground_truth'].float().cuda(device) / 255.
            mask     = sample['mask'].unsqueeze(1).cuda(device)
            gt *= mask

            H, W     = inputs.shape[2:]
            # ----------- Prepare empty containers -----------
            out_sum = torch.zeros(1, 3, H, W, device=device)
            w_sum   = torch.zeros(1, 1, H, W, device=device)

            # ----------- Sliding window -----------
            for y in range(0, H - PATCH + 1, STRIDE):
                for x in range(0, W - PATCH + 1, STRIDE):
                    patch = inputs[..., y:y+PATCH, x:x+PATCH]
                    patch2 = image[..., y:y+PATCH, x:x+PATCH]
                    pred = model(patch)
                    pred = pred * window
                    out_sum[..., y:y+PATCH, x:x+PATCH] += pred
                    w_sum[...,  y:y+PATCH, x:x+PATCH] += window

            # ----------- Normalize to obtain the complete result -----------
            full_pred = out_sum / w_sum.clamp_min(1e-6)
            full_pred = torch.nn.functional.normalize(full_pred, dim=1)
            full_pred *= mask
            filename = sample['filename']

            # ----------- Angular MAE calculation -----------
            # Network outputs normal maps in [-1, 1] range; ground truth must be normalized to this interval for evaluation.
            # pred_n = (full_pred *2.0 -1.0) * mask
            gt_n = (gt *2.0 -1.0) * mask
            
            cos = criterion(full_pred, gt_n)
            cos = torch.clamp(cos, -1.0, 1.0)
            ang = (torch.acos(cos) * 180.0 / pi).unsqueeze(1)
            ang = ang * mask

            M = torch.sum(mask)
            valid = ang[mask.bool()]
            mae = valid.mean()                      # Mean
            median_ang = valid.median()             # Median
            rmse = torch.sqrt((valid ** 2).mean())  # Root Mean Square Error

            total_loss   += mae.item()
            total_samples += 1
            
            img = ((full_pred+1)*0.5)*mask
            save_image(img, f'{args.results_dir}/{filename[0]}_{mae:.4f}.bmp')
            
            # ----------- Additional error heatmaps -----------

            stats_text = (
                f"Mean (MAE): {mae.item():.2f}°\n"
                f"Median   : {median_ang.item():.2f}°\n"
                f"RMSE     : {rmse.item():.2f}°"
            )

            theta_max = 50.0
            ang_clamped = torch.clamp(ang.squeeze(1), 0.0, theta_max)
            ang_clamped[mask.squeeze(1) == 0] = float('nan')
            fig, ax = plt.subplots(figsize=(6, 6))
            im = ax.imshow(ang_clamped.squeeze().cpu().numpy(),       # Use angular data, not pre-colored BGR images
                        cmap='jet',
                        vmin=0, vmax=theta_max)
            ax.axis('off')
            ax.text(
                0.02, 0.98, stats_text,
                transform=ax.transAxes,
                ha='left', va='top',
                fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.5, pad=0.4),
                color='white',
                zorder=10,
            )
            ax.set_title('Angular Error')

            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Error (°)')

            fig.tight_layout()
            fig.savefig(f'{args.error_maps_dir}/{filename[0]}.png', dpi=300, bbox_inches='tight')
            plt.close(fig)


            
            
            

if __name__ == "__main__":
    main()
