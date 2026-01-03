import torch.distributed as dist
import torch
import UPIE
import Unet
import ResNet
import DCC
from torchvision.transforms.functional import affine
from torchvision.utils import save_image
from torch.nn.functional import normalize
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from Datasets import MyDataset, RandomCrop, FixedCrop, RandomMove, unfold_image, concat_image, unfold_enhanced_image, RandomMovePad, concat_enhanced_image
from torch.utils.data import DataLoader
# SfP Baseline
from AttentionU2Net.U2Net import AttentionU2Net
from DeepSfP_Net import DeepSfP
from TransUNet import TransUnet # SfPW
from TransSfP.TransSfPNet.TransSfPNet import TransSfPNet

from math import pi
import math
from utils_window import PATCH, OVERLAP, STRIDE, hann2d


def init_distributed(local_rank, nprocs, url='tcp://localhost:25464'):
    """
    Initialize the distributed training environment.
    """
    dist.init_process_group(
        backend='nccl',  # Use NCCL backend for GPU communication
        init_method=url,  # URL specifying the initialization method (IP + port)
        world_size=nprocs,  # Total number of processes
        rank=local_rank  # Rank of the current process
    )
    # Set the current GPU device
    torch.cuda.set_device(local_rank)


def create_model_and_optimizer(args):
    """
    Called during training.
    Creates model and optimizer, returns (model, optimizer, scheduler).
    """
    model = AttentionU2Net()

    # Move model to the specified device (local GPU)
    # print(args.local_rank)
    model = model.cuda(args.local_rank)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

    return model, optimizer, scheduler


def load_checkpoint(model, optimizer, checkpoints_dir, model_name, local_rank):
    """
    Load model and optimizer parameters from the specified path: checkpoints_dir + model_name.
    Returns (model, optimizer, start_epoch).
    """
    checkpoint_path = checkpoints_dir + model_name
    print("Loading model:", checkpoint_path)

    checkpoint = torch.load(checkpoint_path, map_location=f'cuda:{local_rank}')
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    start_epoch = checkpoint["epoch"]

    return model, optimizer, start_epoch


def wrap_model_distributed(model, local_rank):
    """
    Convert BatchNorm layers to SyncBatchNorm and wrap the model with DistributedDataParallel.
    """
    # Convert BatchNorm layers to SyncBatchNorm
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # Wrap with DistributedDataParallel for parallel training
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    return model


def create_dataloaders(args):
    """
    Create training/validation Dataset & DataLoader objects, and return (train_loader, val_loader, train_sampler, val_sampler)
    """
    # Training set
    train_set = MyDataset(
        csv_file='Underwater Dataset/train_list_withoutcleanwater.csv',
        root_dir='Underwater Dataset/Baseline_Data',
        transform=RandomCrop()  # RandomCrop for data augmentation
    )

    # Validation set
    val_set = MyDataset(
        csv_file='Underwater Dataset/val_list_withoutcleanwater.csv',
        root_dir='Underwater Dataset/Baseline_Data',
        transform=False
    )

    # Distributed samplers
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, shuffle=True, drop_last=True)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_set, shuffle=False, drop_last=True)

    # Adjusted batch size (total batch size / nprocs)
    train_batch_size = int(args.train_batch_size / args.nprocs)
    val_batch_size = int(args.val_batch_size / args.nprocs)

    # DataLoader
    train_loader = DataLoader(
        train_set,
        batch_size=train_batch_size,
        num_workers=16,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True
    )

    val_loader = DataLoader(
        val_set,
        batch_size=val_batch_size,
        num_workers=16,
        pin_memory=True,
        sampler=val_sampler,
        drop_last=True
    )

    return train_loader, val_loader, train_sampler, val_sampler


def test_dataloaders(args):
    """
    Create test set Dataset & DataLoader, and return (test_loader, test_sampler)
    """

    # Test set
    test_set = MyDataset(
        csv_file='Underwater Dataset/test_list_withoutcleanwater.csv',
        root_dir='Underwater Dataset/Baseline_Data',
        transform=False
    )

    # Distributed sampler
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_set, shuffle=False, drop_last=False)

    # Adjusted batch size (total batch size / nprocs)
    test_batch_size = int(args.test_batch_size / args.nprocs)

    # DataLoader
    test_loader = DataLoader(
        test_set,
        batch_size=test_batch_size,
        num_workers=4,
        pin_memory=True,
        sampler=test_sampler,
        drop_last=True
    )

    return test_loader, test_sampler


def save_checkpoint(model, optimizer, epoch, checkpoints_dir):
    """
    Save checkpoint including model parameters, optimizer state, and current epoch.
    """
    checkpoint = {
        "model": model.module.state_dict(),  # For DDP, access model.module
        "optimizer": optimizer.state_dict(),
        "epoch": epoch
    }
    save_path = f"{checkpoints_dir}/{epoch}.pth"
    torch.save(checkpoint, save_path)
    print(f"Model saved to:{save_path}")


def train_sfp(train_loader, model, criterion, optimizer, epoch, writer, local_rank, args, train_loss_list):
    model.train()
    running_loss = 0
    total_samples = 0
    for i, sample in enumerate(train_loader):
        images = sample['image']
        images.requires_grad_(True)
        images = images.cuda(local_rank, non_blocking=True)
        ground_truths = sample['ground_truth']
        ground_truths = ground_truths.cuda(local_rank, non_blocking=True) / 255.0
        mask = sample['mask']
        mask = mask.cuda(local_rank, non_blocking=True)  # False or True
        mask1 = torch.unsqueeze(mask, 1)
        inputs = sample['input']
        inputs.requires_grad_(True)
        inputs = inputs.cuda(local_rank, non_blocking=True)

        outputs = model(inputs)
        outputs = normalize(outputs, dim=1)

        loss = criterion(outputs, ground_truths, None, None, mask1, train_loader)

        # Synchronization barrier: processes wait here until all peers reach this point, guaranteeing precise and sequential output.
        torch.distributed.barrier()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = ground_truths.size(0)  # Get the batch size
        total_samples += batch_size
        running_loss += loss.item() * batch_size

    # Compute the total loss and sample count across all GPUs
    running_loss_tensor = torch.tensor([running_loss], dtype=torch.float64, device='cuda')
    total_samples_tensor = torch.tensor([total_samples], dtype=torch.float64, device='cuda')

    # Sum running_loss and total_samples across all GPUs
    running_loss_tensor = sync_tensor(running_loss_tensor)
    total_samples_tensor = sync_tensor(total_samples_tensor)

    epoch_loss = running_loss_tensor.item() / total_samples_tensor.item()
    if local_rank == 0:
        writer.add_scalar('training_loss', epoch_loss, epoch + 1)
    train_loss_list.append(epoch_loss)

    return model, train_loss_list


def val_sfp(val_loader, model, writer, epoch, local_rank, args, criterion, val_loss_list):
    model.eval()
    device = torch.device(f'cuda:{local_rank}')
    window = hann2d(PATCH, device).unsqueeze(0).unsqueeze(0)  # (1,1,256,256)

    total_loss, total_samples = 0., 0

    with torch.no_grad():
        for i, sample in enumerate(val_loader):

            inputs = sample['input'].cuda(device)  # B=1, C=3/4, H, W
            image = sample['image'].cuda(device)
            gt = sample['ground_truth'].float().cuda(device) / 255.
            mask = sample['mask'].unsqueeze(1).cuda(device)  # (1,1,H,W)
            gt *= mask

            H, W = inputs.shape[2:]
            # ----------- Prepare empty containers -----------
            out_sum = torch.zeros(1, 3, H, W, device=device)
            w_sum = torch.zeros(1, 1, H, W, device=device)

            # ----------- Sliding window -----------
            for y in range(0, H - PATCH + 1, STRIDE):
                for x in range(0, W - PATCH + 1, STRIDE):
                    patch = inputs[..., y:y + PATCH, x:x + PATCH]  # (1,C,256,256)
                    patch2 = image[..., y:y + PATCH, x:x + PATCH]
                    pred = model(patch)  # (1,3,256,256)
                    pred = pred * window  # Weighted
                    out_sum[..., y:y + PATCH, x:x + PATCH] += pred
                    w_sum[..., y:y + PATCH, x:x + PATCH] += window

            # ----------- Normalize to obtain the complete result -----------
            full_pred = out_sum / w_sum.clamp_min(1e-6)  # (1,3,H,W)
            full_pred = torch.nn.functional.normalize(full_pred, dim=1)
            full_pred *= mask

            # ----------- Angular MAE calculation -----------
            # Network outputs normal maps in [-1, 1] range; ground truth must be normalized to this interval for evaluation.
            # pred_n = (full_pred *2.0 -1.0) * mask
            gt_n = (gt * 2.0 - 1.0) * mask

            M = torch.sum(mask)  # Masked valid pixels
            cosine = nn.CosineSimilarity()
            m = torch.sum(cosine(full_pred, gt_n)) / M
            mae = torch.acos(m.clamp(-1 + 1e-6, 1 - 1e-6)) * 180 / pi

            total_loss += mae.item()
            total_samples += 1  # batch_size=1

            if i % 100 == 0 and local_rank == 0:
                img = ((full_pred + 1) * 0.5) * mask
                save_image(img, f'./results_sfp/{sample["filename"][0]}_{epoch}.bmp')

    # ----------- DDP Synchronization + Logging -----------
    val_mae_tensor = torch.tensor([total_loss], device=device)
    val_samples_tensor = torch.tensor([total_samples], device=device)
    val_mae_tensor = sync_tensor(val_mae_tensor)
    val_samples_tensor = sync_tensor(val_samples_tensor)
    val_mae_tensor = val_mae_tensor / val_samples_tensor
    if local_rank == 0:
        writer.add_scalar('validation_mae', val_mae_tensor.item(), epoch + 1)
    val_loss_list.append(val_mae_tensor.item())
    return val_loss_list


def create_window(window_size, channel):
    # Create Gaussian window
    sigma = 1.5  # Experience-based parameter
    coords = torch.arange(window_size).float() - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    window = g.ger(g).unsqueeze(0).unsqueeze(0)
    return window.repeat(channel, 1, 1, 1)


def draw_curve(train_loss_list, title):
    # Visualize using matplotlib
    plt.figure()
    plt.plot(train_loss_list, label=f'{title}')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title(f'{title} Curve')
    plt.legend()
    plt.grid(True)  # Enhance readability by adding gridlines
    plt.savefig(f'{title}.png')


def draw_two_curve(list_1, list_2, title_1, title_2):
    plt.figure()
    plt.plot(list_1, label=f'{title_1}', linestyle='-', marker='o')
    plt.plot(list_2, label=f'{title_2}', linestyle='--', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title(f'{title_1} and {title_2} Curve')
    plt.legend()
    plt.grid(True)  # Enhance readability by adding gridlines
    plt.savefig('two_curve.png')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'  # Current + Mean
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(
            batch)]  # Curly braces and format fields will be substituted by format() arguments
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    args.warmup_epochs = 0
    if epoch < args.warmup_epochs:
        lr *= float(epoch) / float(max(1.0, args.warmup_epochs))
        if epoch == 0:
            lr = 1e-6
    else:
        # progress after warmup
        if args.cos:  # cosine lr schedule
            # lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
            progress = float(epoch - args.warmup_epochs) / float(max(1, args.epochs - args.warmup_epochs))
            lr *= 0.5 * (1. + math.cos(math.pi * progress))
            # print("adjust learning rate now epoch %d, all epoch %d, progress"%(epoch, args.epochs))
        else:  # stepwise lr schedule
            for milestone in args.schedule:
                lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print("Epoch-{}, base lr {}, optimizer.param_groups[0]['lr']".format(epoch + 1, args.lr),
          optimizer.param_groups[0]['lr'])


def sync_tensor(tensor):
    """Synchronize tensor across all GPUs to ensure correct loss calculation"""
    if dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)  # Sum across all GPUs
    return tensor