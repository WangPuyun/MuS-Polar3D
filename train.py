import argparse
import os
import torch.multiprocessing as mp
import config as config
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.backends import cudnn
import torch.nn as nn
from tqdm import tqdm
import glob
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,4,5'

parser = argparse.ArgumentParser(description='PyTorch Network Training')
parser.add_argument("--model_name", type=str, default=None, help="Load pre-trained model to continue training. Default=None (train from scratch), set to '/**.pth' to resume training")
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument("--train_batch_size", type=int, default=16, help="Batch size for distributed training")
parser.add_argument("--val_batch_size", type=int, default=4, help="Batch size for distributed validation")
parser.add_argument('--event_dir', default="./runs", help='Directory for TensorBoard event files')
parser.add_argument("cos", action='store_true', help="use cos decay learning rate")
parser.add_argument("--epochs", type=int, default=1000)
parser.add_argument('--warmup_epochs', type=int, default=50, help='Number of warmup epochs for learning rate')
parser.add_argument('--checkpoints_dir', default="./pt/TransUNet", help='Path to model checkpoint files (for resuming training)')
args = parser.parse_args()

train_loss_list = []  # loss_list maintained only by main process
val_loss_list = []
lr_list = []
min_val_loss = 1000

def main():
    args.nprocs = torch.cuda.device_count()

    mp.spawn(main_worker, nprocs=args.nprocs, args=(args.nprocs, args))  # Multi-GPU (single node)

def main_worker(local_rank, nprocs,args):
    global train_loss_list, val_loss_list, lr_list, min_val_loss
    # 1. Initialize distributed training environment
    args.local_rank = local_rank
    config.init_distributed(local_rank=args.local_rank, nprocs=args.nprocs)

    # 2. Create model and optimizer
    model, optimizer,scheduler = config.create_model_and_optimizer(args)

    # 3. Optional: Load existing checkpoint
    start_epoch = 0
    if args.model_name:
        model, optimizer, start_epoch = config.load_checkpoint(model, optimizer,checkpoints_dir=args.checkpoints_dir,
            model_name=args.model_name, local_rank=args.local_rank )

    # 4. Convert to SyncBN and wrap with DDP
    model = config.wrap_model_distributed(model, local_rank=args.local_rank)

    # 5. Create loss function
    criterion = nn.CosineSimilarity().cuda(args.local_rank)# 3D Reconstruction

    # 6. Create data loader
    train_loader, val_loader, train_sampler, val_sampler = config.create_dataloaders(args)

    # 7. Initialize cuDNN and TensorBoard
    cudnn.benchmark = False # True: faster but memory-intensive; False: slower but memory-efficient
    writer = SummaryWriter(args.event_dir)  # Create event file

    # 8. Training and validation loop
    epoch_iter = tqdm(range(start_epoch, args.epochs), desc="Epoch Progress")

    for epoch in epoch_iter:
        # To maintain consistent data randomness in distributed training, set the sampler's epoch at each epoch
        train_sampler.set_epoch(epoch)  # Pass epoch to sampler for shuffling in training loop
        val_sampler.set_epoch(epoch)

        # config.adjust_learning_rate(optimizer, epoch, args)

        model,train_loss_list = config.train_sfp(train_loader, model, criterion, optimizer, epoch, writer, args.local_rank, args, train_loss_list)
        val_loss_list = config.val_sfp(val_loader, model, writer, epoch, args.local_rank, args, criterion, val_loss_list)
        torch.distributed.barrier()  # Synchronize all processes
        # scheduler.step(val_loss_list[-1])  # Update learning rate
        # Log learning rate (only on main process)
        if args.local_rank == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f'current_lr:{current_lr}')
            lr_list.append(current_lr)
            if val_loss_list[-1] < min_val_loss:
                min_val_loss = val_loss_list[-1]
                best_model_path = './pt/TransUNet_best'
                # Find all .pth files
                pth_files = glob.glob(os.path.join(best_model_path, '*.pth'))
                # Delete file
                for file_path in pth_files:
                    try:
                        os.remove(file_path)
                        print(f"Deleted: {file_path}")
                    except Exception as e:
                        print(f"Failed to delete {file_path}: {e}")
                config.save_checkpoint(model, optimizer,epoch+1, checkpoints_dir=best_model_path)

        # Display the latest loss in the progress bar
        epoch_iter.set_postfix(train_loss=train_loss_list[-1], val_loss=val_loss_list[-1])
        
    # 9. Save model at regular intervals
        if (epoch + 1) % 100 == 0 and args.local_rank == 0:

            # Create a dictionary structure
            data = {
                'epoch': list(range(1, len(train_loss_list) + 1)),
                'train_loss': train_loss_list,
                'val_loss': val_loss_list,
                'lr': lr_list
            }

            # Convert to DataFrame
            df = pd.DataFrame(data)

            # Save as CSV file
            df.to_csv(f'training_log_{epoch+1}.csv', index=False)

            config.draw_curve(train_loss_list, 'train loss_temp')
            config.draw_curve(val_loss_list, 'val loss_temp')
            config.draw_curve(lr_list, 'learning rate_temp')
            config.draw_two_curve(train_loss_list, val_loss_list, 'train loss', 'val loss')

            config.save_checkpoint(model, optimizer,epoch+1, checkpoints_dir=args.checkpoints_dir)
        torch.distributed.barrier()  # Synchronize all processes

    if args.local_rank == 0:
        # Create a dictionary structure
        data = {
            'epoch': list(range(1, len(train_loss_list) + 1)),
            'train_loss': train_loss_list,
            'val_loss': val_loss_list,
            'lr': lr_list
        }

        # Convert to DataFrame
        df = pd.DataFrame(data)

        # Save as CSV file
        df.to_csv('training_log.csv', index=False)
        writer.close()
        config.draw_curve(train_loss_list, 'train loss')
        config.draw_curve(val_loss_list, 'val loss')
        config.draw_curve(lr_list, 'learning rate')
        config.draw_two_curve(train_loss_list, val_loss_list, 'train loss', 'val loss')

if __name__ == "__main__":
    main()