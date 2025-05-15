import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
import torch.distributed as dist
import torch.multiprocessing as mp

from model import Restormer
from utils import parse_args, RainDataset, rgb_to_y, psnr, ssim,GaussianDenoisingDataset

def set_seed(seed, rank):
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_ddp(rank, world_size, backend):
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['LOCAL_RANK'] = str(rank)
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    if backend == "nccl":
        torch.cuda.set_device(rank)

def cleanup_ddp():
    dist.destroy_process_group()

def test_loop(model, data_loader, num_iter, args, rank, device):
    model.eval()
    total_psnr, total_ssim, count = 0.0, 0.0, 0
    
    with torch.no_grad():
        test_bar = tqdm(data_loader, initial=1, dynamic_ncols=True) if rank == 0 else data_loader
        for rain, norain, name, h, w in test_bar:
            rain, norain = rain.to(device, non_blocking=True), norain.to(device, non_blocking=True)
            out = torch.clamp((torch.clamp(model(rain)[:, :, :h, :w], 0, 1).mul(255)), 0, 255).byte()
            norain = torch.clamp(norain[:, :, :h, :w].mul(255), 0, 255).byte()
            y, gt = rgb_to_y(out.double()), rgb_to_y(norain.double())
            current_psnr, current_ssim = psnr(y, gt), ssim(y, gt)
            total_psnr += current_psnr.item()
            total_ssim += current_ssim.item()
            count += 1
            if rank == 0:
                save_path = '{}/{}/{}'.format(args.save_path, args.data_name, name[0])
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                Image.fromarray(out.squeeze(dim=0).permute(1, 2, 0).contiguous().cpu().numpy()).save(save_path)
                test_bar.set_description('Test Iter: [{}/{}] PSNR: {:.2f} SSIM: {:.3f}'
                                         .format(num_iter, args.num_iter, total_psnr / count, total_ssim / count))

    # Convert to tensors for all-reduce
    total_psnr_tensor = torch.tensor([total_psnr, count], dtype=torch.float32, device=device)
    total_ssim_tensor = torch.tensor([total_ssim, count], dtype=torch.float32, device=device)
    
    # All-reduce to get sum from all ranks
    dist.all_reduce(total_psnr_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_ssim_tensor, op=dist.ReduceOp.SUM)
    
    # Calculate final average
    final_psnr = total_psnr_tensor[0].item() / total_psnr_tensor[1].item()
    final_ssim = total_ssim_tensor[0].item() / total_ssim_tensor[1].item()
    
    return final_psnr, final_ssim

def save_loop(model, data_loader, num_iter, results, best_psnr, best_ssim, args, rank, device):
    val_psnr, val_ssim = test_loop(model, data_loader, num_iter, args, rank, device)
    if rank == 0:
        results['PSNR'].append('{:.2f}'.format(val_psnr))
        results['SSIM'].append('{:.3f}'.format(val_ssim))
        df = pd.DataFrame(data=results, index=range(1, (num_iter if args.model_file else num_iter // 1000) + 1))
        df.to_csv(f'{args.save_path}/{args.data_name}.csv', index_label='Iter', float_format='%.3f')
        if val_psnr > best_psnr and val_ssim > best_ssim:
            best_psnr, best_ssim = val_psnr, val_ssim
            with open(f'{args.save_path}/{args.data_name}.txt', 'w') as f:
                f.write(f'Iter: {num_iter} PSNR:{best_psnr:.2f} SSIM:{best_ssim:.3f}')
            torch.save(model.module.state_dict(), f'{args.save_path}/{args.data_name}.pth')
    return best_psnr, best_ssim

def main_worker(rank, world_size, args):
    print(f"[Rank {rank}] Starting DDP setup...")
    setup_ddp(rank, world_size, args.backend)
    print(f"[Rank {rank}] DDP setup done.")

    set_seed(args.seed, rank)

    device = torch.device(f"cuda:{rank}" if args.backend == "nccl" else "cpu")
    print(f"[Rank {rank}] Using device {device}")

    print(f"[Rank {rank}] Initializing model...")
    model = Restormer(args.num_blocks, args.num_heads, args.channels, args.num_refinement, args.expansion_factor).to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank] if args.backend == "nccl" else None)
    print(f"[Rank {rank}] Model initialized. Total parameters: {sum(p.numel() for p in model.parameters())}")

    print(f"[Rank {rank}] Loading test dataset...")
    if args.task_type!='gaussian_denoise':
        test_dataset = RainDataset(args.task_type,args.data_path, args.data_name, 'test')
    else:
        test_dataset = GaussianDenoisingDataset(args.data_path, args.data_name, 'test',(25,25))

    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, sampler=test_sampler, num_workers=args.workers, pin_memory=True)
    print(f"[Rank {rank}] Test dataset loaded.")

    results, best_psnr, best_ssim = {'PSNR': [], 'SSIM': [], 'Loss': []}, 0.0, 0.0

    if args.model_file:
        print(f"[Rank {rank}] Loading pre-trained model from {args.model_file}")
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank} if args.backend == "nccl" else "cpu"
        model.load_state_dict(torch.load(args.model_file, map_location=map_location))
        save_loop(model, test_loader, 1, results, best_psnr, best_ssim, args, rank, device)
    else:
        print(f"[Rank {rank}] Starting training loop.")
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=args.num_iter, eta_min=1e-6)

        total_loss, total_num, i = 0.0, 0, 0
        train_bar = tqdm(range(1, args.num_iter + 1), initial=1, dynamic_ncols=True) if rank == 0 else range(1, args.num_iter + 1)
        for n_iter in train_bar:
            if n_iter == 1 or n_iter - 1 in args.milestone:
                end_iter = args.milestone[i] if i < len(args.milestone) else args.num_iter
                start_iter = args.milestone[i - 1] if i > 0 else 0
                length = args.batch_size[i] * (end_iter - start_iter)
                print(f"[Rank {rank}] Loading train dataset for stage {i}...")
                if args.task_type!='gaussian_denoise':
                    train_dataset = RainDataset(args.task_type,args.data_path, args.data_name, 'train', args.patch_size[i], length*world_size)
                else:
                    train_dataset = GaussianDenoisingDataset(args.data_path, args.data_name, 'train', args.patch_size[i], length*world_size)

                train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
                train_loader = iter(DataLoader(train_dataset, args.batch_size[i], sampler=train_sampler, num_workers=args.workers, pin_memory=True))
                print(f"[Rank {rank}] Train dataset loaded for stage {i}.")
                i += 1

            train_sampler.set_epoch(n_iter)

            model.train()
            rain, norain, name, h, w = next(train_loader)
            rain, norain = rain.to(device, non_blocking=True), norain.to(device, non_blocking=True)
            out = model(rain)
            loss = F.l1_loss(out, norain)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_num += rain.size(0)
            total_loss += loss.item() * rain.size(0)

            if rank == 0:
                train_bar.set_description('Train Iter: [{}/{}] Loss: {:.3f}'.format(n_iter, args.num_iter, total_loss / total_num))

            lr_scheduler.step()
            if n_iter % 1000 == 0 and rank == 0:
                results['Loss'].append('{:.3f}'.format(total_loss / total_num))
                best_psnr, best_ssim = save_loop(model, test_loader, n_iter, results, best_psnr, best_ssim, args, rank, device)

    print(f"[Rank {rank}] Training complete. Cleaning up DDP.")
    cleanup_ddp()

def main():
    args = parse_args()
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    if args.backend == "nccl":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    main_worker(rank, world_size, args)

if __name__ == '__main__':
    main()
