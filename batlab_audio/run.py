import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import timm.optim.optim_factory as optim_factory
from torch.utils.tensorboard import SummaryWriter

import timm
from timm.models.layers import trunc_normal_
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import utils.lr_decay as lrd
import utils.misc as misc
from utils.pos_embed import interpolate_pos_embed
from utils.misc import NativeScalerWithGradNormCount as NativeScaler

import models.models_mae as models_mae

from batlab_audio.engine_run import train_one_epoch, evaluate
from batlab_audio.dataset import get_training_set, get_test_set, get_validation_set


def get_args_parser():
    parser = argparse.ArgumentParser('batlab_audio imputation', add_help=False)
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    parser.add_argument('--print_freq', default=200, type=int)

    # Model parameters
    parser.add_argument('--model', default='mae_vit_base_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--model_type', default='vit', type=str, metavar='MODEL',
                        help='Type of model to train (vit or swin)')
    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    
    parser.add_argument('--train_decoder_only', action='store_true', default=False,
                        help='Instead of training the full model, only train the decoder half')
    
    parser.add_argument('--train_last_layer_only', action='store_true', default=False,
                        help='Instead of training the full model, only train the last layer (after the decoder)')

    # Optimizer parameters
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--layer_decay', type=float, default=0.75,
                        help='layer-wise lr decay from ELECTRA/BEiT')

    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR')

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=None, metavar='PCT',
                        help='Color jitter factor (enabled only when not using Auto/RandAug)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')
    
    parser.add_argument('--mask_ratio', default=0.1, type=float,
                        help='Masking ratio (percentage of removed patches).')
    parser.add_argument('--mask_type', default='random', choices=['random', 'chunked'],
                        help='Whether to mask random patches, chunks of patches, or the block of time [T, end], where T is random.')
    parser.add_argument('--avg_chunk_height', default=2, type=float,
                        help='If using chunked masking, average height (in patches) of a chunk')
    parser.add_argument('--avg_chunk_width', default=4, type=float,
                        help='If using chunked masking, average width (in patches) of a chunk')

    # Dataset parameters
    parser.add_argument('--data_path_train', default='./batlab_audio/single_chirp_data/batlab_data_train_mp3.hdf', type=str,
                        help='train dataset path')
    parser.add_argument('--data_path_val', default='./batlab_audio/single_chirp_data/batlab_data_eval_mp3.hdf', type=str,
                        help='validation dataset path')
    parser.add_argument('--data_path_test', default='./batlab_audio/single_chirp_data/batlab_data_test_mp3.hdf', type=str,
                        help='test dataset path')
    parser.add_argument('--norm_file', default='./batlab_audio/mean_std_128.npy', type=str,
                        help='norm file path')
    parser.add_argument('--sample_rate', default=32000, type=int)
    parser.add_argument('--hop_size', default=10, type=int)
    parser.add_argument('--clip_length', default=10, type=int)
    parser.add_argument('--augment', default=True, type=bool)
    parser.add_argument('--in_mem', default=False, type=bool)
    parser.add_argument('--extra_augment', default=True, type=bool)
    parser.add_argument('--roll', default=True, type=bool)
    parser.add_argument('--wavmix', default=True, type=bool)
    parser.add_argument('--specmix', default=True, type=bool)
    parser.add_argument('--mixup_alpha', default=0.3, type=float)

    parser.add_argument('--u_patchout', default=200, type=int,
                        help='number of masked patches')
    parser.add_argument('--target_size', default=(128,1000), type=tuple,
                        help='target size')

    parser.add_argument('--output_dir', default='./batlab_audio/output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='../batlab_audio/log_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='./checkpoint-399.pth',
                        help='resume from checkpoint')
    parser.add_argument('--resume_dir', default='',
                        help='resume dir')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=False)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser


def main(args):
    misc.init_distributed_mode(args)
    args.distributed = False

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    if args.resume_dir and not args.resume:
        tag = ''
        for root, dirs, files in os.walk(args.resume_dir, topdown=False):
            for name in files:
                if name[-3:] == 'pth':
                    if not tag:
                        tag = os.path.join(root, name)
                    elif int(name.split('checkpoint-')[1].split('.pth')[0]) > int(tag.split('checkpoint-')[1].split('.pth')[0]):
                        tag = os.path.join(root, name)
        args.resume = tag

    cudnn.benchmark = True

    num_tasks = misc.get_world_size() # set datasets and dataloaders
    global_rank = misc.get_rank()

    dataset_train = get_training_set(
        train_hdf5=args.data_path_train, 
        sample_rate=args.sample_rate, 
        augment=args.augment, 
        in_mem=args.in_mem, 
        extra_augment=args.extra_augment, 
        roll=args.roll,
        wavmix=args.wavmix)
    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    print("Sampler_train = %s" % str(sampler_train))
    dataset_test = get_test_set(
        eval_hdf5=args.data_path_test, 
        sample_rate=args.sample_rate)
    if args.dist_eval:
        if len(dataset_test) % num_tasks != 0:
            print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                    'This will slightly alter validation results as extra duplicate entries are added to achieve '
                    'equal num of samples per-process.')
        sampler_test = torch.utils.data.DistributedSampler(
            dataset_test, num_replicas=num_tasks, rank=global_rank, shuffle=True)  # shuffle=True to reduce monitor bias
    else:
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)
    print("Sampler_test = %s" % str(sampler_test))
    dataset_val = get_validation_set(
        validation_hdf5=args.data_path_val, 
        sample_rate=args.sample_rate)
    if args.dist_eval:
        if len(dataset_val) % num_tasks != 0:
            print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                    'This will slightly alter validation results as extra duplicate entries are added to achieve '
                    'equal num of samples per-process.')
        sampler_val = torch.utils.data.DistributedSampler(
            dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True)  # shuffle=True to reduce monitor bias
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    print("Sampler_val = %s" % str(sampler_val))

    if global_rank == 0 and args.log_dir is not None and not args.eval:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, sampler=sampler_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    model = models_mae.__dict__[args.model](
        norm_pix_loss=False,
        norm_file=args.norm_file,
        hopsize=args.hop_size,
        device=device,
        mask_type=args.mask_type,
        specgram_type='stft',
        adaptive_hopsize=True,
        chunked_mask_mean_height=args.avg_chunk_height,
        chunked_mask_mean_width=args.avg_chunk_width,
        train_decoder_only=args.train_decoder_only,
        train_last_layer_only=args.train_last_layer_only
    )
    checkpoint = torch.load(args.resume, map_location='cpu')
    model.load_state_dict(checkpoint['model'])

    print(device)
    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # build optimizer with layer-wise lr decay (lrd)
    param_groups = optim_factory.param_groups_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    loss_scaler = NativeScaler()

    if args.specmix or args.wavmix:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    print("criterion = %s" % str(criterion))

    if args.eval:
        test_stats = evaluate(data_loader_test, model, device, mask_ratio=args.mask_ratio)
        print(f"Loss of the network on the {len(dataset_test)} test images: {test_stats['loss']:.1f}")
        exit(0)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    min_loss_val = float('inf')
    min_loss_test = float('inf')
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )
        if args.output_dir and epoch % 5 == 0:
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        val_stats = evaluate(data_loader_val, model, device, mask_ratio=args.mask_ratio)
        # print(f"Loss of the network on the {len(dataset_val)} val images: {val_stats['loss']:.1f}")
        min_loss_val = min(min_loss_val, val_stats["loss"])
        print(f'Min loss Val: {min_loss_val:.4f}')

        test_stats = evaluate(data_loader_test, model, device, mask_ratio=args.mask_ratio)
        # print(f"Loss of the network on the {len(dataset_test)} test images: {test_stats['loss']:.1f}")
        min_loss_test = min(min_loss_test, test_stats["loss"])
        print(f'Min loss Test: {min_loss_test:.4f}')

        if log_writer is not None:
            log_writer.add_scalar('perf/test_loss', test_stats['loss'], epoch)
            log_writer.add_scalar('perf/val_loss', val_stats['loss'], epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'test_{k}': v for k, v in test_stats.items()},
                        'epoch': epoch,
                        'n_parameters': n_parameters}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
