import argparse
import datetime
import json
import random
import time
import math
import os

import numpy as np
from pathlib import Path

import torch
from torch.utils.data import DataLoader, DistributedSampler

import utils.misc as utils
from models import build_model
from datasets import build_dataset
from engine import train_one_epoch, validate


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_bert', default=1e-6, type=float)
    parser.add_argument('--lr_visu_cnn', default=1e-6, type=float)
    # parser.add_argument('--lr_visu_tra', default=1e-6, type=float)
    parser.add_argument('--lr_clip', default=0, type=float)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=120, type=int)
    parser.add_argument('--lr_power', default=0.9, type=float, help='lr poly power')
    parser.add_argument('--clip_max_norm', default=0., type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--eval', dest='eval', default=False, action='store_true', help='if evaluation only')
    parser.add_argument('--optimizer', default='adamw', type=str)
    parser.add_argument('--lr_scheduler', default='cosine', type=str)
    parser.add_argument('--lr_drop', default=[100, 180], type=int)
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    parser.add_argument('--wpa_loss', dest='wpa_loss', action='store_true',
                        help="Disables wpa losses (loss at encoder)")
    # parser.add_argument('--no_amp', dest='amp', action='store_false',
    #                     help="Disables amp")
    parser.add_argument('--no_multi_scale', dest='multi_scale', action='store_false',
                        help="Disables multi_scale")

    # Augmentation options
    parser.add_argument('--aug_blur', action='store_true',
                        help="If true, use gaussian blur augmentation")
    parser.add_argument('--aug_crop', action='store_true',
                        help="If true, use random crop augmentation")
    parser.add_argument('--aug_scale', action='store_true',
                        help="If true, use multi-scale augmentation")
    parser.add_argument('--aug_translate', action='store_true',
                        help="If true, use random translate augmentation")
    # ZMode
    # * Backbone
    parser.add_argument('--backbone', default='swin-base', type=str,
                        help="resnet101 / swin-base / swin-tiny / swin-small / resnet152")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--num_feature_levels', default=3, type=int, help='number of feature levels')

    # * Attention
    parser.add_argument('--enc_layers', default=0, type=int,
                        help='Number of encoders')
    parser.add_argument('--dec_layers', default=1, type=int,
                        help='Number of decoders')
    parser.add_argument('--ca_layers', default=3, type=int,
                        help='Number of decoders')
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the igmia blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help='Size of the embeddings (dimension of the igmia)')
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the igmia transformer")
    parser.add_argument('--erase', default=0.1, type=float,
                        help="Dropout applied in the igmia transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the igmia")
    parser.add_argument('--activation', default='gelu', type=str,
                        help="Number of attention heads inside the igmia")
    parser.add_argument('--pre_norm', action='store_true')
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)

    # * Score Module
    parser.add_argument('--alpha', default=1.0, type=float)
    parser.add_argument('--sigma', default=0.5, type=float)

    # BERT
    parser.add_argument('--bert_enc_num', default=12, type=int)

    # Dataset parameters
    parser.add_argument('--data_root', type=str, default='./ln_data/',
                        help='path to ReferIt splits data folder')
    parser.add_argument('--split_root', type=str, default='data',
                        help='location of pre-parsed dataset info')
    parser.add_argument('--dataset', default='gref_umd', type=str,
                        help='referit/unc/unc+/gref/gref_umd/flickr')
    parser.add_argument('--max_query_len', default=40, type=int,
                        help='maximum time steps (lang length) per batch')
    parser.add_argument('--imsize', default=640, type=int, help='image size')

    parser.add_argument('--output_dir', default='./outputs',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=13, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--resume_scheduler', default=0, type=bool)
    parser.add_argument('--pretrained_model', default='./checkpoints/cascade_mask_rcnn_swin_base_patch4_window7.pth', type=str,
                        help='./checkpoints/detr-r101.pth'
                             './checkpoints/cascade_mask_rcnn_swin_base_patch4_window7.pth'
                             './checkpoints/cascade_mask_rcnn_swin_small_patch4_window7.pth'
                             './checkpoints/cascade_mask_rcnn_swin_tiny_patch4_window7.pth'
                             './checkpoints/resnet152_pretrained.pth')
    parser.add_argument('--bert_model', default='bert-base-uncased', type=str, help='bert model')
    parser.add_argument('--light', dest='light', default=False, action='store_true', help='if use smaller model')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=8, type=int)


    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser

import shutil
import os
def save_py(dir_list, destination_path):
    for i in dir_list:
        if os.path.isfile(i):
            print(f"{i} 是一个文件")
            shutil.copy(i, os.path.join(destination_path, os.path.basename(i)))
        elif os.path.isdir(i):
            print(f"{i} 是一个文件夹")
            shutil.copytree(i, os.path.join(destination_path, os.path.basename(i)))



def main(args):
    #
    # dir_list = ["./datasets", "./models", "./utils", "./engine.py", "./eval.py", "./fineturn.py", "./train.py",
    #             "./train_3090.sh"]
    # save_py(dir_list, args.output_dir)

    utils.init_distributed_mode(args)
    device = torch.device(args.device)

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # build model
    model = build_model(args)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    visu_cnn_param = [p for n, p in model_without_ddp.named_parameters() if
                      (("backbone" in n) and p.requires_grad)]
    # visu_tra_param = [p for n, p in model_without_ddp.named_parameters() if
    #                   (("visumodel" in n) and ("backbone" not in n) and p.requires_grad)]
    text_tra_param = [p for n, p in model_without_ddp.named_parameters() if
                      (("bert" in n) and p.requires_grad)]
    clip_parma = [p for n, p in model_without_ddp.named_parameters() if
                  (("clip" in n) and p.requires_grad)]
    rest_param = [p for n, p in model_without_ddp.named_parameters() if
                  (("backbone" not in n) and ("bert" not in n) and ("clip" not in n) and p.requires_grad)]

    param_list = [{"params": rest_param},
                  {"params": visu_cnn_param, "lr": args.lr_visu_cnn},
                  {"params": text_tra_param, "lr": args.lr_bert},
                  {"params": clip_parma, "lr": args.lr_clip}]

    # using RMSProp or AdamW
    if args.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(param_list, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(param_list, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(param_list, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(param_list, lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)
    else:
        raise ValueError('Lr scheduler type not supportted ')
    # # using AMP
    # if args.amp:
    #     scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    # else:
    #     scaler = None

    # using polynomial lr scheduler or half decay every 10 epochs or step
    lr = [args.lr, args.lr_visu_cnn, args.lr_bert, args.lr_clip]
    if args.lr_scheduler == 'poly':
        lr_func = lambda epoch: (1 - epoch / args.epochs) ** args.lr_power
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_func)
    elif args.lr_scheduler == 'halfdecay':
        lr_func = lambda epoch: 0.5 ** (epoch // (args.epochs // 5))
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_func)
    elif args.lr_scheduler == 'cosine':
        # lr_func = lambda epoch: 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
        # lr_func = lambda epoch: 0.45 * (
        #             1.22 + math.cos(math.pi * epoch / (0.9 * args.epochs))) if epoch < 0.9 * args.epochs else 0.1
        # lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_func)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=1e-8, last_epoch=-1)
    elif args.lr_scheduler == 'step':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.lr_drop)
    elif args.lr_scheduler == 'onecycle':
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr,
                                                           total_steps=args.epochs,
                                                           div_factor=10.,
                                                           pct_start=0.1)
    else:
        raise ValueError('Lr scheduler type not supportted ')

    # build dataset
    dataset_train = build_dataset('train', args)
    dataset_val = build_dataset('val', args)
    ## note certain dataset does not have 'test' set:
    ## 'unc': {'train', 'val', 'trainval', 'testA', 'testB'}

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train, shuffle=True)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        a,b = model_without_ddp.load_state_dict(checkpoint['model'],strict=False)
        print(a)
        print(b)
        if args.resume_scheduler==1:
            if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
                args.start_epoch = checkpoint['epoch'] + 1
    elif args.pretrained_model is not None:
        checkpoint = torch.load(args.pretrained_model, map_location='cpu')
        clip_checkpoint = torch.load('./checkpoints/clip_RN50x64.pt', map_location='cpu')
        # state_dict
        if "swin" in args.backbone:
            missing_keys, unexpected_keys = model_without_ddp.backbone[0].body.load_state_dict(checkpoint['state_dict'],strict=False)
        elif "resnet101" in args.backbone:
            check = checkpoint['model']
            new_checkpoints = {}
            for i in check:
                if i.startswith("backbone.0.body"):
                    j = "backbone." + i[16:]
                    new_checkpoints[j] = checkpoint['model'][i]
            missing_keys, unexpected_keys = model_without_ddp.backbone[0].body.load_state_dict(new_checkpoints,strict=False)
        elif "resnet152" in args.backbone:
            # missing_keys, unexpected_keys = model_without_ddp.load_state_dict(checkpoint['state_dict'], strict=False)
            pass
        else:
            missing_keys, unexpected_keys = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
            print(missing_keys)
        missing_keys, unexpected_keys = model_without_ddp.clip.load_state_dict(clip_checkpoint, strict=False)

        # print('Unexpected keys when loading resnet model:')
        # print(unexpected_keys)

    output_dir = Path(args.output_dir)
    if args.output_dir and utils.is_main_process():
        with (output_dir / "log.txt").open("a") as f:
            f.write(str(args) + "\n")

    print("Start training")
    start_time = time.time()
    best_accu = 0
    print(args.start_epoch, args.epochs)
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)

        train_stats = train_one_epoch(model, data_loader_train, optimizer,
                                      device, epoch, args.clip_max_norm)
        lr_scheduler.step()

        val_stats = validate(model, data_loader_val, device)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'validation_{k}': v for k, v in val_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # checkpoint_paths = []
            # extra checkpoint before LR drop and every 10 epochs
            if (epoch + 1) % 10 == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')

            if val_stats['accu'] > best_accu:
                checkpoint_paths.append(output_dir / 'best_checkpoint.pth')
                best_accu = val_stats['accu']

            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                    'val_accu': val_stats['accu']
                }, checkpoint_path)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('ZMod training script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
