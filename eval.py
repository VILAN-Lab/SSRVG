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
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, DistributedSampler

import datasets
import utils.misc as utils
from models import build_model
from datasets import build_dataset
from engine import train_one_epoch, evaluate


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=5e-5, type=float)
    parser.add_argument('--lr_bert', default=1e-6, type=float)
    parser.add_argument('--lr_visu_cnn', default=1e-6, type=float)
    # parser.add_argument('--lr_visu_tra', default=1e-6, type=float)
    parser.add_argument('--lr_clip', default=0, type=float)
    parser.add_argument('--batch_size', default=20, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=90, type=int)
    parser.add_argument('--lr_power', default=0.9, type=float, help='lr poly power')
    parser.add_argument('--clip_max_norm', default=0., type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--eval', dest='eval', default=False, action='store_true', help='if evaluation only')
    parser.add_argument('--optimizer', default='adamw', type=str)
    parser.add_argument('--lr_scheduler', default='cosine', type=str)
    parser.add_argument('--lr_drop', default=[70, 90], type=int)
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
                        help="resnet101 / SwinTransformer")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--num_feature_levels', default=3, type=int, help='number of feature levels')

    # * Attention
    parser.add_argument('--enc_layers', default=0, type=int,
                        help='Number of encoders')
    parser.add_argument('--dec_layers', default=3, type=int,
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
    parser.add_argument('--dataset', default='referit', type=str,
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
    parser.add_argument('--pretrained_model', default='./checkpoints/cascade_mask_rcnn_swin_base_patch4_window7.pth',
                        type=str,
                        help='./checkpoints/detr-r101.pth or ./checkpoints/cascade_mask_rcnn_swin_base_patch4_window7.pth')
    parser.add_argument('--bert_model', default='bert-base-uncased', type=str, help='bert model')
    parser.add_argument('--light', dest='light', default=False, action='store_true', help='if use smaller model')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=8, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    # evalutaion options
    parser.add_argument('--eval_set', default='test', type=str)
    parser.add_argument('--eval_model', default='/home/user/D/32T/zsy/tmm_msvlt/outputs/five_total_train_finetune_referit/best_checkpoint.pth', type=str)
    return parser


def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    device = torch.device(args.device)

    # # fix the seed for reproducibility
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

    # build dataset
    dataset_test = build_dataset(args.eval_set, args)
    ## note certain dataset does not have 'test' set:
    ## 'unc': {'train', 'val', 'trainval', 'testA', 'testB'}
    # dataset_test  = build_dataset('test', args)
    
    if args.distributed:
        sampler_test = DistributedSampler(dataset_test, shuffle=False)
    else:
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)
    
    batch_sampler_test = torch.utils.data.BatchSampler(
        sampler_test, args.batch_size, drop_last=False)

    data_loader_test = DataLoader(dataset_test, args.batch_size, sampler=sampler_test,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    checkpoint = torch.load(args.eval_model, map_location='cpu')
    a,b = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
    print(a)

    # output log
    output_dir = Path(args.output_dir)
    if args.output_dir and utils.is_main_process():
        with (output_dir / "eval_log.txt").open("a") as f:
            f.write(str(args) + "\n")
    
    start_time = time.time()
    
    # perform evaluation
    accuracy = evaluate(model, data_loader_test, device)
    
    if utils.is_main_process():
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))

        log_stats = {'test_model:': args.eval_model,
                    '%s_set_accuracy'%args.eval_set: accuracy,
                    }
        print(log_stats)
        if args.output_dir and utils.is_main_process():
                with (output_dir / "eval_log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('M3T evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
