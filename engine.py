# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import math
import os
import sys
import torch
import torch.distributed as dist

from tqdm import tqdm
from typing import Iterable

import utils.misc as utils
import utils.loss_utils as loss_utils
import utils.eval_utils as eval_utils

import numpy as np

def train_one_epoch(model: torch.nn.Module, data_loader: Iterable,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):

    # model = torch.compile(model)

    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    accumulated_iterations = 0
    for batch, it in metric_logger.log_every(data_loader, print_freq, header):

        img_data, text_data, target, r1_mask, bbox_mask, a, b = batch

        # copy to GPU
        img_data = img_data.to(device)
        text_data = text_data.to(device)
        target = target.to(device)
        a = a.to(device)
        b = b.to(device)
        r1_mask = r1_mask.to(device)
        bbox_mask = bbox_mask.to(device)
        # sig_img = sig_img.to(device)
        # negative_mask = negative_mask.to(device)

        # model forward
        # torch.autograd.set_detect_anomaly(True)
        output = model(img_data, text_data, r1_mask)

        loss_dict = loss_utils.zmod_loss(output, [target, a, b], bbox_mask)
        losses = sum(loss_dict[k] for k in loss_dict.keys())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {k: v for k, v in loss_dict_reduced.items()}
        losses_reduced_unscaled = sum(loss_dict_reduced_unscaled.values())
        loss_value = losses_reduced_unscaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        accumulated_iterations += 1
        if accumulated_iterations == 1:
            # Update parameters every four iterations
            optimizer.step()
            optimizer.zero_grad()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            accumulated_iterations = 0

        metric_logger.update(loss=loss_value, **loss_dict_reduced_unscaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def validate(model: torch.nn.Module, data_loader: Iterable, device: torch.device, amp: bool = False):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Eval:'

    for batch, i in metric_logger.log_every(data_loader, 10, header):
        img_data, text_data, target, r1_mask, bbox_mask, a, b = batch
        batch_size = img_data.tensors.size(0)
        # copy to GPU
        img_data = img_data.to(device)
        text_data = text_data.to(device)
        target = target.to(device)
        # sig_img = sig_img.to(device)
        r1_mask = r1_mask.to(device)
        
        outputs = model(img_data, text_data, r1_mask)
        iou, accu = eval_utils.zmod_eval_val(outputs['bbox']['pred_boxes'], target)
        iou_value = torch.mean(iou).item()
        accu_value = accu.item()
        metric_logger.update_v2('iou', iou_value, batch_size)
        metric_logger.update_v2('accu', accu_value, batch_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return stats


import time
@torch.no_grad()
def evaluate(model: torch.nn.Module, data_loader: Iterable, device: torch.device):
    model.eval()

    pred_box_list = []
    gt_box_list = []
    total_time = 0.0
    for _, batch in enumerate(tqdm(data_loader)):
        img_data, text_data, target, r1_mask, bbox_mask, a, b = batch
        batch_size = img_data.tensors.size(0)
        # copy to GPU
        img_data = img_data.to(device)
        text_data = text_data.to(device)
        target = target.to(device)
        r1_mask = r1_mask.to(device)
        with torch.no_grad():
            start_time = time.time()  # 记录开始时间
            output = model(img_data, text_data, r1_mask)
            end_time = time.time()  # 记录结束时间
            elapsed_time = end_time - start_time
            total_time += elapsed_time
            print(total_time / (_+1))
        pred_box_list.append(output['bbox']['pred_boxes'].cpu())
        gt_box_list.append(target.cpu())

    pred_boxes = torch.cat(pred_box_list, dim=0)
    gt_boxes = torch.cat(gt_box_list, dim=0)
    total_num = gt_boxes.shape[0]
    accu_num = eval_utils.zmod_eval_test(pred_boxes, gt_boxes)

    result_tensor = torch.tensor([accu_num, total_num]).to(device)
    
    # torch.cuda.synchronize()
    # dist.all_reduce(result_tensor)

    accuracy = float(result_tensor[0]) / float(result_tensor[1])
    
    return accuracy
        