import torch
import numpy as np
import torch.nn.functional as F

from utils.box_utils import bbox_iou, xywh2xyxy, xyxy2xywh, generalized_box_iou
from utils.misc import get_world_size


def zmod_loss(outputs, targets, bbox_mask):
    """Compute the losses related to the bounding boxes, 
       including the L1 regression loss and the GIoU loss
    """

    losses = {}
    bbox = outputs['bbox']
    # losses['head_bbox'], losses['head_giou'] = get_bbox_loss(bbox['head_pred_boxes'], targets)
    # losses['context_bbox'], losses['context_giou'] = get_bbox_loss(bbox['context_pred_boxes'], targets)

    losses['fuse_bbox_context'], losses['fuse_giou_context'] = get_bbox_loss(outputs['bbox_first']['pred_boxes'], targets[2])
    losses['fuse_bbox'], losses['fuse_giou'] = get_bbox_loss(bbox['pred_boxes'], targets[1])

    if 'aux_outputs' in bbox:
        num_layer = len(bbox['aux_outputs'])
        for i, aux_outputs in enumerate(bbox['aux_outputs']):
            loss_bbox, loss_giou = get_bbox_loss(bbox['aux_outputs'][i]['pred_boxes'][-1], targets[1])
            losses['fuse_bbox'] += loss_bbox / num_layer
            losses['fuse_giou'] += loss_giou / num_layer
        losses['fuse_bbox'] /= 2
        losses['fuse_giou'] /= 2

    return losses


def get_bbox_loss(outputs, targets):
    batch_size = outputs.shape[0]
    # world_size = get_world_size()
    num_boxes = batch_size

    loss_bbox = F.l1_loss(outputs, targets, reduction='none')
    loss_giou = 1 - torch.diag(generalized_box_iou(
        xywh2xyxy(outputs),
        xywh2xyxy(targets)
    ))
    loss_bbox = loss_bbox.sum() / num_boxes
    loss_giou = loss_giou.sum() / num_boxes
    return loss_bbox, loss_giou


def focal_loss(inputs, targets, alpha: float = 0.8, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    bs = inputs.shape[0]
    prob = inputs
    ce_loss = F.binary_cross_entropy(prob, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean()
