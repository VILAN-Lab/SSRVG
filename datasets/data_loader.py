# -*- coding: utf-8 -*-

"""
ReferIt, UNC, UNC+ and GRef referring image segmentation PyTorch dataset.

Define and group batches of images, segmentations and queries.
Based on:
https://github.com/chenxi116/TF-phrasecut-public/blob/master/build_batches.py
"""

import os
import re
import cv2
import sys
import json
import torch
import numpy as np
import os.path as osp
import scipy.io as sio
import torch.utils.data as data


sys.path.append('.')

from PIL import Image
from pytorch_pretrained_bert.tokenization import BertTokenizer
from transformers import RobertaTokenizer
from utils.word_utils import Corpus


def read_examples(input_line, unique_id):
    """Read a list of `InputExample`s from an input file."""
    examples = []
    # unique_id = 0
    line = input_line  # reader.readline()
    # if not line:
    #     break
    line = line.strip()
    text_a = None
    text_b = None
    m = re.match(r"^(.*) \|\|\| (.*)$", line)
    if m is None:
        text_a = line
    else:
        text_a = m.group(1)
        text_b = m.group(2)
    examples.append(
        InputExample(unique_id=unique_id, text_a=text_a, text_b=text_b))
    # unique_id += 1
    return examples


## Bert text encoding
class InputExample(object):
    def __init__(self, unique_id, text_a, text_b):
        self.unique_id = unique_id
        self.text_a = text_a
        self.text_b = text_b


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids):
        self.unique_id = unique_id
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def convert_examples_to_features(examples, seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""
    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > seq_length - 2:
                tokens_a = tokens_a[0:(seq_length - 2)]
        tokens = []
        input_type_ids = []
        tokens.append("[CLS]")
        # tokens.append("<s>")
        input_type_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            input_type_ids.append(0)
        tokens.append("[SEP]")
        # tokens.append("</s>")
        input_type_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                input_type_ids.append(1)
            tokens.append("[SEP]")
            input_type_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < seq_length:
            input_ids.append(0)
            input_mask.append(0)
            input_type_ids.append(0)

        assert len(input_ids) == seq_length
        assert len(input_mask) == seq_length
        assert len(input_type_ids) == seq_length
        features.append(
            InputFeatures(
                unique_id=example.unique_id,
                tokens=tokens,
                input_ids=input_ids,
                input_mask=input_mask,
                input_type_ids=input_type_ids))
    return features


class DatasetNotFoundError(Exception):
    pass

from transformers import AutoProcessor

class ZModDataset(data.Dataset):
    SUPPORTED_DATASETS = {
        'referit': {'splits': ('train', 'val', 'trainval', 'test')},
        'unc': {
            'splits': ('train', 'val', 'trainval', 'testA', 'testB'),
            'params': {'dataset': 'refcoco', 'split_by': 'unc'}
        },
        'unc+': {
            'splits': ('train', 'val', 'trainval', 'testA', 'testB'),
            'params': {'dataset': 'refcoco+', 'split_by': 'unc'}
        },
        'gref': {
            'splits': ('train', 'val'),
            'params': {'dataset': 'refcocog', 'split_by': 'google'}
        },
        'gref_umd': {
            'splits': ('train', 'val', 'test', 'adv', 'easy', 'hard'),
            'params': {'dataset': 'refcocog', 'split_by': 'umd'}
        },
        'flickr': {
            'splits': ('train', 'val', 'test')}
    }

    def __init__(self, data_root, split_root='data', dataset='referit',
                 transform=None, return_idx=False, testmode=False,
                 split='train', max_query_len=128,
                 bert_model='bert-base-uncased'):
        self.images = []
        self.data_root = data_root
        self.split_root = split_root
        self.dataset = dataset
        self.query_len = max_query_len
        self.transform = transform
        self.testmode = testmode
        self.split = split
        # self.tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=True)
        self.tokenizer = BertTokenizer.from_pretrained('./bert/bert-base-uncased-vocab.txt', do_lower_case=True)
        # self.tokenizer = RobertaTokenizer.from_pretrained('./checkpoints/roberta-base', do_lower_case=True)
        self.return_idx = return_idx

        assert self.transform is not None

        if split == 'train':
            self.augment = True
        else:
            self.augment = False

        if self.dataset == 'referit':
            self.dataset_root = osp.join(self.data_root, 'referit')
            self.im_dir = osp.join(self.dataset_root, 'images')
            self.split_dir = osp.join(self.dataset_root, 'splits')
        elif self.dataset == 'flickr':
            self.dataset_root = osp.join(self.data_root, 'Flickr30k')
            self.im_dir = osp.join(self.dataset_root, 'flickr30k_images')
        else:  ## refcoco, etc.
            self.dataset_root = osp.join(self.data_root, 'other')
            self.im_dir = osp.join(
                self.dataset_root, 'images', 'mscoco', 'images', 'train2014')
            self.split_dir = osp.join(self.dataset_root, 'splits')

        if not self.exists_dataset():
            # self.process_dataset()
            print('Please download index cache to data folder: \n \
                https://drive.google.com/open?id=1cZI562MABLtAzM6YU4WmKPFFguuVr0lZ')
            exit(0)

        dataset_path = osp.join(self.split_root, self.dataset)
        valid_splits = self.SUPPORTED_DATASETS[self.dataset]['splits']

        if split not in valid_splits:
            raise ValueError(
                'Dataset {0} does not have split {1}'.format(
                    self.dataset, split))

        splits = [split]
        if self.dataset != 'referit':
            splits = ['train', 'val'] if split == 'trainval' else [split]
        for split in splits:
            imgset_file = '{0}_{1}.pth'.format(self.dataset, split)
            # imgset_path = osp.join(dataset_path, "new_" + imgset_file)
            imgset_path = osp.join(dataset_path, imgset_file)
            self.images += torch.load(imgset_path)

    def exists_dataset(self):
        return osp.exists(osp.join(self.split_root, self.dataset))

    def pull_item(self, idx):
        if self.dataset == 'flickr':
            try:
                img_file, bbox, phrase, r1_word, llm_r1 = self.images[idx]
            except:
                img_file, bbox, phrase, r1_word = self.images[idx]
                llm_r1 = None
        else:
            try:
                img_file, _, bbox, phrase, attri, _, llm_r1 = self.images[idx]
            except:
                try:
                    img_file, _, bbox, phrase, attri, _ = self.images[idx]
                    llm_r1 = None
                except:
                    img_file, _, bbox, phrase, attri = self.images[idx]
                    llm_r1 = None
            r1_word = attri[0][1][0]
        ## box format: to x1y1x2y2
        if not (self.dataset == 'referit' or self.dataset == 'flickr'):
            bbox = np.array(bbox, dtype=int)
            bbox[2], bbox[3] = bbox[0] + bbox[2], bbox[1] + bbox[3]
        else:
            bbox = np.array(bbox, dtype=int)

        img_path = osp.join(self.im_dir, img_file)
        img = Image.open(img_path).convert("RGB")
        # img = cv2.imread(img_path)
        # ## duplicate channel if gray image
        # if img.shape[-1] > 1:
        #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # else:
        #     img = np.stack([img] * 3)

        bbox = torch.tensor(bbox)
        bbox = bbox.float()
        return img, phrase, bbox, r1_word, llm_r1


    def remove_punctuation(self, input_string):
        return re.sub(r'[^\w\s]', '', input_string)

    # # 示例
    # print(remove_punctuation("Hello, world! How's it going?"))  # 输出: "Hello world Hows it going"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img, phrase, bbox, r1_word, llm_r1 = self.pull_item(idx)
        # phrase = phrase.decode("utf-8").encode().lower()
        phrase = phrase.lower()
        r1_word = r1_word.lower()
        r1_backup_word = r1_word

        # llm_r1 = llm_r1[:-1].lower() if llm_r1.endswith(".") else llm_r1.lower()
        # llm_r1 = self.remove_punctuation(llm_r1)
        # if llm_r1 in phrase:
        #     if r1_word in llm_r1:
        #         pass
        #     else:
        #         r1_word = llm_r1

        # r1_word = llm_r1 if llm_r1 in phrase else r1_word

        input_dict = {'img': img, 'box': bbox, 'text': phrase, 'subj': r1_word}
        input_dict = self.transform(input_dict)
        img = input_dict['img']
        bbox = input_dict['box']

        pos_box = self.jitter_bbox(bbox.unsqueeze(0), 0, 0.03)
        neg_box = self.jitter_bbox(bbox.unsqueeze(0), 0, 0.15)

        phrase = input_dict['text']
        img_mask = input_dict['mask']
        r1_word = input_dict['subj']
        # r1_mask = np.zeros(self.query_len, dtype=bool)
        ## encode phrase to bert input
        examples = read_examples(phrase, idx)
        features = convert_examples_to_features(
            examples=examples, seq_length=self.query_len, tokenizer=self.tokenizer)
        word_id = features[0].input_ids
        word_mask = features[0].input_mask
        r1_mask = [1] * len(word_mask)
        try:
            find = 0
            if r1_word is not 'none':
                r1_word_ = self.tokenizer.tokenize(r1_word)
                r1_id = self.tokenizer.convert_tokens_to_ids(r1_word_)
                l = len(r1_id)
                for i, _ in enumerate(word_id[: -l]):
                    if r1_id == word_id[i: i + l]:
                        find = 1
                        r1_mask[i: i + l] = [0] * l
                        break
                if find == 0:
                    r1_word_ = self.tokenizer.tokenize(" " + r1_word)
                    r1_id = self.tokenizer.convert_tokens_to_ids(r1_word_)
                    l = len(r1_id)
                    for i, _ in enumerate(word_id[: -l]):
                        if r1_id == word_id[i: i + l]:
                            r1_mask[i: i + l] = [0] * l
                            break
        except:
            print(r1_word)
        x1, y1, x2, y2 = int(20 * (bbox[0] - bbox[2] / 2)), \
                         int(20 * (bbox[1] - bbox[3] / 2)), \
                         int(20 * (bbox[0] + bbox[2] / 2)), \
                         int(20 * (bbox[1] + bbox[3] / 2))

        bbox_mask = torch.zeros([20, 20])
        bbox_mask[x1: x2, y1: y2] += 1


        return np.array(img, dtype=np.float32), \
               np.array(img_mask, dtype=np.int16), \
               np.array(word_id, dtype=int),\
               np.array(word_mask, dtype=np.int16), \
               np.array(bbox, dtype=np.float32), \
               np.array(r1_mask, dtype=int), \
               bbox_mask.numpy(),\
               np.array(pos_box[0], dtype=np.float32), \
               np.array(neg_box[0], dtype=np.float32)

    def jitter_bbox(self, bboxes, min_range=0., max_range=0.05, truncation=True):
        """
        Jitter the bbox.
        """
        n = bboxes.size(0)
        h = bboxes[:, 2] - bboxes[:, 0]
        w = bboxes[:, 3] - bboxes[:, 1]
        noise = torch.stack([h, w, h, w], dim=-1)
        if min_range == 0:
            noise_rate = torch.normal(0, max_range / 2., size=(n, 4))
        else:
            noise_rate1 = torch.rand((n, 4)) * (max_range - min_range) + min_range
            noise_rate2 = -torch.rand((n, 4)) * (max_range - min_range) - min_range
            selector = (torch.rand((n, 4)) < 0.5).float()
            noise_rate = noise_rate1 * selector + noise_rate2 * (1. - selector)
        # new_bboxes = bboxes + noise * noise_rate
        # iou = bbox_iou(new_bboxes, bboxes)
        bboxes = bboxes + noise * noise_rate
        if truncation:
            bboxes = torch.clamp(bboxes, 0.0, 1.0)

        return bboxes