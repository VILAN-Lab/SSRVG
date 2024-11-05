# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""

import torch
import torch.nn.functional as F
from torch import nn

from utils.misc import NestedTensor

from pytorch_pretrained_bert.modeling import BertModel
from transformers import RobertaModel


class BERT(nn.Module):
    def __init__(self, name: str, train_bert: bool, hidden_dim: int, max_len: int, enc_num):
        super().__init__()
        if name == 'bert-base-uncased':
            self.num_channels = 768
        else:
            self.num_channels = 1024
        self.enc_num = enc_num

        # self.bert = BertModel.from_pretrained(name)
        self.bert = BertModel.from_pretrained('./bert/bert-base-uncased')
        # self.bert = RobertaModel.from_pretrained('./checkpoints/roberta-base')

        if not train_bert:
            for p in self.bert.parameters():
                p.requires_grad = False

    def forward(self, tensor_list: NestedTensor):

        if self.enc_num > 0:
            all_encoder_layers, _ = self.bert(tensor_list.tensors, token_type_ids=None, attention_mask=tensor_list.mask)
            # use the output of the X-th transformer encoder layers
            xs = all_encoder_layers[self.enc_num - 1]

            # xs = self.bert(tensor_list.tensors, token_type_ids=None, attention_mask=tensor_list.mask).last_hidden_state
        else:
            xs = self.bert.embeddings.word_embeddings(tensor_list.tensors)

        mask = tensor_list.mask.to(torch.bool)
        mask = ~mask
        out = NestedTensor(xs, mask)

        return out


def build_bert(args):
    # position_embedding = build_position_encoding(args)
    train_bert = args.lr_bert > 0
    bert = BERT(args.bert_model, train_bert, args.hidden_dim, args.max_query_len, args.bert_enc_num)
    # model = Joiner(bert, position_embedding)
    # model.num_channels = bert.num_channels
    return bert
