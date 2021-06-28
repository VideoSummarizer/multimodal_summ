# -*- coding: utf-8 -*-

import argparse
import math
import os
import dill
import io

from collections import OrderedDict

from tqdm import tqdm

from torchtext import data

import torch
import torch.nn as nn
import torch.optim as optim

from transformer import Models

import torch.nn as nn

from consts import global_consts as gc



def guided_attention(asr_vector, ocr_vector):   
    #### asr_vector - (batch_size, num_seq_n, 512)
    #### ocr_vector - (batch_size, num_seq_m, 512)
    ocr_vector_tran = ocr_vector.permute(0, 2, 1)
    affinity_matrix_int = torch.bmm(asr_vector, self.W_b)
    affinity_matrix = torch.bmm(affinity_matrix_int, ocr_vector_tran)

    affinity_matrix_sum = torch.sum(affinity_matrix, dim=1)
    affinity_matrix_sum = torch.unsqueeze(affinity_matrix_sum, dim=1)
    alpha_h = affinity_matrix/affinity_matrix_sum

    alpha_h_tran = alpha_h.permute(0,2,1)
    a_h = torch.bmm(alpha_h_tran, asr_vector)

    cos = nn.CosineSimilarity(dim=2, eps=1e-6)
    gates = (1 - cos(ocr_vector.cpu(), a_h.cpu())).to(device)

    gated_image_features = a_h * gates[:, :, None]     

    return gated_image_features

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        conf = gc.config
        vocab = 50000
        proj_dim_a = conf['proj_dim_a']
        proj_dim_v = conf['proj_dim_v']
        proj_dim_l = conf['proj_dim_l']
        #print(gc.dim_a, proj_dim_a,conf['n_layers'], conf['dropout'],gc.dim_l, proj_dim_a, proj_dim_v)
        self.proj_a = nn.Linear(40, proj_dim_a)
        self.proj_v = nn.Linear(2048, proj_dim_v)
        self.proj_l = guided_attention(asr_vector, ocr_vector)
        self.transformer_decoder = Models.translation_lm((proj_dim_l, proj_dim_a, proj_dim_v), conf['n_layers'], conf['dropout'])
        dim_total_proj = conf['dim_total_proj']
        dim_total_proj = 2048
        dim_total = gc.dim_l + proj_dim_a + proj_dim_v
        self.gru = nn.GRU(input_size=dim_total, hidden_size=dim_total_proj)
        self.finalW = nn.Linear(dim_total_proj, vocab)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, words, covarep, facet, inputLens):
        state = self.transformer_decoder((words, self.proj_a(covarep), self.proj_v(facet)))
        _, gru_last_h = self.gru(state.transpose(0, 1))
        return self.finalW(gru_last_h.squeeze()).squeeze()
