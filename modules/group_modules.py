# Copyright (c) 2019, NAVER CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import math
import functools

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.proj_adaptive_softmax import ProjectedAdaptiveLogSoftmax
from utils.log_uniform_sampler import LogUniformSampler, sample_logits

from modules.positionwiseff import *
from modules.attention import *


def channel_shuffle(input, groups):
    length, batch, channel = input.shape
    channel_per_group = channel // groups
    out = input.view(length, batch, groups, channel_per_group)
    out = out.transpose(2, 3).contiguous().view(length, batch, channel)
    return out


class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()

        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, input):
        # input : qlen x bsz x d_model
        mean = input.mean(-1, keepdim=True)
        std = input.std(-1, keepdim=True)

        return self.gamma * (input - mean) / (std + self.eps) + self.beta


class GroupHeadedLayerNorm(nn.Module):
    def __init__(self, n_group, d_model, eps=1e-6):
        super().__init__()

        self.d_model = d_model
        self.n_group = n_group
        self.d_group = self.d_model//self.n_group

        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, input):
        # input : qlen x bsz x d_model
        qlen, bsz, d_model = input.shape
        groupHeadedInput = input.view(qlen, bsz, self.n_group, self.d_group)
        mean = groupHeadedInput.mean(-1, keepdim=True)
        std = groupHeadedInput.std(-1, keepdim=True)
        normalized_groupHeadedInput = ( groupHeadedInput - mean ) / (std + self.eps)
        out = normalized_groupHeadedInput.view(qlen, bsz, self.d_model)
        return self.gamma * out + self.beta


class LearnablePositionalEmbedding(nn.Module):
    def __init__(self, demb, max_len=1024):
        super().__init__()

        self.demb = demb

        pos_seq = torch.arange(max_len-1, -1, -1.0)
        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        sinusoid = torch.ger(pos_seq, inv_freq) # max_len x (demb/2)
        pos_enc = torch.cat([sinusoid.sin(), sinusoid.cos()], dim=-1) # max_len x demb

        self.pos_enc = nn.Parameter(pos_enc) # max_len x demb

    def forward(self, klen):
        return self.pos_enc[-klen:,None,:]


class GroupAttention(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, n_group, dropatt=0.0,
                 tgt_len=None, ext_len=None, mem_len=None, split=False):
        super().__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout

        self.scale = 1 / (d_head ** 0.5)
        self.n_group = n_group
        self.d_group = d_model // n_group
        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)

        self.q_norm = GroupHeadedLayerNorm(self.n_group, self.d_model)
        self.q_net = nn.Conv1d(self.d_model, self.d_model, kernel_size=1, groups=self.n_group, bias=False)
        self.inter_q_net = nn.Conv1d(self.d_model, self.d_group, kernel_size=1, groups=1, bias=False)

        self.kv_norm = LayerNorm(self.d_model)
        self.k_net = nn.Conv1d(self.d_model, self.d_model, kernel_size=1, groups=1, bias=False)
        self.v_net = nn.Conv1d(self.d_model, self.d_model, kernel_size=1, groups=1, bias=False)

        self.r_net = nn.Conv1d(self.d_model, self.n_head * self.d_head, kernel_size=1, groups=self.n_group, bias=False)

        self.intra_linear = nn.Conv1d(self.d_model, self.d_model, kernel_size=1, groups=self.n_group, bias=False)
        self.inter_linear = nn.Linear(self.d_model, self.d_group, bias=False)

    def _rel_shift(self, x, zero_triu=False):
        zero_pad = torch.zeros((x.size(0), 1, *x.size()[2:]),
                               device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=1)

        x_padded = x_padded.view(x.size(1) + 1, x.size(0), *x.size()[2:])

        x = x_padded[1:].view_as(x)

        if zero_triu:
            ones = torch.ones((x.size(0), x.size(1)))
            x = x * torch.tril(ones, x.size(1) - x.size(0))[:,:,None,None]

        return x

    def forward(self, w, r, r_w_bias, r_r_bias, attn_mask=None, mems=None):
        """
        w : qlen x bsz x d_embed (=d_model)
        r : klen x 1 x d_model
        r_w_bias : n_head x d_head
        r_r_bias : n_head x d_head
        attn_mask : qlen x klen x 1
        mems : ...
        """

        qlen, rlen, bsz = w.size(0), r.size(0), w.size(1)

        if mems is not None:
            w_norm = self.q_norm(w) # klen x bsz x d_model
            kv_norm = self.kv_norm(torch.cat([mems, w], 0))
        else:
            w_norm = self.q_norm(w) # qlen x bsz x d_model
            kv_norm = self.kv_norm(w)

        w_norm = w_norm.permute(1,2,0) # bsz x d_embed x klen
        kv_norm = kv_norm.permute(1,2,0)

        #### intra & inter group operation for queries
        w_head_q = self.q_net(w_norm).permute(2,0,1)
        w_head_q = w_head_q.view(-1, bsz, self.n_group, self.d_group)
        w_head_q_global = self.inter_q_net(w_norm).permute(2,0,1).view(-1, bsz, 1 , self.d_group)
        w_head_q = w_head_q + w_head_q_global
        w_head_q = w_head_q.view(-1, bsz, self.n_head, self.d_head)

        #### keys & values
        w_head_k = self.k_net(kv_norm).permute(2,0,1).view(-1, bsz, self.n_head, self.d_head)  # klen x bsz x n_head x d_head
        w_head_v = self.v_net(kv_norm).permute(2,0,1).view(-1, bsz, self.n_head, self.d_head)  # klen x bsz x n_head x d_head
        
        #### relative position
        r = r.permute(1,2,0) # 1 x d_model x klen
        r_head_k = self.r_net(r) # 1 x d_model x klen
        r_head_k = r_head_k.permute(2,0,1).view(rlen, self.n_head, self.d_head) # klen x n_head x d_head

        #### compute attention score
        rw_head_q = w_head_q + r_w_bias                                         # qlen x bsz x n_head x d_head
        AC = torch.einsum('ibnd,jbnd->ijbn', (rw_head_q, w_head_k))             # qlen x klen x bsz x n_head
        rr_head_q = w_head_q + r_r_bias
        BD = torch.einsum('ibnd,jnd->ijbn', (rr_head_q, r_head_k))              # qlen x klen x bsz x n_head
        BD = self._rel_shift(BD)
        attn_score = AC + BD
        attn_score.mul_(self.scale)

        #### compute attention probability
        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                attn_score = attn_score.float().masked_fill(
                    attn_mask[None,:,:,None], -float('inf')).type_as(attn_score)
            elif attn_mask.dim() == 3:
                attn_score = attn_score.float().masked_fill(
                    attn_mask[:,:,:,None], -float('inf')).type_as(attn_score)

        attn_prob = F.softmax(attn_score, dim=1) # qlen x klen x bsz x n_head
        attn_prob = self.dropatt(attn_prob)

        #### compute attention vector
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, w_head_v)) # qlen x bsz x n_head x d_head

        attn_out = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head) # qlen x bsz x d_model (=n_head x d_head)

        #### inter & intra-group operation
        attn_inter_out = self.inter_linear(attn_out) # qlen x bsz x d_head
        attn_out = self.intra_linear(attn_out.permute(1,2,0))
        attn_out = attn_out.permute(2,0,1) # qlen x bsz x d_embed

        attn_out = attn_out.view(qlen, bsz, self.n_group, self.d_group) + attn_inter_out.view(qlen, bsz, 1 , self.d_group)
        attn_out = attn_out.view(qlen, bsz, self.n_group*self.d_group)

        attn_out = self.drop(attn_out) # qlen x bsz x d_model

        ##### residual connection
        attn_out = w + attn_out # qlen x bsz x d_model

        return attn_out # qlen x bsz x d_model


class GroupFeedforward(nn.Module):
    def __init__(self, d_model, d_inner, dropout, n_group):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_inner
        self.n_group = n_group
        self.d_group = d_model//n_group

        self.inter_linear = nn.Conv1d(d_model, d_model, kernel_size=1, groups=n_group)

        self.intra_linear_1 = nn.Sequential(
            nn.Conv1d(d_model*2, d_inner, kernel_size=1, groups=n_group),
            nn.ReLU(inplace=True)
        )
        self.intra_linear_2 = nn.Conv1d(d_inner, d_model, kernel_size=1, groups=n_group)

        self.lnorm1 = GroupHeadedLayerNorm(self.n_group, self.d_model)
        self.lnorm2 = GroupHeadedLayerNorm(self.n_group*2, self.d_model*2)
        self.lnorm3 = GroupHeadedLayerNorm(self.n_group, self.d_inner)

    def forward(self, input):
        """
        input : qlen x bsz x d_model
        """
        qlen, bsz = input.size(0), input.size(1)

        #### inter-group operations
        inter_input = channel_shuffle(self.inter_linear(self.lnorm1(input).permute(1,2,0)).permute(2,0,1), self.n_group)  # qlen x bsz x d_model
        inter_input = inter_input.view(qlen, bsz, self.n_group, self.d_group) # qlen x bsz x n_head x d_head

        #### intra-group operations
        intra_input = input.view(qlen, bsz, self.n_group, self.d_group) # qlen x bsz x n_head x d_head
        out = torch.cat([intra_input, inter_input], dim=-1).view(qlen, bsz, -1) # qlen x bsz x n_head x 2*d_head ->  qlen x bsz x 2*d_model
        out = self.lnorm2(out)  # qlen x bsz x d_model
        out = self.intra_linear_1(out.permute(1,2,0)).permute(2,0,1).contiguous()
        out = self.lnorm3(out)
        out = self.intra_linear_2(out.permute(1,2,0)).permute(2,0,1).contiguous() # qlen x bsz x d_model
        
        #### residual connection
        out = input + out # qlen x bsz x d_model

        return out # qlen x bsz x d_model


class GroupTransformer(nn.Module):
    def __init__(self, n_head, d_model, d_head, d_inner, dropout, n_group, 
                 **kwargs):
        super().__init__()
        assert d_model == d_head * n_head

        self.attn = GroupAttention(
                            n_head, d_model, d_head, dropout, n_group, **kwargs)
        self.ff = GroupFeedforward(
                        d_model, d_inner, dropout, n_group)

    def forward(self, dec_inp, r, r_w_bias, r_r_bias, dec_attn_mask=None, mems=None):

        output = self.attn(
                     dec_inp, r, r_w_bias, r_r_bias,
                     attn_mask=dec_attn_mask,
                     mems=mems)
        output = self.ff(output)

        return output
