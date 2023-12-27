"""
Linear Transformer proposed in "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention"
Modified from: https://github.com/idiap/fast-transformers/blob/master/fast_transformers/attention/linear_attention.py
"""
from einops import rearrange

import torch
import torch.nn as nn
from torch.nn import Module, Dropout
import matplotlib.pyplot as plt


def elu_feature_map(x):
    return torch.nn.functional.elu(x) + 1


class LinearAttention(Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.feature_map = elu_feature_map
        self.eps = eps

    def forward(self, queries, keys, values, q_mask=None, kv_mask=None):
        """ Multi-Head linear attention proposed in "Transformers are RNNs"
        Args:
            queries: [N, L, H, D]
            keys: [N, S, H, D]
            values: [N, S, H, D]
            q_mask: [N, L]
            kv_mask: [N, S]
        Returns:
            queried_values: (N, L, H, D)
        """
        Q = self.feature_map(queries)
        K = self.feature_map(keys)

        # set padded position to zero
        if q_mask is not None:
            Q = Q * q_mask[:, :, None, None]
        if kv_mask is not None:
            K = K * kv_mask[:, :, None, None]
            values = values * kv_mask[:, :, None, None]

        v_length = values.size(1)
        values = values / v_length  # prevent fp16 overflow
        KV = torch.einsum("nshd,nshv->nhdv", K, values)  # (S,D)' @ S,V
        Z = 1 / (torch.einsum("nlhd,nhd->nlh", Q, K.sum(dim=1)) + self.eps)
        queried_values = torch.einsum("nlhd,nhdv,nlh->nlhv", Q, KV, Z) * v_length

        return queried_values.contiguous()


class FullAttention(Module):
    def __init__(self, use_dropout=False, attention_dropout=0.1):
        super().__init__()
        self.use_dropout = use_dropout
        self.dropout = Dropout(attention_dropout)

    def forward(self, queries, keys, values, q_mask=None, kv_mask=None):
        """ Multi-head scaled dot-product attention, a.k.a full attention.
        Args:
            queries: [N, L, H, D]
            keys: [N, S, H, D]
            values: [N, S, H, D]
            q_mask: [N, L]
            kv_mask: [N, S]
        Returns:
            queried_values: (N, L, H, D)
        """

        # Compute the unnormalized attention and apply the masks
        QK = torch.einsum("nlhd,nshd->nlsh", queries, keys)
        if kv_mask is not None:
            QK.masked_fill_(~(q_mask[:, :, None, None] * kv_mask[:, None, :, None]), float('-inf'))

        # Compute the attention and the weighted average
        softmax_temp = 1. / queries.size(3)**.5  # sqrt(D)
        A = torch.softmax(softmax_temp * QK, dim=2)
        if self.use_dropout:
            A = self.dropout(A)

        queried_values = torch.einsum("nlsh,nshd->nlhd", A, values)

        return queried_values.contiguous()


class MaskLinearAttention(Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim

        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.attns = nn.ParameterList([torch.nn.Parameter(torch.tensor([1.0]), requires_grad=True) for _ in range(4)])
        self.row_conv = nn.Conv1d(self.dim, 1, kernel_size=1, padding=0, stride=1)
        self.sig1 = nn.Sigmoid()
        self.col_conv = nn.Conv1d(self.dim, 1, kernel_size=1, padding=0, stride=1)
        self.sig2 = nn.Sigmoid()

    def forward(self, q, k, v, q_mask=None, kv_mask=None):
        """ Multi-Head linear attention proposed in "Transformers are RNNs"
        Args:
            queries: [N, L, H, D]
            keys: [N, S, H, D]
            values: [N, S, H, D]
            q_mask: [N, L]
            kv_mask: [N, S]
        Returns:
            queried_values: (N, L, H, D)
        """

        b, N, head = q.shape[0], q.shape[1], q.shape[2]
        if N == 4800:
            h, w = 60, 80
        else:
            h, w = int(N ** 0.5), int(N ** 0.5)

        # set padded position to zero
        if q_mask is not None:
            q = q * q_mask[:, :, None, None]
        if kv_mask is not None:
            k = k * kv_mask[:, :, None, None]
            v = v * kv_mask[:, :, None, None]
        
        q, k, v = q.permute(0, 2, -1, 1), k.permute(0, 2, -1, 1), v.permute(0, 2, -1, 1)    # [B H C N]
        q = torch.nn.functional.normalize(q, dim=-1) 
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = q @ k.transpose(-2, -1)
        B, H, C1, C2 = attn.shape
        # plt.imshow(attn[0].sum(dim=0).detach().cpu().numpy())
        # plt.show()
        w_col = self.sig1(self.col_conv(attn.reshape(-1, C1, C2))).reshape(B,H,1,C2)
        w_row = self.sig2(self.row_conv(attn.permute(0,1,-1,2).reshape(-1, C2, C1))).permute(0,-1,1).reshape(B,H,C1,1)
        attn = ((attn * w_col) + (attn * w_row)) * self.temperature
        # plt.imshow(attn[0].sum(dim=0).detach().cpu().numpy())
        # plt.show()
        # exit()
        _, _, C, _ = q.shape

        attn_masks = []
        for top_k in [int(C/2), int(C*2/3), int(C*3/4), int(C*4/5)]:
            mask = torch.zeros(b, self.num_heads, C, C, device=q.device, requires_grad=False)
            index = torch.topk(attn, k=top_k, dim=-1, largest=True)[1]
            mask.scatter_(-1, index, 1.)
            attn_mask = torch.where(mask > 0, attn, torch.full_like(attn, float('-inf')))
            attn_mask = attn_mask.softmax(dim=-1)
            attn_masked_v = attn_mask @ v
            attn_masks.append(attn_masked_v)

        out = sum([attn_mask * attn for (attn_mask, attn) in zip(attn_masks, self.attns)])
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=head, h=h, w=w) 
        B, C, H, W = out.shape
        out = out.reshape(B, C, H*W).permute(0, -1, 1).reshape(B, H*W, head, -1)
        return  out