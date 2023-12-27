import copy
import math
import numbers
import torch
import torch.nn as nn
from torch.nn import Module, Dropout
from .linear_attention import LinearAttention, MaskLinearAttention, FullAttention
import torch.nn.functional as F
import numpy as np

def drop_path(x, drop_prob=0., training=False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    # handle tensors with different dimensions, not just 4D tensors.
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(
        shape, dtype=x.dtype, device=x.device)
    output = x.div(keep_prob) * random_tensor.floor()
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.1):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Position_Encoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.feature_dim = dim

    @staticmethod
    def embed_rotary(x, cos, sin):
        x2 = torch.stack([-x[..., 1::2], x[..., ::2]], dim=-1).reshape_as(x).contiguous()
        x = x * cos + x2 * sin
        return x

    @staticmethod
    def embed_pos(x, pe):
        """ combine feature and position code
		"""
        return Position_Encoding.embed_rotary(x, pe[..., 0], pe[..., 1])

    def forward(self, feature):
        bsize, npoint, _ = feature.shape
        position = torch.arange(npoint, device=feature.device).unsqueeze(dim=0).repeat(bsize, 1).unsqueeze(dim=-1)
        # [1, 1, d/2]
        div_term = torch.exp(torch.arange(0, self.feature_dim, 2, dtype=torch.float, device=feature.device) * (
                -math.log(10000.0) / self.feature_dim)).view(1, 1, -1)
        sinx = torch.sin(position * div_term)  # [B, N, d//2]
        cosx = torch.cos(position * div_term)
        sinx, cosx = map(lambda feat: torch.stack([feat, feat], dim=-1).view(bsize, npoint, -1), [sinx, cosx])
        position_code = torch.stack([cosx, sinx], dim=-1)
        if position_code.requires_grad:
            position_code = position_code.detach()
        return position_code


class rewrite_linear(nn.Linear):
    def __init__(self, in_features, out_features, bias = True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.register_parameter("cross_weight", nn.Parameter(self.weight.clone()))
        if self.bias is not None:
            self.register_parameter("cross_bias", nn.Parameter(self.bias.clone()))

    def forward(self, input, score):
        weight = score * self.weight + (1 - score) * self.cross_weight
        if self.bias is not None:
            bias = score * self.bias + (1 - score) * self.cross_bias
        else:
            bias = None
        return F.linear(input, weight, bias)


class rewrite_LayNorm2d(nn.LayerNorm):
    def __init__(self, normalized_shape, eps = 0.00001, elementwise_affine = True, device=None, dtype=None) -> None:
        super().__init__(normalized_shape, eps, elementwise_affine, device, dtype)
        self.register_parameter('specific_weight', nn.Parameter(self.weight.clone()))
        if self.bias is not None:
            self.register_parameter('specific_bias', nn.Parameter(self.bias.clone()))
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)  # type: ignore[assignment]
        self.normalized_shape = tuple(normalized_shape)  # type: ignore[arg-type]

    def forward(self, input, score):
        weight = score * self.weight + (1 - score) * self.specific_weight
        if self.bias is not None:
            bias = score * self.bias + (1 - score) * self.specific_bias
        else:
            bias = None
        return F.layer_norm(input, self.normalized_shape, weight, bias)

    
class rewrite_conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, 
                 dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, 
                         dilation, groups, bias, padding_mode, device, dtype)
        self.register_parameter("cross_weight", nn.Parameter(self.weight.clone()))
        if self.bias is not None:
            self.register_parameter("cross_bias", nn.Parameter(self.bias.clone()))
    
    def forward(self, input, score):
        weight = score * self.weight + (1 - score) * self.cross_weight
        if self.bias is not None:
            bias = score * self.bias + (1 - score) * self.cross_bias
        else:
            bias = None
        return self._conv_forward(input, weight, bias)


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = rewrite_conv2d(in_channels = dim, out_channels = dim, 
                                     kernel_size = 3, stride = 1, padding = 1, 
                                     bias = True, groups = dim, dilation = 1)
    def forward(self, x, score):
        B, N, C = x.shape
        if N == 4800:
            H, W = 60, 80
        else:
            H, W = int(N ** 0.5), int(N ** 0.5)
        # seq -> img
        # ablate
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x, score)
        x = x.flatten(2).transpose(1, 2)
        return x

class DSAPEncoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 attention='linear',
                 isdw=False):
        super(DSAPEncoderLayer, self).__init__()

        self.dim = d_model // nhead
        self.nhead = nhead
        self.isdw = isdw

        # multi-head attention
        self.q_proj = rewrite_linear(d_model, d_model, bias=False)
        self.k_proj = rewrite_linear(d_model, d_model, bias=False)
        self.v_proj = rewrite_linear(d_model, d_model, bias=False)
        self.attention = MaskLinearAttention(self.dim, nhead) if attention == 'linear' else FullAttention()
        self.merge = rewrite_linear(d_model, d_model, bias=False)

        if self.isdw:
            self.mlp1 = rewrite_linear(d_model*2, d_model*4, bias=False)
            self.dw_ffn = DWConv(d_model*4)
            self.gelu = nn.GELU()
            self.mlp2 = rewrite_linear(d_model*4, d_model, bias=False)
        else:
            # feed-forward network
            self.mlp1 = rewrite_linear(d_model*2, d_model*4, bias=False)
            self.gelu = nn.GELU()
            self.mlp2 = rewrite_linear(d_model*4, d_model, bias=False)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.score = nn.Parameter(torch.rand(1))
        self.register_buffer("previous_score", torch.zeros(1))
        self.register_buffer("score_weight", torch.tensor([0.9]))
        self.register_buffer("current_iter", torch.zeros(1))

        self.drop_path = DropPath(0.1)

    def forward(self, x, source, x_mask=None, source_mask=None, data=None):
        """
        Args:
            x (torch.Tensor): [N, L, C]
            source (torch.Tensor): [N, S, C]
            x_mask (torch.Tensor): [N, L] (optional)
            source_mask (torch.Tensor): [N, S] (optional)
        """
        bs = x.size(0)
        if self.current_iter == 0:
            self.current_iter = self.current_iter + 1
            self.previous_score = torch.tensor([self.score.item()]).cuda()
            score = self.score
        else:
            score = self.score_weight * self.previous_score + (1 - self.score_weight) * self.score
            self.previous_score = torch.tensor([score.item()]).cuda()
            self.current_iter = self.current_iter + 1
        score = binarizer_fn(score)
        
        source = score * x + (1 - score) * source
        if source_mask is not None:
            source_mask = score * x_mask + (1 - score) * source_mask
        query, key, value = x, source, source
        
        if self.isdw:
            query = self.q_proj(query, score).view(bs, -1, self.nhead, self.dim)
            key = self.k_proj(key, score).view(bs, -1, self.nhead, self.dim)
        else:
            query = self.q_proj(query, score).view(bs, -1, self.nhead, self.dim)  # [N, L, (H, D)]
            key = self.k_proj(key, score).view(bs, -1, self.nhead, self.dim)  # [N, S, (H, D)]

        value = self.v_proj(value, score).view(bs, -1, self.nhead, self.dim)
        if data is not None:
            data.update({
                "q_tensor": query,
                "k_tensor": key,
                "v_tensor": value
            })

        message = self.attention(query, key, value, q_mask=x_mask, kv_mask=source_mask)  # [N, L, (H, D)]
        message = self.merge(message.view(bs, -1, self.nhead*self.dim), score)  # [N, L, C]
        message = self.norm1(message)

        # feed-forward network
        message = self.mlp1(torch.cat([x, message], dim=2), score)
        if self.isdw:
            message = self.dw_ffn(message,  score)
        message = self.gelu(message)
        message = self.mlp2(message, score)
        message = self.drop_path(self.norm2(message))

        return x + message, self.score


# -----------------------------------------------------------------------Fine--------------------------------------------------------------------------
class LocalFeatureTransformer(nn.Module):
    def __init__(self, config):
        super(LocalFeatureTransformer, self).__init__()

        self.config = config
        self.d_model = config['d_model']
        self.nhead = config['nhead']
        self.layer_names = config['layer_names']
        self.isdw = config['isdw']
        
        self.layers = nn.ModuleList([copy.deepcopy(DSAPEncoderLayer(config['d_model'], config['nhead'], config['attention'], self.isdw[_])) for _ in range(len(self.layer_names))])
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feat0, feat1, mask0=None, mask1=None, data=None):
        """
        Args:
            feat0 (torch.Tensor): [N, L, C]
            feat1 (torch.Tensor): [N, S, C]
            mask0 (torch.Tensor): [N, L] (optional)
            mask1 (torch.Tensor): [N, S] (optional)
        """
        assert self.d_model == feat0.size(2), "the feature number of src and transformer must be equal"
        for layer, name in zip(self.layers, self.layer_names):
            feat0, score = layer(feat0, feat1, mask0, mask1, data)
            feat1, score = layer(feat1, feat0, mask1, mask0, data)
        return feat0, feat1

threshold=0.5
class BinarizerFn(torch.autograd.Function):
    """Binarizes {0, 1} a real valued tensor."""

    @staticmethod
    def forward(ctx, inputs):
        outputs = inputs.clone()
        outputs[inputs.le(threshold)] = 1
        outputs[inputs.gt(threshold)] = 0
        return outputs

    @staticmethod
    def backward(self, gradOutput):
        return gradOutput, None

binarizer_fn = BinarizerFn.apply


class DSAPEncoderLayer_Fine(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 attention='linear',
                 isdw=False):
        super(DSAPEncoderLayer_Fine, self).__init__()

        self.dim = d_model // nhead
        self.nhead = nhead
        self.isdw = isdw

        # multi-head attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.attention = MaskLinearAttention(self.dim, nhead) if attention == 'linear' else FullAttention()
        self.merge = nn.Linear(d_model, d_model, bias=False)

        # feed-forward network
        self.mlp1 = nn.Linear(d_model*2, d_model*4, bias=False)
        self.gelu = nn.GELU()
        self.mlp2 = nn.Linear(d_model*4, d_model, bias=False)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop_path = DropPath(0.1)

    def forward(self, x, source, x_mask=None, source_mask=None):
        bs = x.size(0)
        query, key, value = x, source, source
        
        if self.isdw:
            query = self.q_proj(self.lu(query)).view(bs, -1, self.nhead, self.dim)
            key = self.k_proj(key).view(bs, -1, self.nhead, self.dim)
        else:
            # multi-head attention
            query = self.q_proj(query).view(bs, -1, self.nhead, self.dim)  # [N, L, (H, D)]
            key = self.k_proj(key).view(bs, -1, self.nhead, self.dim)  # [N, S, (H, D)]

        value = self.v_proj(value).view(bs, -1, self.nhead, self.dim)
        message = self.attention(query, key, value, q_mask=x_mask, kv_mask=source_mask)  # [N, L, (H, D)]
        message = self.merge(message.view(bs, -1, self.nhead*self.dim))  # [N, L, C]
        message = self.norm1(message)

        # feed-forward network
        message = self.mlp1(torch.cat([x, message], dim=2))
        message = self.gelu(message)
        message = self.mlp2(message)
        message = self.drop_path(self.norm2(message))

        return x + message


class LocalFeatureTransformer_Fine(nn.Module):
    def __init__(self, config):
        super(LocalFeatureTransformer_Fine, self).__init__()

        self.config = config
        self.d_model = config['d_model']
        self.nhead = config['nhead']
        self.layer_names = config['layer_names']

        encoder_layer = DSAPEncoderLayer_Fine(config['d_model'], config['nhead'], config['attention'])
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(len(self.layer_names))])
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feat0, feat1, mask0=None, mask1=None):
        assert self.d_model == feat0.size(2), "the feature number of src and transformer must be equal"
        for layer, name in zip(self.layers, self.layer_names):
            if name == 'self':
                feat0 = layer(feat0, feat0, mask0, mask0)
                feat1 = layer(feat1, feat1, mask1, mask1)
            elif name == 'cross':
                feat0 = layer(feat0, feat1, mask0, mask1)
                feat1 = layer(feat1, feat0, mask1, mask0)
            else:
                raise KeyError

        return feat0, feat1