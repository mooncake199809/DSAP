import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def cal_conv_out_size(w, kernel_size, stride, padding):
    return (w - kernel_size + 2 * padding) // stride + 1


# 预测细匹配结果的偏移量
class FineAdjustNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.win_s2 = 8
        self.psize = 5
        self.feat_dim = 256  # 输入维度
        self.conv_strs = [1, 1]
        self.conv_dims = [512, 512]
        self.conv_kers = [1, 1]
        self.fc_in_dim = 512  # 第二部分输入维度
        self.fc_dims = [256, 256]
        self.out_dim = 2  # xy位移+不确定度

        # Build layers
        self.conv = self.make_conv_layers(self.feat_dim, self.conv_dims, self.conv_kers)
        self.fc = self.make_fc_layers(self.fc_in_dim, self.fc_dims, self.out_dim)
        self.reg = nn.Linear(256, 2)
        self.cla = nn.Linear(256, 1)

    # conv_dims：输出通道列表[256, 256]  conv_kers：卷积核尺寸列表[3, 3]
    def make_conv_layers(self, in_dim, conv_dims, conv_kers, bias=False):
        layers = []
        w = self.psize  # Initial spatial size
        for out_dim, kernel_size, stride in zip(conv_dims, conv_kers, self.conv_strs):
            layers.append(nn.Conv2d(in_dim, out_dim, kernel_size, stride=stride, padding=0, bias=bias))
            layers.append(nn.BatchNorm2d(out_dim))
            w = cal_conv_out_size(w, kernel_size, stride, 0)
            in_dim = out_dim
        layers.append(nn.ReLU())
        layers.append(nn.MaxPool2d(kernel_size=w))
        # layers.append(nn.AvgPool2d(kernel_size=w))
        return nn.Sequential(*layers)

    def make_fc_layers(self, in_dim, fc_dims, fc_out_dim):
        layers = []
        for out_dim in fc_dims:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.BatchNorm1d(out_dim)),
            layers.append(nn.ReLU())
            in_dim = out_dim

        return nn.Sequential(*layers)

    def forward(self, feat1, feat2, data):
        # feat1, feat2: shape (1024, 128, 7, 7)
        M, C, W1, W2 = feat1.shape

        # 测试时，若没有匹配点，就直接返回
        if M == 0 or M == 1:
            data.update({
                'expec_f': torch.empty(0, 3, device=feat1.device),
                'mkpts0_f': data['mkpts0_c'],
                'mkpts1_f': data['mkpts1_c'],
            })
            return 0, 0

        feat = torch.cat([feat1, feat2], dim=1)  # (512B, 256, 7, 7)
        feat = self.conv(feat)  # (512B, 512, 1, 1)
        feat = feat.view(-1, feat.shape[1])  # (512B, 512)
        out = self.fc(feat)  # (512B, 2)
        reg = self.reg(out)
        cla = self.cla(out)
        cla = torch.sigmoid(cla)
        return reg, cla  # [512B 2] [512B 1]


# 对960个点求偏移量，只对33个点进行调整，只保存33个点
class FineMatching(nn.Module):
    """FineMatching with s2d paradigm"""

    def __init__(self):
        super().__init__()

    def forward(self, data, feat_f0, feat_f1):
        M, WW, C = feat_f0.shape
        if M == 0 or M == 1:
            data.update({
                'expec_f': torch.empty(0, 3, device=feat_f0.device),
                'mkpts0_f': data['mkpts0_c'],
                'mkpts1_f': data['mkpts1_c'],
            })
            return
        bias_pred = data["bias_pred"]
        mkpts0_f = data['mkpts0_c']
        mkpts1_f = data['mkpts1_c'] + bias_pred[:len(data['mconf'])]  # 对细匹配点进行调整
        data.update({
            "mkpts0_f": mkpts0_f,
            "mkpts1_f": mkpts1_f
        })  # 最后的匹配结果


if __name__ == '__main__':
    config = {'win_s2': 9, 'fine_window_size': 7}
    data = {'mkpts0_c': 0, 'mkpts1_c': 0}
    feat1 = torch.randn(0, 128, 7, 7)
    feat2 = torch.randn(0, 128, 7, 7)
    net = FineAdjustNet(config)
    reg, cla = net(feat1, feat2, data)
    print(reg.shape)
    print(cla.shape)
