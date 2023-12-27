import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir("..")
from copy import deepcopy

from kornia.utils import create_meshgrid
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from src.utils.plotting import make_matching_figure
import time
from src.loftr import LoFTR, default_cfg
from src.utils import *
import torch

class Show_Matches():
    def __init__(self):
        super(Show_Matches, self).__init__()

    def normalize(self, x):
        max_val = torch.max(x, dim=0, keepdim=True)[0]
        min_val = torch.min(x, dim=0, keepdim=True)[0]
        x = 0.75 * (x - min_val) / (max_val - min_val) + 0.25
        return x

    def initial_set(self, img0_pth, img1_pth):
        img0_raw = cv2.imread(img0_pth, cv2.IMREAD_GRAYSCALE)
        img1_raw = cv2.imread(img1_pth, cv2.IMREAD_GRAYSCALE)
        img0_raw = cv2.resize(img0_raw, (840, 840))
        img1_raw = cv2.resize(img1_raw, (840, 840))

        img0_raw_color = cv2.imread(img0_pth, cv2.IMREAD_COLOR)
        img1_raw_color = cv2.imread(img1_pth, cv2.IMREAD_COLOR)
        img0_raw_color = cv2.resize(img0_raw_color, (840, 840))
        img1_raw_color = cv2.resize(img1_raw_color, (840, 840))

        new_image = np.ones((840, 840 * 2 + 20, 3)) * 255
        new_image[:, :840, :] = img0_raw_color
        new_image[:, 840 + 20:, :] = img1_raw_color

        img0 = torch.from_numpy(img0_raw)[None][None].cuda() / 255.
        img1 = torch.from_numpy(img1_raw)[None][None].cuda() / 255.
        batch = {'image0': img0, 'image1': img1}
        return img0_raw, img1_raw, img0_raw_color, img1_raw_color, new_image, batch

    def self_attention(self, img0_pth, img1_pth, save_path, index, max_index):
        img0_raw, img1_raw, img0_raw_color, img1_raw_color, new_image, batch = self.initial_set(img0_pth, img1_pth)
        with torch.no_grad():
            matcher(batch)

            b_seed_indx = torch.arange(1)[:, None].repeat(1, 9000)  # [2 2048]
            h0 = h1 = w0 = w1 = 105
            scale0 = scale1 = 8
            grid_pt0_c = create_meshgrid(h0, w0, False).reshape(1, h0 * w0, 2).repeat(1, 1, 1)  # [N, 4800, 2]
            grid_pt0_i = scale0 * grid_pt0_c  # 特征点的坐标
            grid_pt1_c = create_meshgrid(h1, w1, False).reshape(1, h1 * w1, 2).repeat(1, 1, 1)
            grid_pt1_i = scale1 * grid_pt1_c
            batch.update({
                "spts0_i": grid_pt0_i,  # [2 2048 2], 图1种子对应的原图坐标
                "spts1_i": grid_pt1_i  # [2 2048 2], 图2种子对应的原图坐标
            })


            mkpts0_f = batch['mkpts0_f']
            mkpts0_f[:, 0] = torch.clamp(mkpts0_f[:, 0], 0, 840)
            mkpts0_f[:, 1] = torch.clamp(mkpts0_f[:, 1], 0, 840)
            mkpts1_f = batch['mkpts1_f']
            mkpts1_f[:, 0] = torch.clamp(mkpts1_f[:, 0], 0, 840)
            mkpts1_f[:, 1] = torch.clamp(mkpts1_f[:, 1], 0, 840)
            mkpts0_f = mkpts0_f.long()
            mkpts1_f = mkpts1_f.long()
            q_tensor = batch["q_tensor"][0].flatten(1)
            k_tensor = batch["k_tensor"][0].flatten(1)
            attention_matrix0 = q_tensor @ q_tensor.T
            attention_matrix1 = k_tensor @ k_tensor.T

            b = batch["b_ids"][index]
            i = batch["i_ids"][index]
            j = batch["j_ids"][index]
            feat0_self = attention_matrix0[i, :]  # [N]
            feat1_self = attention_matrix1[j, :]
            feat0_coord = mkpts0_f[index]
            feat1_coord = mkpts1_f[index]
            feat1_coord[0] = feat1_coord[0] + 860

            print(feat0_self)
            exit()

            i_max_val, i_max_index = torch.topk(feat0_self, k=max_index)
            j_max_val, j_max_index = torch.topk(feat1_self, k=max_index)
            i_max_val = self.normalize(i_max_val)
            j_max_val = self.normalize(j_max_val)
            b, n, c = batch["spts0_i"].shape
            original0_coord = batch["spts0_i"].reshape(n, c).long()
            original1_coord = batch["spts1_i"].reshape(n, c).long()
            max_mkpts0_f = original0_coord[i_max_index]
            max_mkpts1_f = original1_coord[j_max_index]
            max_mkpts1_f[:, 0] = max_mkpts1_f[:, 0] + 860
            assert max_mkpts0_f.shape[0] == max_mkpts1_f.shape[0]

            feat0_self_0 = []
            feat0_self_1 = []
            feat1_self_0 = []
            feat1_self_1 = []
            for i in range(max_mkpts0_f.shape[0]):
                feat0_self_0.append(max_mkpts0_f[i][None])
                feat0_self_1.append(feat0_coord[None])
                feat1_self_0.append(max_mkpts1_f[i][None])
                feat1_self_1.append(feat1_coord[None])
            feat0_self_0 = torch.cat(feat0_self_0, dim=0)
            feat0_self_1 = torch.cat(feat0_self_1, dim=0)
            feat1_self_0 = torch.cat(feat1_self_0, dim=0)
            feat1_self_1 = torch.cat(feat1_self_1, dim=0)
            feat_self_all0 = torch.cat((feat0_self_0, feat1_self_0), dim=0).cpu().numpy()
            feat_self_all1 = torch.cat((feat0_self_1, feat1_self_1), dim=0).cpu().numpy()

        color0 = cm.Reds(i_max_val.cpu().numpy())  # [M 4]
        color0 = (np.array(color0)[:, :3] * 255.0)[:, ::-1].astype(int)  # [M 3]
        color1 = cm.Reds(j_max_val.cpu().numpy())  # [M 4]
        color1 = (np.array(color1)[:, :3] * 255.0)[:, ::-1].astype(int)  # [M 3]

        color = np.concatenate((color0, color1), axis=0)
        for (x0, y0), (x1, y1), c in zip(feat_self_all0, feat_self_all1, color):
            c = c.tolist()
            cv2.line(new_image, (x0, y0), (x1, y1), color=c, thickness=3)
        for (x0, y0), (x1, y1) in zip(feat_self_all0, feat_self_all1):
            cv2.circle(new_image, (x0, y0), color=(0, 255, 0), radius=5, thickness=-1)
            cv2.circle(new_image, (x1, y1), color=(0, 255, 0), radius=5, thickness=-1)
        feat0_coord = feat0_coord.long().cpu().numpy()
        feat1_coord = feat1_coord.long().cpu().numpy()
        cv2.line(new_image, (feat0_coord[0], feat0_coord[1]), (feat1_coord[0], feat1_coord[1]), color=(255, 0, 0),
                 thickness=2)
        cv2.circle(new_image, (feat0_coord[0], feat0_coord[1]), color=(0, 0, 255), radius=5, thickness=-1)
        cv2.circle(new_image, (feat1_coord[0], feat1_coord[1]), color=(0, 0, 255), radius=5, thickness=-1)
        cv2.imwrite(save_path, new_image)

    def cross_attention(self, img0_pth, img1_pth, save_path, index, max_index):
        img0_raw, img1_raw, img0_raw_color, img1_raw_color, new_image, batch = self.initial_set(img0_pth, img1_pth)
        with torch.no_grad():
            matcher(batch)

            b_seed_indx = torch.arange(1)[:, None].repeat(1, 9000)  # [2 2048]
            h0 = h1 = w0 = w1 = 105
            scale0 = scale1 = 8
            grid_pt0_c = create_meshgrid(h0, w0, False).reshape(1, h0 * w0, 2).repeat(1, 1, 1)  # [N, 4800, 2]
            grid_pt0_i = scale0 * grid_pt0_c  # 特征点的坐标
            grid_pt1_c = create_meshgrid(h1, w1, False).reshape(1, h1 * w1, 2).repeat(1, 1, 1)
            grid_pt1_i = scale1 * grid_pt1_c
            batch['spv_pt0_i'] = grid_pt0_i
            batch['spv_pt1_i'] = grid_pt1_i
            pts0_i = batch['spv_pt0_i']  # [2 4800 2]
            pts1_i = batch['spv_pt1_i']
            spts0_i = pts0_i[b_seed_indx, batch["seed0_index"]]
            spts1_i = pts1_i[b_seed_indx, batch["seed1_index"]]
            batch.update({
                "spts0_i": spts0_i,  # [2 2048 2], 图1种子对应的原图坐标
                "spts1_i": spts1_i  # [2 2048 2], 图2种子对应的原图坐标
            })

            mkpts0_f = batch['mkpts0_f']
            mkpts0_f[:, 0] = torch.clamp(mkpts0_f[:, 0], 0, 840)
            mkpts0_f[:, 1] = torch.clamp(mkpts0_f[:, 1], 0, 840)
            mkpts1_f = batch['mkpts1_f']
            mkpts1_f[:, 0] = torch.clamp(mkpts1_f[:, 0], 0, 840)
            mkpts1_f[:, 1] = torch.clamp(mkpts1_f[:, 1], 0, 840)
            mkpts0_f = mkpts0_f.long()
            mkpts1_f = mkpts1_f.long()
            feat_cross_matrix = batch["cross_matrix"]

            b = batch["b_seed_coarse"][index]
            i = batch["i_seed_coarse"][index]
            j = batch["j_seed_coarse"][index]
            feat0_self = feat_cross_matrix[b, i, :]
            feat0_coord = mkpts0_f[index]
            feat1_coord = mkpts1_f[index]

            j_max_val, j_max_index = torch.topk(feat0_self, k=max_index)
            j_max_val = self.normalize(j_max_val)
            b, n, c = batch["spts0_i"].shape
            original1_coord = batch["spts1_i"].reshape(n, c).long()
            max_mkpts1_f = original1_coord[j_max_index]
            max_mkpts1_f[:, 0] = max_mkpts1_f[:, 0] + 860

            feat1_cross = []
            feat0_cross = []
            for i in range(max_mkpts1_f.shape[0]):
                feat1_cross.append(max_mkpts1_f[i][None])
                feat0_cross.append(feat0_coord[None])
            feat1_cross = torch.cat(feat1_cross, dim=0).cpu().numpy()
            feat0_cross = torch.cat(feat0_cross, dim=0).cpu().numpy()

        color = cm.Reds(j_max_val.cpu().numpy())
        color = (np.array(color)[:, :3] * 255.0)[:, ::-1].astype(int)
        for (x0, y0), (x1, y1), c in zip(feat0_cross, feat1_cross, color):
            c = c.tolist()
            cv2.line(new_image, (x0, y0), (x1, y1), color=c, thickness=3, lineType=cv2.LINE_AA)
        for (x0, y0), (x1, y1) in zip(feat0_cross, feat1_cross):
            cv2.circle(new_image, (x0, y0), color=(0, 255, 0), radius=4, thickness=-1)
            cv2.circle(new_image, (x1, y1), color=(0, 255, 0), radius=4, thickness=-1)
        cv2.imwrite(save_path, new_image)


class get_name_class_class():
    def __init__(self, npz_path):
        super(get_name_class_class, self).__init__()
        self.root_dir = "/mnt/share/sda-8T/xt/MegaDepth/"
        self.npz_path = npz_path
        self.scene_id = self.npz_path.split('.')[0]

        min_overlap_score = 0
        self.scene_info = np.load(self.npz_path, allow_pickle=True)
        self.pair_infos = self.scene_info['pair_infos'].copy()
        del self.scene_info['pair_infos']
        self.pair_infos = [pair_info for pair_info in self.pair_infos if pair_info[1] > min_overlap_score]

        self.img_resize = 840
        self.df = None
        self.img_padding = True
        self.depth_max_size = 2000

        # for training LoFTR
        self.augment_fn = None
        self.coarse_scale = 0.125

    def get_path(self, idx):
        (idx0, idx1), overlap_score, central_matches = self.pair_infos[idx]
        img_name0 = os.path.join(self.root_dir, self.scene_info['image_paths'][idx0])
        img_name1 = os.path.join(self.root_dir, self.scene_info['image_paths'][idx1])
        return img_name0, img_name1
    
if __name__ == '__main__':
    _default_cfg = deepcopy(default_cfg)
    _default_cfg['coarse']['temp_bug_fix'] = True  # set to False when using the old ckpt
    matcher = LoFTR(config=_default_cfg)
    matcher.load_state_dict(
        torch.load(
            "/home/dk/LoFTR_NEW/zishiying/LoFTR_Auto_newattention/weights/epoch=59-auc@5=0.574-auc@10=0.729-auc@20=0.839.ckpt")
        ['state_dict'])
    matcher = matcher.eval().cuda()

    # 0015_0.1_0.3  0015_0.3_0.5  0022_0.1_0.3  0022_0.3_0.5
    get_name_class = get_name_class_class(npz_path="assets/megadepth_test_1500_scene_info/0015_0.1_0.3.npz")
    img_name0, img_name1 = get_name_class.get_path(62)
    
    draw_class = Show_Matches()
    for i in range(0, 1, 1):
        draw_class.self_attention(img_name0, img_name1, f"/home/dk/LoFTR_NEW/zishiying/LoFTR_Auto_newattention/visual/Attention/self_atten{i}.jpg", index=i, max_index=300)
        # draw_class.cross_attention(img_name0, img_name1, f"/home/dk/LoFTR_NEW/FMAP_FINAL/func/cross_atten{i}.jpg", index=i, max_index=30)



































