import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir("..")
from copy import deepcopy
import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.cm as cm
from src.utils.plotting import make_matching_figure
import time
from src.loftr import LoFTR, default_cfg
from src.utils.dataset import read_megadepth_gray, read_megadepth_depth, read_scannet_depth, \
    read_scannet_gray, read_scannet_pose, read_scannet_intrinsic, imread_gray
import torch


class Show_Matches():
    def __init__(self, npz_path):
        super(Show_Matches, self).__init__()
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

    def compute_rel_pose(self, scene_name, name0, name1):
        pose0 = self.read_abs_pose(scene_name, name0)
        pose1 = self.read_abs_pose(scene_name, name1)
        return np.matmul(pose1, np.linalg.inv(pose0))  # (4, 4)   T10

    def calcul_project(self, mkpts0, mkpts1, batch):
        depth0 = batch["depth0"]
        K_0, K_1 = batch["K0"], batch["K1"]
        T_0to1 = batch["T_0to1"]
        depth0 = depth0[None]
        K_0 = K_0[None]
        K_1 = K_1[None]
        T_0to1 = T_0to1[None]

        mkpts0_ = torch.from_numpy(mkpts0).long()
        kpts0_depth = depth0[torch.arange(0, 1).cuda().long(), mkpts0_[:, 1], mkpts0_[:, 0]]  							# 1024个特征点对应的深度 [B 1024 2] (u v)
        kpts0_h = torch.cat([mkpts0_, torch.ones_like(mkpts0_[:, [0]])], dim=-1) * kpts0_depth[..., None]

        K_0_inverse = torch.stack([k.inverse() for k in K_0], dim=0)  													# [B 3 3]
        kpts0_cam = torch.bmm(K_0_inverse, kpts0_h[None].transpose(1, 2))  												# TPw = Pc [B 3 1024]
        kpts1_cam = T_0to1[:, :3, :3] @ kpts0_cam + T_0to1[:, :3, [3]]  												# [B, 3, 1024]

        kpts1_KPw = (K_1 @ kpts1_cam).transpose(1, 2)  																	# [6, 1024, 3]
        kpts1_pix = kpts1_KPw[:, :, :2] / (kpts1_KPw[:, :, [2]] + 1e-9)  												# 图1中的像素坐标  [200 2]
        kpts1_pix = kpts1_pix.squeeze().cpu().numpy()
        return kpts1_pix

    def make_matching_figure(
            self, img0, mkpts0, mkpts1, mkpts_gt, dpi=200, path=None):
        # draw image pair
        img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
        assert mkpts0.shape[0] == mkpts1.shape[0], f'mkpts0: {mkpts0.shape[0]} v.s. mkpts1: {mkpts1.shape[0]}'
        fig, axes = plt.subplots(1, 1, figsize=(10, 6), dpi=dpi)
        axes.imshow(img0)
        axes.get_yaxis().set_ticks([])
        axes.get_xaxis().set_ticks([])
        for spine in axes.spines.values():
            spine.set_visible(False)
        plt.tight_layout(pad=1)

        axes.scatter(mkpts0[:, 0], mkpts0[:, 1], c='r', s=15)
        axes.scatter(mkpts1[:, 0], mkpts1[:, 1], c='g', s=15)
        axes.scatter(mkpts_gt[:, 0], mkpts_gt[:, 1], c='b', s=15)
        if path != None:
            plt.savefig(str(path), bbox_inches='tight', pad_inches=0)

    def draw_line(self, idx, number):
        (idx0, idx1), overlap_score, central_matches = self.pair_infos[idx]

        img_name0 = os.path.join(self.root_dir, self.scene_info['image_paths'][idx0])
        img_name1 = os.path.join(self.root_dir, self.scene_info['image_paths'][idx1])


        image0, mask0, scale0 = read_megadepth_gray(img_name0, self.img_resize, self.df, self.img_padding, None)
        image1, mask1, scale1 = read_megadepth_gray(img_name1, self.img_resize, self.df, self.img_padding, None)
        depth0 = read_megadepth_depth(os.path.join(self.root_dir, self.scene_info['depth_paths'][idx0]),
                                      pad_to=self.depth_max_size)
        depth1 = read_megadepth_depth(os.path.join(self.root_dir, self.scene_info['depth_paths'][idx1]),
                                      pad_to=self.depth_max_size)
        K_0 = torch.tensor(self.scene_info['intrinsics'][idx0].copy(), dtype=torch.float).reshape(3, 3)
        K_1 = torch.tensor(self.scene_info['intrinsics'][idx1].copy(), dtype=torch.float).reshape(3, 3)

        T0 = self.scene_info['poses'][idx0]
        T1 = self.scene_info['poses'][idx1]
        T_0to1 = torch.tensor(np.matmul(T1, np.linalg.inv(T0)), dtype=torch.float)[:4, :4]  # (4, 4)
        T_1to0 = T_0to1.inverse()
        img0_raw_color = cv2.imread(img_name0, cv2.IMREAD_COLOR)
        img1_raw_color = cv2.imread(img_name1, cv2.IMREAD_COLOR)

        img0 = image0[None].cuda()
        img1 = image1[None].cuda()
        scale0 = scale0[None].cuda()
        scale1 = scale1[None].cuda()
        batch = {'image0': img0, 'image1': img1,
                 'depth0': depth0, 'depth1': depth1,
                 'T_0to1': T_0to1,  # (4, 4)
                 'T_1to0': T_1to0,
                 'K0': K_0,  		# (3, 3)
                 'K1': K_1,
                 'mask0': mask0,
                 'mask1': mask1,
                 'scale0': scale0,
                 'scale1': scale1}
        # for LoFTR training
        if mask0 is not None:  # img_padding is True
            if self.coarse_scale:
                [ts_mask_0, ts_mask_1] = F.interpolate(torch.stack([mask0, mask1], dim=0)[None].float(),
                                                       scale_factor=self.coarse_scale,
                                                       mode='nearest',
                                                       recompute_scale_factor=False)[0].bool()
            batch.update({'mask0': ts_mask_0[None].cuda(), 'mask1': ts_mask_1[None].cuda()})

        with torch.no_grad():
            matcher(batch)
            mconf = batch['mconf']
            mkpts0_c = batch['mkpts0_c']
            mkpts0_c[:, 0] = torch.clamp(mkpts0_c[:, 0], 0, 840)
            mkpts0_c[:, 1] = torch.clamp(mkpts0_c[:, 1], 0, 840)
            mkpts1_c = batch['mkpts1_c']
            mkpts1_c[:, 0] = torch.clamp(mkpts1_c[:, 0], 0, 840)
            mkpts1_c[:, 1] = torch.clamp(mkpts1_c[:, 1], 0, 840)
            mkpts0_f = batch['mkpts0_f']
            mkpts0_f[:, 0] = torch.clamp(mkpts0_f[:, 0], 0, 840)
            mkpts0_f[:, 1] = torch.clamp(mkpts0_f[:, 1], 0, 840)
            mkpts1_f = batch['mkpts1_f']
            mkpts1_f[:, 0] = torch.clamp(mkpts1_f[:, 0], 0, 840)
            mkpts1_f[:, 1] = torch.clamp(mkpts1_f[:, 1], 0, 840)
            mkpts0_c = mkpts0_c.cpu().numpy()
            mkpts1_c = mkpts1_c.cpu().numpy()
            mkpts0_f = mkpts0_f.cpu().numpy()
            mkpts1_f = mkpts1_f.cpu().numpy()
            mconf = batch['mconf'].cpu().numpy()
            kpts1_pix_gt = self.calcul_project(mkpts0_f, mkpts1_f, batch=batch)
            color = cm.jet(mconf)

            original_dis = np.linalg.norm((kpts1_pix_gt - mkpts1_c), axis=-1, ord=2)
            optimize_dis = np.linalg.norm((mkpts1_f - mkpts1_c), axis=-1, ord=2)
            dis = np.linalg.norm((mkpts1_f - kpts1_pix_gt), axis=-1, ord=2)             # [N]

            max_dis_index = np.argmax(optimize_dis)
            print(optimize_dis[max_dis_index], dis[max_dis_index])
            flag = 0
            while dis[max_dis_index] > 50:
                optimize_dis[max_dis_index] = 0
                max_dis_index = np.argmax(optimize_dis)
                flag += 1
                if flag > 100:
                    print("绘制失败")
                    exit()

            print(max_dis_index, original_dis[max_dis_index], optimize_dis[max_dis_index], dis[max_dis_index])

            name = "0015_0.1_0.3_50"
            cv2.circle(img0_raw_color, (mkpts0_c.astype(np.int)[max_dis_index][0], mkpts0_c.astype(np.int)[max_dis_index][1]),
                        color=(0, 0, 255), radius=6, thickness=-1)
            cv2.imwrite(f"/home/dk/LoFTR_NEW/zishiying/LoFTR_Auto_newattention/visual/fine_process/{name}_coarse.jpg", img0_raw_color)
            cv2.circle(img1_raw_color, (mkpts1_c.astype(np.int)[max_dis_index][0], mkpts1_c.astype(np.int)[max_dis_index][1]),
                      color=(255, 0, 0), radius=6, thickness=-1)
            cv2.imwrite(f"/home/dk/LoFTR_NEW/zishiying/LoFTR_Auto_newattention/visual/fine_process/{name}_beforefine.jpg", img1_raw_color)
            img1_raw_color = cv2.imread(img_name1, cv2.IMREAD_COLOR)
            cv2.circle(img1_raw_color, (mkpts1_f.astype(np.int)[max_dis_index][0], mkpts1_f.astype(np.int)[max_dis_index][1]),
                      color=(0, 255, 0), radius=6, thickness=-1)
            cv2.imwrite(f"/home/dk/LoFTR_NEW/zishiying/LoFTR_Auto_newattention/visual/fine_process/{name}_afterfine.jpg", img1_raw_color)


class Show_Matches_outdoor():
    def __init__(self, npz_path):
        super(Show_Matches_outdoor, self).__init__()
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
    draw_class = Show_Matches(npz_path="assets/megadepth_test_1500_scene_info/0015_0.1_0.3.npz")
    draw_class.draw_line(idx=50, number=100)
