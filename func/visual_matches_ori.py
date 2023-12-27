import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir("..")
from copy import deepcopy

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

class Show_Matches_outdoor():
    def __init__(self):
        super(Show_Matches_outdoor, self).__init__()
        self.img_resize = 832
        self.df = None
        self.img_padding = True


    def draw_line(self, img_name0, img_name1, save_path):
        img_name0 = img_name0
        img_name1 = img_name1

        image0, mask0, scale0 = read_megadepth_gray(img_name0, self.img_resize, self.df, self.img_padding, None)
        image1, mask1, scale1 = read_megadepth_gray(img_name1, self.img_resize, self.df, self.img_padding, None)
        img0_raw_color = cv2.imread(img_name0, cv2.IMREAD_COLOR)
        img1_raw_color = cv2.imread(img_name1, cv2.IMREAD_COLOR)
        img0_raw_color = cv2.cvtColor(img0_raw_color, cv2.COLOR_BGR2RGB)
        img1_raw_color = cv2.cvtColor(img1_raw_color, cv2.COLOR_BGR2RGB)

        img0 = image0[None].cuda()
        img1 = image1[None].cuda()
        scale0 = scale0[None].cuda()
        scale1 = scale1[None].cuda()
        batch = {'image0': img0, 'image1': img1,
                'image0_rgb': (img0 * 255.0),
                'image1_rgb': (img1 * 255.0),
                 'scale0': scale0,
                 'scale1': scale1}
        if mask0 is not None:  # img_padding is True
            [ts_mask_0, ts_mask_1] = F.interpolate(torch.stack([mask0, mask1], dim=0)[None].float(),
                                                   scale_factor=0.125,
                                                   mode='nearest',
                                                   recompute_scale_factor=False)[0].bool()
            batch.update({'mask0': ts_mask_0[None].cuda(), 'mask1': ts_mask_1[None].cuda()})
        with torch.no_grad():
            matcher(batch)
            mkpts0_c = batch['mkpts0_c'].cpu().numpy()
            mkpts1_c = batch['mkpts1_c'].cpu().numpy()
            mkpts0_f = batch['mkpts0_f'].cpu().numpy()
            mkpts1_f = batch['mkpts1_f'].cpu().numpy()
            mconf = batch['mconf'].cpu().numpy()
            color = cm.jet(mconf)
            print(mkpts0_f.shape)
            # color = [[0., 1., 0., 1.] for i in range(mkpts0_f.shape[0])]
        text = [
        ]
        fig = make_matching_figure(img0_raw_color, img1_raw_color, mkpts0_f, mkpts1_f, color, text=text,
                                   path=save_path)

if __name__ == '__main__':
    chose = "outdoor"
    _default_cfg = deepcopy(default_cfg)
    _default_cfg['coarse']['temp_bug_fix'] = True  # set to False when using the old ckpt
    matcher = LoFTR(config=_default_cfg)
    if chose == "outdoor":
        model_path = "/home/dk/LoFTR_NEW/zishiying/LoFTR_Auto_newattention/weights/epoch=59-auc@5=0.574-auc@10=0.729-auc@20=0.839.ckpt"
    elif chose == "indoor":
        model_path = "/home/dk/LoFTR_NEW/zishiying/LoFTR_Auto_without/weights/epoch=55-auc@5=0.543-auc@10=0.704-auc@20=0.821.ckpt"
    matcher.load_state_dict(torch.load(model_path)['state_dict'])
    matcher = matcher.eval().cuda()

    draw_class = Show_Matches_outdoor()
    draw_class.draw_line(img_name0 = f"/home/dk/LoFTR_NEW/LOFTR_MASK_FINAL/565_dwconv_fine/visualization/myimage/23.jpg",
            img_name1 = f"/home/dk/LoFTR_NEW/LOFTR_MASK_FINAL/565_dwconv_fine/visualization/myimage/32.jpg",
            save_path=f"/home/dk/LoFTR_NEW/MatchFormer/DSAP_img12.jpg")


    # draw_class.draw_line(img_name0 = "/home/dk/LoFTR_NEW/LOFTR_MASK_FINAL/565_dwconv_fine/visualization/myimage/TIB1.jpg",
    #                     img_name1 = "/home/dk/LoFTR_NEW/LOFTR_MASK_FINAL/565_dwconv_fine/visualization/myimage/TIB2.jpg",
    #                     save_path="/home/dk/LoFTR_NEW/LOFTR_MASK_FINAL/565_dwconv_fine/visualization/myimage/TIB.jpg")











