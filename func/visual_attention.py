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
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from src.utils.plotting import make_matching_figure
import time
from src.loftr import LoFTR_POS, default_cfg
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
		img0_raw = cv2.resize(img0_raw, (640, 480))
		img1_raw = cv2.resize(img1_raw, (640, 480))

		img0_raw_color = cv2.imread(img0_pth, cv2.IMREAD_COLOR)
		img1_raw_color = cv2.imread(img1_pth, cv2.IMREAD_COLOR)
		img0_raw_color = cv2.resize(img0_raw_color, (640, 480))
		img1_raw_color = cv2.resize(img1_raw_color, (640, 480))

		new_image = np.ones((480, 1300, 3)) * 255
		new_image[:, :640, :] = img0_raw_color
		new_image[:, 660:, :] = img1_raw_color

		img0 = torch.from_numpy(img0_raw)[None][None].cuda() / 255.
		img1 = torch.from_numpy(img1_raw)[None][None].cuda() / 255.
		batch = {'image0': img0, 'image1': img1}
		return img0_raw, img1_raw, img0_raw_color, img1_raw_color, new_image, batch

	def self_attention(self, img0_pth, img1_pth, save_path, index, max_index):
		img0_raw, img1_raw, img0_raw_color, img1_raw_color, new_image, batch = self.initial_set(img0_pth, img1_pth)
		with torch.no_grad():
			matcher(batch, current_epoch=20)
			mkpts0_f = batch['mkpts0_f']
			mkpts0_f[:, 0] = torch.clamp(mkpts0_f[:, 0], 0, 640)
			mkpts0_f[:, 1] = torch.clamp(mkpts0_f[:, 1], 0, 480)
			mkpts1_f = batch['mkpts1_f']
			mkpts1_f[:, 0] = torch.clamp(mkpts1_f[:, 0], 0, 640)
			mkpts1_f[:, 1] = torch.clamp(mkpts1_f[:, 1], 0, 480)
			mkpts0_f = mkpts0_f.long()
			mkpts1_f = mkpts1_f.long()
			feat0 = batch["feat0"];
			feat0_self_matrix = torch.sigmoid(batch["self0_matrix"] / feat0.shape[-1] ** 0.5)
			feat1_self_matrix = torch.sigmoid(batch["self1_matrix"] / feat0.shape[-1] ** 0.5)

			b = batch["b_seed_coarse"][index]
			i = batch["i_seed_coarse"][index]
			j = batch["j_seed_coarse"][index]
			feat0_self = feat0_self_matrix[b, i, :]  # [N]
			feat1_self = feat1_self_matrix[b, j, :]
			feat0_coord = mkpts0_f[index]
			feat1_coord = mkpts1_f[index]
			feat1_coord[0] = feat1_coord[0] + 660

			i_max_val, i_max_index = torch.topk(feat0_self, k=max_index)
			j_max_val, j_max_index = torch.topk(feat1_self, k=max_index)
			i_max_val = self.normalize(i_max_val)
			j_max_val = self.normalize(j_max_val)
			b, n, c = batch["spts0_i"].shape
			original0_coord = batch["spts0_i"].reshape(n, c).long()
			original1_coord = batch["spts1_i"].reshape(n, c).long()
			max_mkpts0_f = original0_coord[i_max_index]
			max_mkpts1_f = original1_coord[j_max_index]
			max_mkpts1_f[:, 0] = max_mkpts1_f[:, 0] + 660
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
			cv2.line(new_image, (x0, y0), (x1, y1), color=c, thickness=2)
		for (x0, y0), (x1, y1) in zip(feat_self_all0, feat_self_all1):
			cv2.circle(new_image, (x0, y0), color=(0, 255, 0), radius=3, thickness=-1)
			cv2.circle(new_image, (x1, y1), color=(0, 255, 0), radius=3, thickness=-1)
		feat0_coord = feat0_coord.long().cpu().numpy()
		feat1_coord = feat1_coord.long().cpu().numpy()
		cv2.line(new_image, (feat0_coord[0], feat0_coord[1]), (feat1_coord[0], feat1_coord[1]), color=(255, 0, 0),
				 thickness=2)
		cv2.imwrite(save_path, new_image)

	def cross_attention(self, img0_pth, img1_pth, save_path, index, max_index):
		img0_raw, img1_raw, img0_raw_color, img1_raw_color, new_image, batch = self.initial_set(img0_pth, img1_pth)
		with torch.no_grad():
			matcher(batch, current_epoch=20)
			mkpts0_f = batch['mkpts0_f']
			mkpts0_f[:, 0] = torch.clamp(mkpts0_f[:, 0], 0, 640)
			mkpts0_f[:, 1] = torch.clamp(mkpts0_f[:, 1], 0, 480)
			mkpts1_f = batch['mkpts1_f']
			mkpts1_f[:, 0] = torch.clamp(mkpts1_f[:, 0], 0, 640)
			mkpts1_f[:, 1] = torch.clamp(mkpts1_f[:, 1], 0, 480)
			mkpts0_f = mkpts0_f.long()
			mkpts1_f = mkpts1_f.long()
			feat0 = batch["feat0"]
			feat_cross_matrix = torch.sigmoid(batch["cross_matrix"] / feat0.shape[-1] ** 0.5)

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
			max_mkpts1_f[:, 0] = max_mkpts1_f[:, 0] + 660

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
			cv2.line(new_image, (x0, y0), (x1, y1), color=c, thickness=2, lineType=cv2.LINE_AA)
		for (x0, y0), (x1, y1) in zip(feat0_cross, feat1_cross):
			cv2.circle(new_image, (x0, y0), color=(0, 255, 0), radius=3, thickness=-1)
			cv2.circle(new_image, (x1, y1), color=(0, 255, 0), radius=3, thickness=-1)
		new_image = torch.sigmoid(torch.from_numpy(new_image)).numpy()
		src = plt.imshow(new_image[:,:,0], cmap='Reds')
		plt.colorbar(src)
		plt.show()
		exit()
		cv2.imwrite(save_path, new_image)


if __name__ == '__main__':
	_default_cfg = deepcopy(default_cfg)
	_default_cfg['coarse']['temp_bug_fix'] = True  # set to False when using the old ckpt
	matcher = LoFTR_POS(config=_default_cfg, mode="eval")
	matcher.load_state_dict(
		torch.load("/home/dk/LOFTR/LoFTR_2/weights/Scannet/final-auc@5=21.58-auc@10=40.76-auc@20=58.74.ckpt")[
			'state_dict'])
	matcher = matcher.eval().cuda()

	scene_name, scene_sub_name, stem_name_0, stem_name_1 = 737, 0, 990, 1110
	scene_name = f'scene{scene_name:04d}_{scene_sub_name:02d}'
	intrinsics = dict(np.load('/home/dk/LOFTR/SuperGlue/data/scannet/test_intrinsics.npz'))
	root_dir = '/mnt/sda2/xt/FEAM/scannet_test_1500/'
	pose_dir = '/mnt/sda2/xt/FEAM/scannet_test_1500/'
	img_name0 = os.path.join(root_dir, scene_name, 'color', f'{stem_name_0}.jpg')
	img_name1 = os.path.join(root_dir, scene_name, 'color', f'{stem_name_1}.jpg')

	draw_class = Show_Matches()
	for i in range(0, 100):
		draw_class.self_attention(img_name0, img_name1, f"/home/dk/LOFTR/LoFTR_2/visualization/Matches/image0/self_attention/self_atten{i}.jpg", index=i, max_index=30)
		draw_class.cross_attention(img_name0, img_name1, f"/home/dk/LOFTR/LoFTR_2/visualization/Matches/image0/self_attention/cross_atten{i}.jpg", index=i, max_index=30)


































