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
from src.utils.plotting import make_matching_figure
import time
from src.loftr import LoFTR, default_cfg
from src.utils.dataset import read_megadepth_gray, read_megadepth_depth, read_scannet_depth, \
    read_scannet_gray, read_scannet_pose, read_scannet_intrinsic, imread_gray
import torch
import matplotlib.cm as cm

def estimate_pose(kpts0, kpts1, K0, K1, thresh, conf=0.99999):
    if len(kpts0) < 5:
        return None
    # normalize keypoints
    print(kpts0.shape)
    print(K0.shape)
    kpts0 = (kpts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
    kpts1 = (kpts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]

    # normalize ransac threshold
    ransac_thr = thresh / np.mean([K0[0, 0], K1[1, 1], K0[0, 0], K1[1, 1]])

    # compute pose with cv2
    E, mask = cv2.findEssentialMat(
        kpts0, kpts1, np.eye(3), threshold=ransac_thr, prob=conf, method=cv2.RANSAC)
    if E is None:
        print("\nE is None while trying to recover pose.\n")
        return None

    # recover pose from E
    best_num_inliers = 0
    ret = None
    for _E in np.split(E, len(E) / 3):
        n, R, t, _ = cv2.recoverPose(_E, kpts0, kpts1, np.eye(3), 1e9, mask=mask)
        if n > best_num_inliers:
            ret = (R, t[:, 0], mask.ravel() > 0)
            best_num_inliers = n

    return ret


def compute_pose_errors(data):
    pixel_thr = 0.5
    conf = 0.99999
    data.update({'R_errs': [], 't_errs': [], 'inliers': []})

    m_bids = data['m_bids'].cpu().numpy()
    pts0 = data['mkpts0_f'].cpu().numpy()
    pts1 = data['mkpts1_f'].cpu().numpy()
    K0 = data['K0'].cpu().numpy()
    K1 = data['K1'].cpu().numpy()
    T_0to1 = data['T_0to1'].cpu().numpy()

    for bs in range(K0.shape[0]):
        mask = m_bids == bs
        ret = estimate_pose(pts0[mask], pts1[mask], K0, K1, pixel_thr, conf=conf)

        if ret is None:
            data['R_errs'].append(np.inf)
            data['t_errs'].append(np.inf)
            data['inliers'].append(np.array([]).astype(np.bool))
        else:
            R, t, inliers = ret
            t_err, R_err = relative_pose_error(T_0to1, R, t, ignore_gt_t_thr=0.0)
            data['R_errs'].append(R_err)
            data['t_errs'].append(t_err)
            data['inliers'].append(inliers)


def relative_pose_error(T_0to1, R, t, ignore_gt_t_thr=0.0):
    # angle error between 2 vectors
    t_gt = T_0to1[:3, 3]
    n = np.linalg.norm(t) * np.linalg.norm(t_gt)
    t_err = np.rad2deg(np.arccos(np.clip(np.dot(t, t_gt) / n, -1.0, 1.0)))
    t_err = np.minimum(t_err, 180 - t_err)  # handle E ambiguity
    if np.linalg.norm(t_gt) < ignore_gt_t_thr:  # pure rotation is challenging
        t_err = 0

    # angle error between 2 rotation matrices
    R_gt = T_0to1[:3, :3]
    cos = (np.trace(np.dot(R.T, R_gt)) - 1) / 2
    cos = np.clip(cos, -1., 1.)  # handle numercial errors
    R_err = np.rad2deg(np.abs(np.arccos(cos)))

    return t_err, R_err
	
class Show_Matches():
	def __init__(self):
		super(Show_Matches, self).__init__()

	def imread_gray(self, path, augment_fn=None):
		cv_type = cv2.IMREAD_GRAYSCALE if augment_fn is None \
			else cv2.IMREAD_COLOR
		image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
		if augment_fn is not None:
			image = cv2.imread(str(path), cv2.IMREAD_COLOR)
			image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
			image = augment_fn(image)
			image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
		return image  # (h, w)

	def read_scannet_gray(self, path, resize=(640, 480), augment_fn=None):
		image = imread_gray(path, augment_fn)
		image = cv2.resize(image, resize)
		image = torch.from_numpy(image).float()[None] / 255
		return image

	def read_abs_pose(self, scene_name, name):
		pth = os.path.join(pose_dir, scene_name, 'pose', f'{name}.txt')
		return read_scannet_pose(pth)

	def compute_rel_pose(self, scene_name, name0, name1):
		pose0 = self.read_abs_pose(scene_name, name0)
		pose1 = self.read_abs_pose(scene_name, name1)
		return np.matmul(pose1, np.linalg.inv(pose0))  # (4, 4)   T10

	def calcul_project(self, mkpts0, mkpts1):
		depth0 = read_scannet_depth(os.path.join(root_dir, scene_name, 'depth', f'{stem_name_0}.png'))
		K_0 = K_1 = torch.tensor(intrinsics[scene_name].copy(), dtype=torch.float).reshape(3, 3)
		T_0to1 = torch.tensor(self.compute_rel_pose(scene_name, stem_name_0, stem_name_1), dtype=torch.float32)
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
		gt_error = torch.norm(torch.from_numpy(kpts1_pix - mkpts1), dim=-1, keepdim=False)
		return gt_error

	def draw_line(self, coarse_path, fine_path, thre, test_time=False):
		img0_raw = cv2.imread(img_name0, cv2.IMREAD_GRAYSCALE)
		img1_raw = cv2.imread(img_name1, cv2.IMREAD_GRAYSCALE)
		img0_raw = cv2.resize(img0_raw, (640, 480))
		img1_raw = cv2.resize(img1_raw, (640, 480))

		img0_raw_color = cv2.imread(img_name0, cv2.IMREAD_COLOR)
		img1_raw_color = cv2.imread(img_name1, cv2.IMREAD_COLOR)
		img0_raw_color = cv2.resize(img0_raw_color, (640, 480))
		img1_raw_color = cv2.resize(img1_raw_color, (640, 480))
		img0_raw_color = cv2.cvtColor(img0_raw_color, cv2.COLOR_BGR2RGB)
		img1_raw_color = cv2.cvtColor(img1_raw_color, cv2.COLOR_BGR2RGB)

		img0 = torch.from_numpy(img0_raw)[None][None].cuda() / 255.
		img1 = torch.from_numpy(img1_raw)[None][None].cuda() / 255.
		batch = {'image0': img0, 'image1': img1}

		with torch.no_grad():
			if test_time == True:
				start = time.time()
				for i in range(100):
					print(i)
					if i == 0:
						continue
					matcher(batch, current_epoch=20)
				print(f"平均时间:{(time.time()-start)/100}")

			matcher(batch, current_epoch=20)
			mkpts0_c = batch['mkpts0_c']
			mkpts0_c[:, 0] = torch.clamp(mkpts0_c[:, 0], 0, 640)
			mkpts0_c[:, 1] = torch.clamp(mkpts0_c[:, 1], 0, 480)
			mkpts1_c = batch['mkpts1_c']
			mkpts1_c[:, 0] = torch.clamp(mkpts1_c[:, 0], 0, 640)
			mkpts1_c[:, 1] = torch.clamp(mkpts1_c[:, 1], 0, 480)
			mkpts0_f = batch['mkpts0_f']
			mkpts0_f[:, 0] = torch.clamp(mkpts0_f[:, 0], 0, 640)
			mkpts0_f[:, 1] = torch.clamp(mkpts0_f[:, 1], 0, 480)
			mkpts1_f = batch['mkpts1_f']
			mkpts1_f[:, 0] = torch.clamp(mkpts1_f[:, 0], 0, 640)
			mkpts1_f[:, 1] = torch.clamp(mkpts1_f[:, 1], 0, 480)
			mkpts0_c = mkpts0_c.cpu().numpy()
			mkpts1_c = mkpts1_c.cpu().numpy()
			mkpts0_f = mkpts0_f.cpu().numpy()
			mkpts1_f = mkpts1_f.cpu().numpy()
			mconf = batch['mconf'].cpu().numpy()
			project_err = self.calcul_project(mkpts0_f, mkpts1_f)
			print(project_err)
			print(np.median(project_err.cpu().numpy()))
			project_err_mask = project_err < thre
			print(f"匹配点数:{project_err_mask.shape[0]}, 正确匹配点数{project_err_mask.sum()}, 准确率{project_err_mask.sum() / project_err_mask.shape[0]}")
			color = [[0., 1., 0., 1.] if err else [1., 0., 0., 1.] for err in project_err_mask]
		text = [
		]
		fig = make_matching_figure(img0_raw_color, img1_raw_color, mkpts0_c, mkpts1_c, color, text=text, path=coarse_path)
		fig = make_matching_figure(img0_raw_color, img1_raw_color, mkpts0_f, mkpts1_f, color, text=text, path=fine_path)

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
		gt_error = torch.norm(torch.from_numpy(kpts1_pix - mkpts1), dim=-1, keepdim=False)
		return gt_error

	def draw_line(self, idx, coarse_path, fine_path, thre):
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

		print(img_name0)
		print(img_name1)
		print(os.path.join(self.root_dir, self.scene_info['depth_paths'][idx0]),)
		print(os.path.join(self.root_dir, self.scene_info['depth_paths'][idx1]))
		print(K_0)
		print(K_1)
		print(T_0to1)
		print(T_1to0)


		img0_raw_color = cv2.imread(img_name0, cv2.IMREAD_COLOR)
		img1_raw_color = cv2.imread(img_name1, cv2.IMREAD_COLOR)

		cv2.imwrite("/home/dk/LoFTR_NEW/zishiying/LoFTR_Auto_newattention/visual/stacking/22_13_205_1.jpg", img0_raw_color)
		cv2.imwrite("/home/dk/LoFTR_NEW/zishiying/LoFTR_Auto_newattention/visual/stacking/22_13_205_2.jpg", img1_raw_color)
		img0_raw_color = cv2.cvtColor(img0_raw_color, cv2.COLOR_BGR2RGB)
		img1_raw_color = cv2.cvtColor(img1_raw_color, cv2.COLOR_BGR2RGB)

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
			mkpts0_c = batch['mkpts0_c'].cpu().numpy()
			mkpts1_c = batch['mkpts1_c'].cpu().numpy()
			mkpts0_f = batch['mkpts0_f'].cpu().numpy()
			mkpts1_f = batch['mkpts1_f'].cpu().numpy()
			mconf = batch['mconf'].cpu().numpy()
			project_err = self.calcul_project(mkpts0_f, mkpts1_f, batch=batch)
			# print(project_err)
			# print(np.median(project_err.cpu().numpy()))
			project_err_mask = project_err < thre
			P = project_err_mask.sum() / project_err_mask.shape[0]
			if P > 0:
				print(f"{idx}----匹配点数:{project_err_mask.shape[0]}, 正确匹配点数{project_err_mask.sum()}, 准确率{P} \n")
			# return f"{idx}----匹配点数:{project_err_mask.shape[0]}, 正确匹配点数{project_err_mask.sum()}, 准确率{P} \n"
			color = [[0., 1., 0., 1.] if err else [1., 0., 0., 1.] for err in project_err_mask]
			compute_pose_errors(batch)
		text = [
			# f'LoFTR, P: {(project_err_mask.sum() / project_err_mask.shape[0]) * 100:.1f}%',
			# 'Matches: {}'.format(len(mkpts0_f)),
			# f'ΔR:{(batch["R_errs"][0]):.2f}°, Δt:{(batch["t_errs"][0]):.2f}°'
		]
		print(f'FMAP, P: {(project_err_mask.sum() / project_err_mask.shape[0]) * 100:.1f}%')
		print('Matches: {}'.format(len(mkpts0_f)))
		print(f'ΔR:{(batch["R_errs"][0]):.2f}°, Δt:{(batch["t_errs"][0]):.2f}°')
		print("------------------------------------------------------------------------")
		print(img0_raw_color.shape, img1_raw_color.shape)
		fig = make_matching_figure(img0_raw_color, img1_raw_color, mkpts0_f, mkpts1_f, color, text=text,
								   path=fine_path)


if __name__ == '__main__':
	_default_cfg = deepcopy(default_cfg)
	_default_cfg['coarse']['temp_bug_fix'] = True  # set to False when using the old ckpt
	matcher = LoFTR(config=_default_cfg)
	matcher.load_state_dict(
		torch.load("/home/dk/LoFTR_NEW/zishiying/LoFTR_Auto_newattention/weights/epoch=59-auc@5=0.574-auc@10=0.729-auc@20=0.839.ckpt")['state_dict'])
	matcher = matcher.eval().cuda()

	scene_name, scene_sub_name, stem_name_0, stem_name_1 = 786, 0, 1365, 1545
	scene_name = f'scene{scene_name:04d}_{scene_sub_name:02d}'
	intrinsics = dict(np.load('/home/dk/LOFTR/SuperGlue/data/scannet/test_intrinsics.npz'))
	root_dir = '/mnt/share/sda-8T/xt/FEAM/scannet_test_1500/'
	pose_dir = '/mnt/share/sda-8T/xt/FEAM/scannet_test_1500/'
	img_name0 = os.path.join(root_dir, scene_name, 'color', f'{stem_name_0}.jpg')
	img_name1 = os.path.join(root_dir, scene_name, 'color', f'{stem_name_1}.jpg')

	chose = "outdoor"
	if chose == "indoor":
		draw_class = Show_Matches()
		draw_class.draw_line(coarse_path="/home/dk/LoFTR_NEW/LoFTR_tsinghua_pos/visual/matches/SMAP_coarse.png",
							 fine_path="/home/dk/LoFTR_NEW/LoFTR_tsinghua_pos/visual/matches/SMAP.png",
							 thre=10, test_time=False)
	else:
		draw_class = Show_Matches_outdoor(npz_path="assets/megadepth_test_1500_scene_info/0022_0.1_0.3.npz")
		# for index in range(0,10000):
		# 	res = draw_class.draw_line(idx=index,coarse_path="/home/dk/LoFTR_NEW/LoFTR_tsinghua_pos/visual/matches/outdoor/FMAP2.png",
		# 						 fine_path="/home/dk/LoFTR_NEW/LoFTR_tsinghua_pos/visual/matches/outdoor/FMAP2.png",
		# 						 thre=10)
		# 	save_txt = "/home/dk/LoFTR_NEW/zishiying/LoFTR_Auto_newattention/func/15_3_5.txt"
		# 	with open(save_txt, 'a') as f:
		# 		f.writelines(res)
		# f.close()

		for index in range(0,1):
			draw_class.draw_line(idx=205,coarse_path="/home/dk/LoFTR_NEW/zishiying/LoFTR_Auto_newattention/visual/stacking/22_13_205.png",
								 fine_path="/home/dk/LoFTR_NEW/zishiying/LoFTR_Auto_newattention/visual/stacking/22_13_205.png",
								 thre=3000)
































































