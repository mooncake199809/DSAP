import cv2
import torch
import time
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir("..")
from src.loftr import LoFTR, default_cfg
from copy import deepcopy
from thop import profile

_default_cfg = deepcopy(default_cfg)
_default_cfg['coarse']['temp_bug_fix'] = True  # set to False when using the old ckpt
matcher = LoFTR(config=_default_cfg)
matcher.load_state_dict(
    torch.load(
        "/home/dk/LoFTR_NEW/zishiying/LoFTR_Auto_newattention/weights/epoch=59-auc@5=0.574-auc@10=0.729-auc@20=0.839.ckpt")
    ['state_dict'])
matcher = matcher.eval().cuda()
image0 = cv2.imread("/home/dk/LOFTR/SuperGlue/img_input/22_13_46_1.jpg", cv2.IMREAD_GRAYSCALE)
image1 = cv2.imread("/home/dk/LOFTR/SuperGlue/img_input/22_13_46_2.jpg", cv2.IMREAD_GRAYSCALE)

# image0 = cv2.imread("/mnt/share/sda-8T/xt/FEAM/scannet_test_1500/scene0790_00/color/675.jpg", cv2.IMREAD_GRAYSCALE)
# image1 = cv2.imread("/mnt/share/sda-8T/xt/FEAM/scannet_test_1500/scene0790_00/color/885.jpg", cv2.IMREAD_GRAYSCALE)
image0 = cv2.resize(image0, (840, 840))
image1 = cv2.resize(image1, (840, 840))
img0 = torch.from_numpy(image0)[None][None].cuda() / 255.
img1 = torch.from_numpy(image1)[None][None].cuda() / 255.
batch = {'image0': img0, 'image1': img1}

with torch.no_grad():
    all_time = 0
    for i in range(20):
        start_time = time.time()
        matcher(batch)
        if i == 0:
            continue
        all_time += time.time() - start_time
        print(time.time() - start_time)
    print(all_time/20)
    flops, params = profile(matcher, (batch,))
    print(flops/1e9)
    print(params/1e6)




