# from configs.data.base import cfg

# TEST_BASE_PATH = "assets/megadepth_test_1500_scene_info"

# cfg.DATASET.TEST_DATA_SOURCE = "MegaDepth"
# cfg.DATASET.TEST_DATA_ROOT = "/mnt/lustre/xietao/megadepth"
# cfg.DATASET.TEST_NPZ_ROOT = "/mnt/lustre/xietao/megadepth/megadepth_indices/scene_info_val_1500"
# cfg.DATASET.TEST_LIST_PATH = "/mnt/lustre/xietao/megadepth/megadepth_indices/trainvaltest_list/val_list.txt"

# cfg.DATASET.MGDPT_IMG_RESIZE = 840
# cfg.DATASET.MIN_OVERLAP_SCORE_TEST = 0.0



from configs.data.base import cfg

TEST_BASE_PATH = "assets/megadepth_test_1500_scene_info"

cfg.DATASET.TEST_DATA_SOURCE = "MegaDepth"
cfg.DATASET.TEST_DATA_ROOT = "/mnt/share/sda-8T/xt/MegaDepth/"
cfg.DATASET.TEST_NPZ_ROOT = f"{TEST_BASE_PATH}"
cfg.DATASET.TEST_LIST_PATH = f"{TEST_BASE_PATH}/megadepth_test_1500.txt"

cfg.DATASET.MGDPT_IMG_RESIZE = 840
cfg.DATASET.MIN_OVERLAP_SCORE_TEST = 0.0


