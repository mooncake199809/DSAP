from src.config.default import _CN as cfg

cfg.DSAP.COARSE.TEMP_BUG_FIX = False
cfg.DSAP.MATCH_COARSE.MATCH_TYPE = 'dual_softmax'

cfg.TRAINER.MSLR_MILESTONES = [3, 6, 9, 12, 17, 20, 23, 26, 29]
