# DSAP: A Dynamic Sparse Attention Perception Matcher for Accurate Local Feature Matching
This is the PyTorch implementation of our paper "DSAP: A Dynamic Sparse Attention Perception Matcher for Accurate Local Feature Matching".

# Quick Demo
![demo_img](https://github.com/mooncake199809/DSAP/blob/main/demo/img_res.jpg)
We provide a demo to directly visualize the matching results of DSAP.
You can directly modify your images path to test your own images.
```bash
python demo_dsap.py
```

# Installation
Our project is built upon the official code of LoFTR and trained on the MegaDepth dataset.
Please follow [LoFTR](https://github.com/zju3dv/LoFTR) to install the environment and MegaDepth dataset.

# Training
We can run the scripts/reproduce_train/outdoor_ds.sh file to train DSAP.
```bash
bash scripts/reproduce_train/outdoor_ds.sh
```

# Evaluation
The pre-trained models can be downloaded from [DSAP_Mega](https://1drv.ms/u/s!AsLK4P4ia2R9biMdEyi_uQ-0No0?e=kLSPnh).
```bash
bash scripts/reproduce_test/outdoor_ds.sh
```

# Acknowledgements
We appreciate the previous open-source repository [LoFTR](https://github.com/zju3dv/LoFTR).
We thank for the excellent contribution of LoFTR.

