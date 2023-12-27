import torch
model_params = torch.load("/home/dk/LoFTR_NEW/zishiying/LoFTR_Auto_newattention/weights/epoch=59-auc@5=0.574-auc@10=0.729-auc@20=0.839.ckpt")
print(model_params.keys())
num = 0
for keys, values in model_params["state_dict"].items():
    if "score" in keys and "previous" not in keys and "_weight" not in keys:
        num+=1
        print(keys, values)
print(num)


