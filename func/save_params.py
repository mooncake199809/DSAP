# import torch
# model_params = torch.load("/home/dk/LoFTR_NEW/zishiying/LoFTR_Auto_newattention/weights/epoch=59-auc@5=0.574-auc@10=0.729-auc@20=0.839.ckpt")
# only_params = model_params["state_dict"]
# num = 0
# new_model_params = {}
# score_name_list = []
# score_list = []
# for keys, values in model_params["state_dict"].items():
#     if "score" in keys and "previous" not in keys and "_weight" not in keys:
#         num+=1
#         print(keys, values, "cross") if values > 0.5 else print(keys, values, "self")
#         score_name_list.append(keys)
#         score_list.append(values)

# only_params_list = list(only_params.keys())
# for index, score_name in enumerate(score_name_list):
#     current_name = score_name[:score_name.rfind(".")]
#     for allmodel_name in only_params_list:
#         if current_name in allmodel_name and "cross" in allmodel_name:
#             if score_list[index] > 0.5:
#                 res = only_params.pop(allmodel_name[:allmodel_name.rfind(".")]+'.weight', None)
#                 only_params_list.remove(allmodel_name[:allmodel_name.rfind(".")]+'.weight')
#             else:
#                 res = only_params.pop(allmodel_name[:allmodel_name.rfind(".")]+'.cross_weight', None)
#                 only_params_list.remove(allmodel_name[:allmodel_name.rfind(".")]+'.cross_weight')

# only_params_list = list(only_params.keys())
# for allmodel_name in only_params_list:
#     if "score" in allmodel_name or "current" in allmodel_name:
#         res = only_params.pop(allmodel_name, None)
#         only_params_list.remove(allmodel_name)
# torch.save(only_params, "/home/dk/LoFTR_NEW/zishiying/LoFTR_Auto_newattention/weights/only_params/only_paramas.ckpt")


import torch
torch.save(torch.load("/home/dk/LoFTR_NEW/quanzhongfuyong/LoFTR_FuY_originfine/new_weights/vdmatcher_outdoor_large.ckpt")["state_dict"],
           "/home/dk/LoFTR_NEW/quanzhongfuyong/LoFTR_FuY_originfine/new_weights/large_only_params.ckpt")












