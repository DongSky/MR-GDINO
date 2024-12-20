import torch
import os
import pickle


subtask_key = [
    "TASK_NAME1",
    "TASK_NAME2",
    ...
]



ckpt_list = [
    "OUTPUT_DIR_TASK1",
    "OUTPUT_DIR_TASK2",
    ...
             ]



save_dict = {}
for i in range(len(ckpt_list)):

    model_dict = torch.load("{}/checkpoint.pth".format(ckpt_list[i]))
    
    save_dict[subtask_key[i]] = {"coop_prompt":model_dict['model']['coop.coop_prompt'].cpu()}



with open("multi_model_prompt_params.pkl",'wb') as f:
    pickle.dump(save_dict, f, -1)