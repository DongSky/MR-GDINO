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
    
    new_model_dict = {}
    
    # print(model_dict['model'].keys())
    
    for key in model_dict['model']:
        if key.startswith("transformer.encoder.fusion_layers"):
            new_model_dict[key.replace("transformer.","")] = model_dict['model'][key]


            
    
    save_dict[subtask_key[i]] = new_model_dict



with open("multi_model_lora_params.pkl",'wb') as f:
    pickle.dump(save_dict, f, -1)