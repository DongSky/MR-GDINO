import torch
import pickle
import os
# subset = os.listdir("task_image_feat_bank")

root = "FEATRURE_SAVE_ROOT"

task_list = [
            "TASK_NAME1",
            "TASK_NAME2",
            ...
             ]



# with open(os.path.join(root,'mean_feat','task_feats.pkl'),'rb') as f:
#     save_dict = pickle.load(f)

save_dict = {}

for subtask in task_list:
    subtask_path = os.path.join(root, subtask)
    
    data_point = os.listdir(subtask_path)
    
    data_list = []
    
    for i in range(len(data_point)):
        data_path = os.path.join(root, subtask, data_point[i])
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
            data_list.append(data)
    data_mean = torch.mean(torch.stack(data_list),dim=0)
    save_dict[subtask] = data_mean
    
save_path = os.path.join(root,'mean_feat')
if not os.path.exists(save_path):
    os.mkdir(save_path)
with open(os.path.join(save_path,'task_feats.pkl'),'wb') as f:
    pickle.dump(save_dict, f, -1)
    
