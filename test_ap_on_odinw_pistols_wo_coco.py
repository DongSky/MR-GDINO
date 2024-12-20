import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler

# from groundingdino.models import build_model
import groundingdino.datasets.transforms as T
from groundingdino.util import box_ops, get_tokenlizer
from groundingdino.util.misc import clean_state_dict, collate_fn
from groundingdino.util.slconfig import SLConfig
import json
# from torchvision.datasets import CocoDetection
import torchvision
import pickle
from groundingdino.util.vl_utils import build_captions_and_token_span, create_positive_map_from_span
from groundingdino.datasets.cocogrounding_eval import CocoGroundingEvaluator



def build_model(args):
    # we use register to maintain models from catdet6 on.
    from models.registry import MODULE_BUILD_FUNCS

    assert args.modelname in MODULE_BUILD_FUNCS._module_dict
    build_func = MODULE_BUILD_FUNCS.get(args.modelname)
    model = build_func(args)
    return model


def load_model(args_origin,  model_checkpoint_path: str, device: str = "cuda"):
    model_config_path = args_origin.config_file
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    args.coco_val_path = args_origin.coco_val_path
    args.datasets = 'odinw'
    args.use_coop = args_origin.use_coop
    args.use_moe_lora = args_origin.use_moe_lora
    args.use_adapter = args_origin.use_adapter
    args.use_prompt = args_origin.use_prompt
    args.use_zira = args_origin.use_zira
    model,_,_ = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    model.eval()
    return model


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms):
        super().__init__(img_folder, ann_file)
        self._transforms = transforms

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)  # target: list

        # import ipdb; ipdb.set_trace()

        w, h = img.size
        boxes = [obj["bbox"] for obj in target]
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]  # xywh -> xyxy
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)
        # filt invalid boxes/masks/keypoints
        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]

        target_new = {}
        image_id = self.ids[idx]
        target_new["image_id"] = image_id
        target_new["boxes"] = boxes
        target_new["orig_size"] = torch.as_tensor([int(h), int(w)])

        if self._transforms is not None:
            img, target = self._transforms(img, target_new)

        return img, target

categories_aerial = [{"id": 1, "name": "boat", "supercategory": "movable-objects"}, {"id": 2, "name": "car", "supercategory": "movable-objects"}, {"id": 3, "name": "dock", "supercategory": "movable-objects"}, {"id": 4, "name": "jetski", "supercategory": "movable-objects"}, {"id": 5, "name": "lift", "supercategory": "movable-objects"}]
categories_coco = [{"supercategory": "person","id": 1,"name": "person"},{"supercategory": "vehicle","id": 2,"name": "bicycle"},{"supercategory": "vehicle","id": 3,"name": "car"},{"supercategory": "vehicle","id": 4,"name": "motorcycle"},{"supercategory": "vehicle","id": 5,"name": "airplane"},{"supercategory": "vehicle","id": 6,"name": "bus"},{"supercategory": "vehicle","id": 7,"name": "train"},{"supercategory": "vehicle","id": 8,"name": "truck"},{"supercategory": "vehicle","id": 9,"name": "boat"},{"supercategory": "outdoor","id": 10,"name": "traffic light"},{"supercategory": "outdoor","id": 11,"name": "fire hydrant"},{"supercategory": "outdoor","id": 13,"name": "stop sign"},{"supercategory": "outdoor","id": 14,"name": "parking meter"},{"supercategory": "outdoor","id": 15,"name": "bench"},{"supercategory": "animal","id": 16,"name": "bird"},{"supercategory": "animal","id": 17,"name": "cat"},{"supercategory": "animal","id": 18,"name": "dog"},{"supercategory": "animal","id": 19,"name": "horse"},{"supercategory": "animal","id": 20,"name": "sheep"},{"supercategory": "animal","id": 21,"name": "cow"},{"supercategory": "animal","id": 22,"name": "elephant"},{"supercategory": "animal","id": 23,"name": "bear"},{"supercategory": "animal","id": 24,"name": "zebra"},{"supercategory": "animal","id": 25,"name": "giraffe"},{"supercategory": "accessory","id": 27,"name": "backpack"},{"supercategory": "accessory","id": 28,"name": "umbrella"},{"supercategory": "accessory","id": 31,"name": "handbag"},{"supercategory": "accessory","id": 32,"name": "tie"},{"supercategory": "accessory","id": 33,"name": "suitcase"},{"supercategory": "sports","id": 34,"name": "frisbee"},{"supercategory": "sports","id": 35,"name": "skis"},{"supercategory": "sports","id": 36,"name": "snowboard"},{"supercategory": "sports","id": 37,"name": "sports ball"},{"supercategory": "sports","id": 38,"name": "kite"},{"supercategory": "sports","id": 39,"name": "baseball bat"},{"supercategory": "sports","id": 40,"name": "baseball glove"},{"supercategory": "sports","id": 41,"name": "skateboard"},{"supercategory": "sports","id": 42,"name": "surfboard"},{"supercategory": "sports","id": 43,"name": "tennis racket"},{"supercategory": "kitchen","id": 44,"name": "bottle"},{"supercategory": "kitchen","id": 46,"name": "wine glass"},{"supercategory": "kitchen","id": 47,"name": "cup"},{"supercategory": "kitchen","id": 48,"name": "fork"},{"supercategory": "kitchen","id": 49,"name": "knife"},{"supercategory": "kitchen","id": 50,"name": "spoon"},{"supercategory": "kitchen","id": 51,"name": "bowl"},{"supercategory": "food","id": 52,"name": "banana"},{"supercategory": "food","id": 53,"name": "apple"},{"supercategory": "food","id": 54,"name": "sandwich"},{"supercategory": "food","id": 55,"name": "orange"},{"supercategory": "food","id": 56,"name": "broccoli"},{"supercategory": "food","id": 57,"name": "carrot"},{"supercategory": "food","id": 58,"name": "hot dog"},{"supercategory": "food","id": 59,"name": "pizza"},{"supercategory": "food","id": 60,"name": "donut"},{"supercategory": "food","id": 61,"name": "cake"},{"supercategory": "furniture","id": 62,"name": "chair"},{"supercategory": "furniture","id": 63,"name": "couch"},{"supercategory": "furniture","id": 64,"name": "potted plant"},{"supercategory": "furniture","id": 65,"name": "bed"},{"supercategory": "furniture","id": 67,"name": "dining table"},{"supercategory": "furniture","id": 70,"name": "toilet"},{"supercategory": "electronic","id": 72,"name": "tv"},{"supercategory": "electronic","id": 73,"name": "laptop"},{"supercategory": "electronic","id": 74,"name": "mouse"},{"supercategory": "electronic","id": 75,"name": "remote"},{"supercategory": "electronic","id": 76,"name": "keyboard"},{"supercategory": "electronic","id": 77,"name": "cell phone"},{"supercategory": "appliance","id": 78,"name": "microwave"},{"supercategory": "appliance","id": 79,"name": "oven"},{"supercategory": "appliance","id": 80,"name": "toaster"},{"supercategory": "appliance","id": 81,"name": "sink"},{"supercategory": "appliance","id": 82,"name": "refrigerator"},{"supercategory": "indoor","id": 84,"name": "book"},{"supercategory": "indoor","id": 85,"name": "clock"},{"supercategory": "indoor","id": 86,"name": "vase"},{"supercategory": "indoor","id": 87,"name": "scissors"},{"supercategory": "indoor","id": 88,"name": "teddy bear"},{"supercategory": "indoor","id": 89,"name": "hair drier"},{"supercategory": "indoor","id": 90,"name": "toothbrush"}]
categories_aqua = [{"id": 1, "name": "fish", "supercategory": "creatures"}, {"id": 2, "name": "jellyfish", "supercategory": "creatures"}, {"id": 3, "name": "penguin", "supercategory": "creatures"}, {"id": 4, "name": "puffin", "supercategory": "creatures"}, {"id": 5, "name": "shark", "supercategory": "creatures"}, {"id": 6, "name": "starfish", "supercategory": "creatures"}, {"id": 7, "name": "stingray", "supercategory": "creatures"}]
categories_rabbit = [{"id": 1, "name": "Cottontail-Rabbit", "supercategory": "Cottontail-Rabbit"}]
categories_egohand = [{"id": 1, "name": "hand", "supercategory": "hands"}]
categories_mushroom = [{"id": 1, "name": "CoW", "supercategory": "mushroom"}, {"id": 2, "name": "chanterelle", "supercategory": "mushroom"}]
categories_package = [{"id": 1, "name": "package", "supercategory": "packages"}]
categories_voc = [{"id": 1, "name": "aeroplane", "supercategory": "VOC"}, {"id": 2, "name": "bicycle", "supercategory": "VOC"}, {"id": 3, "name": "bird", "supercategory": "VOC"}, {"id": 4, "name": "boat", "supercategory": "VOC"}, {"id": 5, "name": "bottle", "supercategory": "VOC"}, {"id": 6, "name": "bus", "supercategory": "VOC"}, {"id": 7, "name": "car", "supercategory": "VOC"}, {"id": 8, "name": "cat", "supercategory": "VOC"}, {"id": 9, "name": "chair", "supercategory": "VOC"}, {"id": 10, "name": "cow", "supercategory": "VOC"}, {"id": 11, "name": "diningtable", "supercategory": "VOC"}, {"id": 12, "name": "dog", "supercategory": "VOC"}, {"id": 13, "name": "horse", "supercategory": "VOC"}, {"id": 14, "name": "motorbike", "supercategory": "VOC"}, {"id": 15, "name": "person", "supercategory": "VOC"}, {"id": 16, "name": "pottedplant", "supercategory": "VOC"}, {"id": 17, "name": "sheep", "supercategory": "VOC"}, {"id": 18, "name": "sofa", "supercategory": "VOC"}, {"id": 19, "name": "train", "supercategory": "VOC"}, {"id": 20, "name": "tvmonitor", "supercategory": "VOC"}]
categories_pistol = [{"id": 1, "name": "pistol", "supercategory": "Guns"}]
categories_pothole = [{"id": 1, "name": "pothole", "supercategory": "potholes"}]
categories_raccoon = [{"id": 1, "name": "raccoon", "supercategory": "raccoons"}]
categories_shellfish = [{"id": 1, "name": "Crab", "supercategory": "shellfish"}, {"id": 2, "name": "Lobster", "supercategory": "shellfish"}, {"id": 3, "name": "Shrimp", "supercategory": "shellfish"}]
categories_thermal = [{"id": 1, "name": "dog", "supercategory": "dogs-person"}, {"id": 2, "name": "person", "supercategory": "dogs-person"}]
categories_vehicle = [{"id": 1, "name": "Ambulance", "supercategory": "vehicles"}, {"id": 2, "name": "Bus", "supercategory": "vehicles"}, {"id": 3, "name": "Car", "supercategory": "vehicles"}, {"id": 4, "name": "Motorcycle", "supercategory": "vehicles"}, {"id": 5, "name": "Truck", "supercategory": "vehicles"}]

id_map_aerial = {0:1, 1:2, 2:3, 3:4, 4:5}
id_map_aqua = {0:1, 1:2, 2:3, 3:4, 4:5, 5:6, 6:7}
id_map_rabbit = {0:1}
id_map_egohand = {0:1}
id_map_mushroom = {0:1, 1:2}
id_map_package = {0:1}
id_map_voc = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 11, 11: 12, 12: 13, 13: 14, 14: 15, 15: 16, 16: 17, 17: 18, 18: 19, 19: 20}
id_map_pistol = {0:1}
id_map_pothole = {0:1}
id_map_raccoon = {0:1}
id_map_shellfish = {0:1, 1:2, 2:3}
id_map_thermal = {0:1, 1:2}
id_map_vehicle = {0:1, 1:2, 2:3, 3:4, 4:5}
id_map_coco = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 11, 11: 13, 12: 14, 13: 15, 14: 16, 15: 17, 16: 18, 17: 19, 18: 20, 19: 21, 20: 22, 21: 23, 22: 24, 23: 25, 24: 27, 25: 28, 26: 31, 27: 32, 28: 33, 29: 34, 30: 35, 31: 36, 32: 37, 33: 38, 34: 39, 35: 40, 36: 41, 37: 42, 38: 43, 39: 44, 40: 46,
                  41: 47, 42: 48, 43: 49, 44: 50, 45: 51, 46: 52, 47: 53, 48: 54, 49: 55, 50: 56, 51: 57, 52: 58, 53: 59, 54: 60, 55: 61, 56: 62, 57: 63, 58: 64, 59: 65, 60: 67, 61: 70, 62: 72, 63: 73, 64: 74, 65: 75, 66: 76, 67: 77, 68: 78, 69: 79, 70: 80, 71: 81, 72: 82, 73: 84, 74: 85, 75: 86, 76: 87, 77: 88, 78: 89, 79: 90}

class PostProcessCocoGrounding(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""

    def __init__(self, num_select=300, coco_api=None, tokenlizer=None) -> None:
        super().__init__()
        self.num_select = num_select

        assert coco_api is not None
        # category_dict = coco_api.dataset['categories']
        category_dict = categories_pistol + categories_aerial + categories_aqua + categories_rabbit + categories_egohand + categories_mushroom + categories_package + categories_voc +  categories_pothole + categories_raccoon + categories_shellfish + categories_thermal + categories_vehicle
    

        cat_list = [item['name'] for item in category_dict]
        # print(cat_list)
        captions, cat2tokenspan = build_captions_and_token_span(cat_list, True)
        # print(captions)
        # print(cat2tokenspan)
        # exit()
        tokenspanlist = [cat2tokenspan[cat.lower()] for cat in cat_list]
        positive_map = create_positive_map_from_span(
            tokenlizer(captions), tokenspanlist, max_text_len=4500)  # 80, 256. normed
        
        id_map = {} 
        max_real_id = 0

        for key in id_map_pistol.keys():
            id_map[key] = id_map_pistol[key] 
            max_real_id = max(max_real_id, id_map[key])
        offset = len(id_map.keys())
        
        
        for key in id_map_aerial.keys():
            id_map[key+offset] = id_map_aerial[key] + offset
            max_real_id = max(max_real_id, id_map[key+offset])
        offset = len(id_map.keys())
        for key in id_map_aqua.keys():
            id_map[key+offset] = id_map_aqua[key] + offset
            max_real_id = max(max_real_id, id_map[key+offset])
        offset = len(id_map.keys())
        for key in id_map_rabbit.keys():
            id_map[key+offset] = id_map_rabbit[key] + offset
            max_real_id = max(max_real_id, id_map[key+offset])
        offset = len(id_map.keys())
        for key in id_map_egohand.keys():
            id_map[key+offset] = id_map_egohand[key] + offset
            max_real_id = max(max_real_id, id_map[key+offset])
        offset = len(id_map.keys())
        for key in id_map_mushroom.keys():
            id_map[key+offset] = id_map_mushroom[key] + offset
            max_real_id = max(max_real_id, id_map[key+offset])
        offset = len(id_map.keys())
        for key in id_map_package.keys():
            id_map[key+offset] = id_map_package[key] + offset
            max_real_id = max(max_real_id, id_map[key+offset])
        offset = len(id_map.keys())
        for key in id_map_voc.keys():
            id_map[key+offset] = id_map_voc[key] + offset
            max_real_id = max(max_real_id, id_map[key+offset])
        offset = len(id_map.keys())

        for key in id_map_pothole.keys():
            id_map[key+offset] = id_map_pothole[key] + offset
            max_real_id = max(max_real_id, id_map[key+offset])
        offset = len(id_map.keys())
        for key in id_map_raccoon.keys():
            id_map[key+offset] = id_map_raccoon[key] + offset
            max_real_id = max(max_real_id, id_map[key+offset])
        offset = len(id_map.keys())
        for key in id_map_shellfish.keys():
            id_map[key+offset] = id_map_shellfish[key] + offset
            max_real_id = max(max_real_id, id_map[key+offset])
        offset = len(id_map.keys())
        for key in id_map_thermal.keys():
            id_map[key+offset] = id_map_thermal[key] + offset
            max_real_id = max(max_real_id, id_map[key+offset])
        offset = len(id_map.keys())
        for key in id_map_vehicle.keys():
            id_map[key+offset] = id_map_vehicle[key] + offset
            max_real_id = max(max_real_id, id_map[key+offset])


        # build a mapping from label_id to pos_map
        # new_pos_map = torch.zeros((91, 256))
        new_pos_map = torch.zeros((max_real_id+1, 4500))
        for k, v in id_map.items():
            new_pos_map[v] = positive_map[k]
        self.positive_map = new_pos_map

    @torch.no_grad()
    def forward(self, outputs, target_sizes, not_to_xyxy=False):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        num_select = self.num_select
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        # pos map to logit
        prob_to_token = out_logits.sigmoid()  # bs, 100, 256
        pos_maps = self.positive_map.to(prob_to_token.device)
        # (bs, 100, 256) @ (91, 256).T -> (bs, 100, 91)
        prob_to_label = prob_to_token @ pos_maps.T

        # if os.environ.get('IPDB_SHILONG_DEBUG', None) == 'INFO':
        #     import ipdb; ipdb.set_trace()

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = prob_to_label
        topk_values, topk_indexes = torch.topk(
            prob.view(out_logits.shape[0], -1), num_select, dim=1)
        scores = topk_values
        topk_boxes = topk_indexes // prob.shape[2]
        labels = topk_indexes % prob.shape[2]

        if not_to_xyxy:
            boxes = out_bbox
        else:
            boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)

        boxes = torch.gather(
            boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b}
                   for s, l, b in zip(scores, labels, boxes)]

        return results


def main(args):
    # config
    cfg = SLConfig.fromfile(args.config_file)

    # build model
    model = load_model(args, args.checkpoint_path)

    model = model.to(args.device)
    model = model.eval()

    # build dataloader
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    dataset = CocoDetection(
        args.image_dir, args.anno_path, transforms=transform)
    data_loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)

    # build post processor
    tokenlizer = get_tokenlizer.get_tokenlizer(cfg.text_encoder_type)
    postprocessor = PostProcessCocoGrounding(
        coco_api=dataset.coco, tokenlizer=tokenlizer)

    # build evaluator
    evaluator = CocoGroundingEvaluator(
        dataset.coco, iou_types=("bbox",), useCats=True)

    # build captions
    # category_dict = dataset.coco.dataset['categories']
    category_dict =  categories_pistol + categories_aerial + categories_aqua + categories_rabbit + categories_egohand + categories_mushroom + categories_package + categories_voc + categories_pothole + categories_raccoon + categories_shellfish + categories_thermal + categories_vehicle
    cat_list = [item['name'] for item in category_dict]
    # cat_list = [item['name'] for item in categories_aerial] + [item['name'] for item in categories_coco]
    caption = " . ".join(cat_list) + ' .'
    print("Input text prompt:", caption)


    legal_task_id = ['aerial','aqua', 'cotton', 'egohand', 'mushroom', 'package', 'pascalvoc','pistol', 'pothole', 'raccoon',   'shellfish', 'thermal',   'vehicle']
    
    with open("subtasks_prompt_wo_coco_10_shot.pkl", 'rb') as f:
        subtasks_prompt_data = pickle.load(f)
        
    with open("subtasks_lora_wo_coco_10_shot.pkl", 'rb') as f:
        subtasks_lora_data = pickle.load(f)
        
        
        
    with open("task_image_feat_bank_10_shot/mean_feat/task_feats.pkl", 'rb') as f:
        subtasks_mean_feat = pickle.load(f)
        
    
    subtasks_prompt_data_new = {}
    subtasks_lora_data_new = {}
    subtasks_mean_feat_new = {}
    
    for key in legal_task_id:
        subtasks_prompt_data_new[key] = subtasks_prompt_data[key]
        subtasks_mean_feat_new[key] = subtasks_mean_feat[key]
        subtasks_lora_data_new[key] = subtasks_lora_data[key]
    
    subtasks_prompt_data = subtasks_prompt_data_new
    subtasks_mean_feat = subtasks_mean_feat_new
    subtasks_lora_data = subtasks_lora_data_new
    
    
    # run inference
    start = time.time()
    for i, (images, targets) in enumerate(data_loader):
        # get images and captions
        images = images.tensors.to(args.device)
        bs = images.shape[0]
    
        input_captions = [caption] * bs
        

        # feed to the model
        with torch.no_grad():
            if args.use_retrieve:
                outputs = model(images, captions=input_captions,subtasks_prompts_list=subtasks_prompt_data, subtasks_lora_data = subtasks_lora_data, subtasks_mean_feat=subtasks_mean_feat)
            else:
                if not args.use_zira:
                    outputs = model(images, captions=input_captions)
                else:
                    outputs, _, _ = model(images, captions=input_captions)
        
        # outputs = model(images, captions=input_captions)

        orig_target_sizes = torch.stack(
            [t["orig_size"] for t in targets], dim=0).to(images.device)
        results = postprocessor(outputs, orig_target_sizes)
  
        cocogrounding_res = {
            target["image_id"]: output for target, output in zip(targets, results)}
        evaluator.update(cocogrounding_res)

        if (i+1) % 30 == 0:
            used_time = time.time() - start
            eta = len(data_loader) / (i+1e-5) * used_time - used_time
            print(
                f"processed {i}/{len(data_loader)} images. time: {used_time:.2f}s, ETA: {eta:.2f}s")

    evaluator.synchronize_between_processes()
    evaluator.accumulate()
    evaluator.summarize()

    print("Final results:", evaluator.coco_eval["bbox"].stats.tolist())

    os.makedirs(args.save_path, exist_ok=True)

        
    with open(os.path.join(args.save_path, "results.json"), 'w') as f:
        json.dump({'result':evaluator.coco_eval["bbox"].stats.tolist()}, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Grounding DINO eval on COCO", add_help=True)
    # load model
    parser.add_argument("--config_file", "-c", type=str,
                        required=True, help="path to config file")
    parser.add_argument(
        "--checkpoint_path", "-p", type=str, required=True, help="path to checkpoint file"
    )
    parser.add_argument("--device", type=str, default="cuda",
                        help="running device (default: cuda)")

    # post processing
    parser.add_argument("--num_select", type=int, default=300,
                        help="number of topk to select")

    # coco info
    parser.add_argument("--anno_path", type=str,
                        required=True, help="coco root")
    parser.add_argument("--image_dir", type=str,
                        required=True, help="coco image dir")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="number of workers for dataloader")
    parser.add_argument("--coco_val_path", type=str, 
                        help="number of workers for dataloader")  
    parser.add_argument("--use_coop",  action='store_true',
                        help="number of workers for dataloader")  
    parser.add_argument("--use_retrieve",  action='store_true',
                        help="number of workers for dataloader")  
    parser.add_argument("--use_moe_lora",  action='store_true',
                        help="number of workers for dataloader")
    parser.add_argument("--use_adapter",  action='store_true',
                        help="number of workers for dataloader")
    parser.add_argument("--use_prompt",  action='store_true',
                        help="number of workers for dataloader") 
    parser.add_argument("--use_zira",  action='store_true',
                        help="number of workers for dataloader")  
    parser.add_argument("--save_path",  type=str,
                        help="number of workers for dataloader")   
    args = parser.parse_args()

    main(args)

