import argparse
import os
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import sys
import pickle
# please make sure https://github.com/IDEA-Research/GroundingDINO is installed correctly.
import groundingdino.datasets.transforms as T
# from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from groundingdino.util.vl_utils import create_positive_map_from_span, build_captions_and_token_span
from torchvision.ops import nms
import copy

PALETTE = [[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
               [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
               [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
               [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
               [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
               [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
               [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
               [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
               [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
               [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
               [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
               [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
               [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0],
               [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
               [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255],
               [11, 200, 200], [255, 82, 0], [0, 255, 245], [0, 61, 255],
               [0, 255, 112], [0, 255, 133], [255, 0, 0], [255, 163, 0],
               [255, 102, 0], [194, 255, 0], [0, 143, 255], [51, 255, 0],
               [0, 82, 255], [0, 255, 41], [0, 255, 173], [10, 0, 255],
               [173, 255, 0], [0, 255, 153], [255, 92, 0], [255, 0, 255],
               [255, 0, 245], [255, 0, 102], [255, 173, 0], [255, 0, 20],
               [255, 184, 184], [0, 31, 255], [0, 255, 61], [0, 71, 255],
               [255, 0, 204], [0, 255, 194], [0, 255, 82], [0, 10, 255],
               [0, 112, 255], [51, 0, 255], [0, 194, 255], [0, 122, 255],
               [0, 255, 163], [255, 153, 0], [0, 255, 10], [255, 112, 0],
               [143, 255, 0], [82, 0, 255], [163, 255, 0], [255, 235, 0],
               [8, 184, 170], [133, 0, 255], [0, 255, 92], [184, 0, 255],
               [255, 0, 31], [0, 184, 255], [0, 214, 255], [255, 0, 112],
               [92, 255, 0], [0, 224, 255], [112, 224, 255], [70, 184, 160],
               [163, 0, 255], [153, 0, 255], [71, 255, 0], [255, 0, 163],
               [255, 204, 0], [255, 0, 143], [0, 255, 235], [133, 255, 0],
               [255, 0, 235], [245, 0, 255], [255, 0, 122], [255, 245, 0],
               [10, 190, 212], [214, 255, 0], [0, 204, 255], [20, 0, 255],
               [255, 255, 0], [0, 153, 255], [0, 41, 255], [0, 255, 204],
               [41, 0, 255], [41, 255, 0], [173, 0, 255], [0, 245, 255],
               [71, 0, 255], [122, 0, 255], [0, 255, 184], [0, 92, 255],
               [184, 255, 0], [0, 133, 255], [255, 214, 0], [25, 194, 194],
               [102, 255, 0], [92, 0, 255]]



def center_to_corner_batch(bboxes):
    # bboxes 是形状为 (N, 4) 的tensor，每一行是 [x, y, w, h]
    x, y, w, h = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
    
    # 计算左上角和右下角坐标
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2
    
    # 拼接成新的tensor，形状为 (N, 4)
    return torch.stack([x1, y1, x2, y2], dim=1)

         


def plot_boxes_to_image(image_pil, tgt, text_2_color_map):
    H, W = tgt["size"]
    boxes = tgt["boxes"]
    labels = tgt["labels"]
    assert len(boxes) == len(labels), "boxes and labels must have same length"

    draw = ImageDraw.Draw(image_pil)
    mask = Image.new("L", image_pil.size, 0)
    mask_draw = ImageDraw.Draw(mask)

    # Load a larger font (replace with a suitable font path if needed)
    font_size = 20  # Adjust the font size as needed
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)  # You can change to any available font
    except IOError:
        font = ImageFont.load_default()

    # Draw boxes and labels
    for box, label in zip(boxes, labels):
        # from 0..1 to 0..W, 0..H
        labal_name = label.split("(")[0]
        
        box = box * torch.Tensor([W, H, W, H])
        # from xywh to xyxy
        box[:2] -= box[2:] / 2
        box[2:] += box[:2]
        # random color
        # color = tuple(np.random.randint(0, 255, size=3).tolist())
        color = tuple(text_2_color_map[labal_name])
        # draw
        x0, y0, x1, y1 = map(int, box)

        draw.rectangle([x0, y0, x1, y1], outline=color, width=6)

        if hasattr(font, "getbbox"):
            bbox = draw.textbbox((x0, y0), str(label), font)
        else:
            w, h = draw.textsize(str(label), font)
            bbox = (x0, y0, w + x0, y0 + h)

        draw.rectangle(bbox, fill=color)
        draw.text((x0, y0), str(label), fill="white", font=font)

        mask_draw.rectangle([x0, y0, x1, y1], fill=255, width=6)

    return image_pil, mask



def load_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image


def build_model(args):
    # we use register to maintain models from catdet6 on.
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from models.registry import MODULE_BUILD_FUNCS

    assert args.modelname in MODULE_BUILD_FUNCS._module_dict
    build_func = MODULE_BUILD_FUNCS.get(args.modelname)
    model = build_func(args)
    return model



def load_model(args_origin, model_checkpoint_path, cpu_only=False):
    model_config_path = args_origin.config_file
    args = SLConfig.fromfile(model_config_path)
    args.device = "cuda" if not cpu_only else "cpu"
    args.datasets = 'odinw'
    args.coco_val_path = args_origin.coco_val_path
    args.use_coop = args_origin.use_coop
    args.use_moe_lora = args_origin.use_moe_lora
    args.use_adapter = args_origin.use_adapter
    args.use_prompt = args_origin.use_prompt
    args.use_zira = args_origin.use_zira
    # model = build_model(args)
    
    model,_,_ = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    if args_origin.use_double_model:
        args.use_coop = False
        args.use_moe_lora = False
        model_base,_,_ = build_model(args)
        checkpoint = torch.load(args_origin.base_model_path, map_location="cpu")
        load_res = model_base.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        print(load_res)
        _ = model_base.eval()
        return model, model_base
    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold=None, with_logits=True, cpu_only=False, token_spans=None, subtasks_prompt_data=None, subtasks_lora_data = None, subtasks_mean_feat=None, tau=None) :
    assert text_threshold is not None or token_spans is not None, "text_threshould and token_spans should not be None at the same time!"
    caption = caption.lower()
    caption = caption.strip()
    # print(len(caption.split(".")), caption.split("."))
    
    # # print(len(caption),caption)
    # exit()
    if not caption.endswith("."):
        caption = caption + "."
    device = "cuda" if not cpu_only else "cpu"
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        if not args.use_zira:
            outputs = model(image[None], captions=[caption], subtasks_prompts_list=subtasks_prompt_data, subtasks_lora_data = subtasks_lora_data, subtasks_mean_feat=subtasks_mean_feat, tau=tau)
        else:
            outputs,_,_ = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"][0]  # (nq, 4)

    # filter output
    if token_spans is None:
        logits_filt = logits.cpu().clone()
        boxes_filt = boxes.cpu().clone()
        filt_mask = logits_filt.max(dim=1)[0] > box_threshold
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4

        # get phrase
        tokenlizer = model.tokenizer
        tokenized = tokenlizer(caption)
        # build pred
        pred_phrases = []
        for logit, box in zip(logits_filt, boxes_filt):

            pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
            if with_logits:
                pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
            else:
                pred_phrases.append(pred_phrase)
    else:
        # given-phrase mode
        positive_maps = create_positive_map_from_span(
            model.tokenizer(text_prompt),
            token_span=token_spans,
            max_text_len=4500
            
        ).to(image.device) # n_phrase, 256

        logits_for_phrases = positive_maps @ logits.T # n_phrase, nq
        all_logits = []
        all_phrases = []
        all_boxes = []
        for (token_span, logit_phr) in zip(token_spans, logits_for_phrases):
            # get phrase
            phrase = ' '.join([caption[_s:_e] for (_s, _e) in token_span])
            # get mask
            filt_mask = logit_phr > box_threshold
            # filt box
            all_boxes.append(boxes[filt_mask])
            # filt logits
            all_logits.append(logit_phr[filt_mask])
            if with_logits:
                logit_phr_num = logit_phr[filt_mask]
                all_phrases.extend([phrase + f"({str(logit.item())[:4]})" for logit in logit_phr_num])
            else:
                all_phrases.extend([phrase for _ in range(len(filt_mask))])
        boxes_filt = torch.cat(all_boxes, dim=0).cpu()
        logit_filt = torch.cat(all_logits, dim=0).cpu()
        pred_phrases = all_phrases


    return boxes_filt, pred_phrases, logit_filt


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Grounding DINO example", add_help=True)
    parser.add_argument("--config_file", "-c", type=str, required=True, help="path to config file")
    parser.add_argument(
        "--checkpoint_path", "-p", type=str, required=True, help="path to checkpoint file"
    )
    parser.add_argument("--image_path", "-i", type=str, required=True, help="path to image file")
    parser.add_argument("--text_prompt", "-t", type=str, required=True, help="text prompt")
    parser.add_argument(
        "--output_dir", "-o", type=str, default="outputs", required=True, help="output directory"
    )

    parser.add_argument("--box_threshold", type=float, default=0.36, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="text threshold")
    parser.add_argument("--token_spans", type=str, default=None, help=
                        "The positions of start and end positions of phrases of interest. \
                        For example, a caption is 'a cat and a dog', \
                        if you would like to detect 'cat', the token_spans should be '[[[2, 5]], ]', since 'a cat and a dog'[2:5] is 'cat'. \
                        if you would like to detect 'a cat', the token_spans should be '[[[0, 1], [2, 5]], ]', since 'a cat and a dog'[0:1] is 'a', and 'a cat and a dog'[2:5] is 'cat'. \
                        ")

    parser.add_argument("--cpu-only", action="store_true", help="running on cpu only!, default=False")
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
    parser.add_argument("--use_double_model",  action='store_true',
                        help="number of workers for dataloader")  
    parser.add_argument("--base_model_path",  type=str,
                        help="number of workers for dataloader")  
    parser.add_argument("--coco_val_path", type=str, 
                        help="number of workers for dataloader") 
    parser.add_argument("--retrieval_tau", type=float, default=0.89,
                        help="number of workers for dataloader") 
    args = parser.parse_args()

    # cfg
    config_file = args.config_file  # change the path of the model config file
    checkpoint_path = args.checkpoint_path  # change the path of the model
    image_path = args.image_path
    text_prompt = args.text_prompt
    output_dir = args.output_dir
    box_threshold = args.box_threshold
    text_threshold = args.text_threshold
    # token_spans = args.token_spans
    
    class_name = text_prompt.split(".")
    
    for i in range(len(class_name)):
        class_name[i] = class_name[i].strip()
    class_name = class_name[:-1]
    text_2_color_map = {}
    
    for i in range(len(class_name)):
        text_2_color_map[class_name[i]] = PALETTE[i]
    
    captions, cat2tokenspan = build_captions_and_token_span(class_name, True)
    

    # print(captions)
    # print(cat2tokenspan)
    # exit()
    token_spans = [cat2tokenspan[cat.lower()] for cat in class_name]

    # make dir
    os.makedirs(output_dir, exist_ok=True)
    # load image
    image_pil, image = load_image(image_path)
    # load model
    if not args.use_double_model:
        model = load_model(args, checkpoint_path, cpu_only=args.cpu_only)
    else:
        model, model_base = load_model(args, checkpoint_path, cpu_only=args.cpu_only)

    # visualize raw image
    image_pil.save(os.path.join(output_dir, "raw_image.jpg"))

    # set the text_threshold to None if token_spans is set.
    if token_spans is not None:
        text_threshold = None
        print("Using token_spans. Set the text_threshold to None.")
        
        
    # print(text_prompt)

    with open("multi_model_prompt_params.pkl", 'rb') as f:
        subtasks_prompt_data = pickle.load(f)
        
    with open("multi_model_lora_params.pkl", 'rb') as f:
        subtasks_lora_data = pickle.load(f)
        
    # print(subtasks_prompt_data.keys())
        
    with open("FEATRURE_SAVE_ROOT/mean_feat/task_feats.pkl", 'rb') as f:
        subtasks_mean_feat = pickle.load(f)

    # run model
    boxes_filt, pred_phrases, logit_filt = get_grounding_output(
        model, 
        image, 
        text_prompt, 
        box_threshold, 
        text_threshold, 
        cpu_only=args.cpu_only, 
        token_spans=token_spans,
        subtasks_prompt_data=subtasks_prompt_data, 
        subtasks_lora_data = subtasks_lora_data, 
        subtasks_mean_feat=subtasks_mean_feat,
        tau = args.retrieval_tau
    )
    


    # visualize pred
    size = image_pil.size
    pred_dict = {
        "boxes": boxes_filt,
        "size": [size[1], size[0]],  # H,W
        "labels": pred_phrases,
        "scores": logit_filt
    }

    if args.use_double_model:
        boxes_filt_base, pred_phrases_base, logit_filt_base = get_grounding_output(
            model_base, image, text_prompt, box_threshold, text_threshold, cpu_only=args.cpu_only, token_spans=token_spans
        )    
        pred_dict_base = {
            "boxes": boxes_filt_base,
            "size": [size[1], size[0]],  # H,W
            "labels": pred_phrases_base,
            "scores": logit_filt_base
        }
        
        bboxes_all = torch.cat([boxes_filt, boxes_filt_base], dim = 0)
        
        logits_all = torch.cat([logit_filt, logit_filt_base], dim = 0)
        
        
        pred_phrases_all = copy.deepcopy(pred_phrases)
        
        pred_phrases_all.extend(pred_phrases_base)
        
        iou_threshold = 0.5
        

        bboxes_all_transform = center_to_corner_batch(bboxes_all)
        selected_indices = nms(bboxes_all_transform, logits_all, iou_threshold)
        
        # print(selected_indices)
        
        
        selected_boxes = bboxes_all[selected_indices]
        # print(selected_indices)
        # exit()
        
        new_phrases = [pred_phrases_all[idx] for idx in selected_indices]
        
        pred_dict = {
            "boxes": selected_boxes,
            "size": [size[1], size[0]],  # H,W
            "labels": new_phrases,
        }
        
 
    # print(class_name)
    # exit()
    image_with_box = plot_boxes_to_image(image_pil, pred_dict, text_2_color_map)[0]
    save_path = os.path.join(output_dir, "pred.jpg")
    image_with_box.save(save_path)
    print(f"\n======================\n{save_path} saved.\nThe program runs successfully!")
