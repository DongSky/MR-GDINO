import argparse
import jsonlines
from tqdm import tqdm
import json
from pycocotools.coco import COCO

# this id_map is only for coco dataset which has 80 classes used for training but 90 categories in total.
# which change the start label -> 0
# {"0": "person", "1": "bicycle", "2": "car", "3": "motorcycle", "4": "airplane", "5": "bus", "6": "train", "7": "truck", "8": "boat", "9": "traffic light", "10": "fire hydrant", "11": "stop sign", "12": "parking meter", "13": "bench", "14": "bird", "15": "cat", "16": "dog", "17": "horse", "18": "sheep", "19": "cow", "20": "elephant", "21": "bear", "22": "zebra", "23": "giraffe", "24": "backpack", "25": "umbrella", "26": "handbag", "27": "tie", "28": "suitcase", "29": "frisbee", "30": "skis", "31": "snowboard", "32": "sports ball", "33": "kite", "34": "baseball bat", "35": "baseball glove", "36": "skateboard", "37": "surfboard", "38": "tennis racket", "39": "bottle", "40": "wine glass", "41": "cup", "42": "fork", "43": "knife", "44": "spoon", "45": "bowl", "46": "banana", "47": "apple", "48": "sandwich", "49": "orange", "50": "broccoli", "51": "carrot", "52": "hot dog", "53": "pizza", "54": "donut", "55": "cake", "56": "chair", "57": "couch", "58": "potted plant", "59": "bed", "60": "dining table", "61": "toilet", "62": "tv", "63": "laptop", "64": "mouse", "65": "remote", "66": "keyboard", "67": "cell phone", "68": "microwave", "69": "oven", "70": "toaster", "71": "sink", "72": "refrigerator", "73": "book", "74": "clock", "75": "vase", "76": "scissors", "77": "teddy bear", "78": "hair drier", "79": "toothbrush"}

id_map = {0:1, 1:2, 2:3, 3:4, 4:5}

# id_map_aerial = {0:1, 1:2, 2:3, 3:4, 4:5}
# id_map_aqua = {0:1, 1:2, 2:3, 3:4, 4:5, 5:6, 6:7}
# id_map_rabbit = {0:1}
# id_map_egohand = {0:1}
# id_map_mushroom = {0:1, 1:2}
# id_map_package = {0:1}
# id_map_voc = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 11, 11: 12, 12: 13, 13: 14, 14: 15, 15: 16, 16: 17, 17: 18, 18: 19, 19: 20}
# id_map_pistol = {0:1}
# id_map_pothole = {0:1}
# id_map_raccoon = {0:1}
# id_map_shellfish = {0:1, 1:2, 2:3}
# id_map_thermal = {0:1, 1:2}
# id_map_vehicle = {0:1, 1:2, 2:3, 3:4, 4:5}
key_list=list(id_map.keys())
val_list=list(id_map.values())

def dump_label_map(output="./out.json"):
    ori_map = {"1": "person", "2": "bicycle", "3": "car", "4": "motorcycle", "5": "airplane", "6": "bus", "7": "train", "8": "truck", "9": "boat", "10": "traffic light", "11": "fire hydrant", "13": "stop sign", "14": "parking meter", "15": "bench", "16": "bird", "17": "cat", "18": "dog", "19": "horse", "20": "sheep", "21": "cow", "22": "elephant", "23": "bear", "24": "zebra", "25": "giraffe", "27": "backpack", "28": "umbrella", "31": "handbag", "32": "tie", "33": "suitcase", "34": "frisbee", "35": "skis", "36": "snowboard", "37": "sports ball", "38": "kite", "39": "baseball bat", "40": "baseball glove", "41": "skateboard", "42": "surfboard", "43": "tennis racket", "44": "bottle", "46": "wine glass", "47": "cup", "48": "fork", "49": "knife", "50": "spoon", "51": "bowl", "52": "banana", "53": "apple", "54": "sandwich", "55": "orange", "56": "broccoli", "57": "carrot", "58": "hot dog", "59": "pizza", "60": "donut", "61": "cake", "62": "chair", "63": "couch", "64": "potted plant", "65": "bed", "67": "dining table", "70": "toilet", "72": "tv", "73": "laptop", "74": "mouse", "75": "remote", "76": "keyboard", "77": "cell phone", "78": "microwave", "79": "oven", "80": "toaster", "81": "sink", "82": "refrigerator", "84": "book", "85": "clock", "86": "vase", "87": "scissors", "88": "teddy bear", "89": "hair drier", "90": "toothbrush"}
    new_map = {}
    for key, value in ori_map.items():
        label = int(key)
        ind=val_list.index(label)
        label_trans = key_list[ind]
        new_map[label_trans] = value
    with open(output,"w") as f:
        json.dump(new_map, f)

def coco_to_xyxy(bbox):
    x, y, width, height = bbox
    x1 = round(x, 2) 
    y1 = round(y, 2)
    x2 = round(x + width, 2)
    y2 = round(y + height, 2)
    return [x1, y1, x2, y2]


def coco2odvg(args):
    coco = COCO(args.input) 
    cats = coco.loadCats(coco.getCatIds())
    nms = {cat['id']:cat['name'] for cat in cats}
    metas = []

    for img_id, img_info in tqdm(coco.imgs.items()):
        ann_ids = coco.getAnnIds(imgIds=img_id)
        instance_list = []
        for ann_id in ann_ids:
            ann = coco.anns[ann_id]
            bbox = ann['bbox']
            bbox_xyxy = coco_to_xyxy(bbox)
            label = ann['category_id']
            category = nms[label]
            ind=val_list.index(label)
            label_trans = key_list[ind]
            instance_list.append({
                "bbox": bbox_xyxy,
                "label": label_trans,
                "category": category
                }
            )
        metas.append(
            {
                "filename": img_info["file_name"],
                "height": img_info["height"],
                "width": img_info["width"],
                "detection": {
                    "instances": instance_list
                }
            }
        )
    print("  == dump meta ...")
    with jsonlines.open(args.output, mode="w") as writer:
        writer.write_all(metas)
    print("  == done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("coco to odvg format.", add_help=True)
    parser.add_argument("--input", '-i', required=True, type=str, help="input list name")
    parser.add_argument("--output", '-o', required=True, type=str, help="output list name")
    args = parser.parse_args()

    coco2odvg(args)
