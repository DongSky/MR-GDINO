import argparse
import os
import sys
import time
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler
from pycocotools import mask as mask_utils
# from groundingdino.models import build_model
import groundingdino.datasets.transforms as T
from groundingdino.util import box_ops, get_tokenlizer
from groundingdino.util.misc import clean_state_dict, collate_fn
from groundingdino.util.slconfig import SLConfig
import pickle
# from torchvision.datasets import CocoDetection
import torchvision
from collections import defaultdict
from groundingdino.util.vl_utils import build_captions_and_token_span, create_positive_map_from_span
from groundingdino.datasets.cocogrounding_eval import CocoGroundingEvaluator
from datasets.lvis_eval import LvisEvaluator
from PIL import Image
from datasets.lvis import LVIS



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
    log = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    # print(log)
    # exit()
    model.eval()
    return model


# class CocoDetection(torchvision.datasets.CocoDetection):
#     def __init__(self, img_folder, ann_file, transforms):
#         super().__init__(img_folder, ann_file)
#         self._transforms = transforms

#     def __getitem__(self, idx):
#         img, target = super().__getitem__(idx)  # target: list

#         # import ipdb; ipdb.set_trace()

#         w, h = img.size
#         boxes = [obj["bbox"] for obj in target]
#         boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
#         boxes[:, 2:] += boxes[:, :2]  # xywh -> xyxy
#         boxes[:, 0::2].clamp_(min=0, max=w)
#         boxes[:, 1::2].clamp_(min=0, max=h)
#         # filt invalid boxes/masks/keypoints
#         keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
#         boxes = boxes[keep]

#         target_new = {}
#         image_id = self.ids[idx]
#         target_new["image_id"] = image_id
#         target_new["boxes"] = boxes
#         target_new["orig_size"] = torch.as_tensor([int(h), int(w)])

#         if self._transforms is not None:
#             img, target = self._transforms(img, target_new)

#         return img, target




    
    
class LvisDetectionBase(torchvision.datasets.VisionDataset):
    def __init__(self, root, annFile, transform=None, target_transform=None, transforms=None):
        super(LvisDetectionBase, self).__init__(root, transforms, transform, target_transform)
        self.lvis = LVIS(annFile)
        self.ids = list(sorted(self.lvis.imgs.keys()))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        lvis = self.lvis
        img_id = self.ids[index]
        ann_ids = lvis.get_ann_ids(img_ids=img_id)
        target = lvis.load_anns(ann_ids)

        path = "/".join(self.lvis.load_imgs(img_id)[0]["coco_url"].split("/")[-2:])

        img = Image.open(os.path.join(self.root, path)).convert("RGB")
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target
    

    def __len__(self):
        return len(self.ids)



class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target
    
    
class LvisDetection(LvisDetectionBase):
    def __init__(self, img_folder, ann_file, transforms, return_masks=False, **kwargs):
        super(LvisDetection, self).__init__(img_folder, ann_file)
        self.ann_file = ann_file
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)

    def __getitem__(self, idx):
        img, target = super(LvisDetection, self).__getitem__(idx)

        image_id = self.ids[idx]
        target = {"image_id": image_id, "annotations": target}
        img, target = self.prepare(img, target)
        
        target_new = {}
        image_id = self.ids[idx]
        target_new["image_id"] = image_id
        target_new["boxes"] = target['boxes']
        target_new["orig_size"] = target['orig_size']
        
        
   
        if self._transforms is not None:
            img, target = self._transforms(img, target_new)
        # print(image.shape, target_new)
        return img, target
    
    def get_raw_image(self, idx):
        img, target = super(LvisDetection, self).__getitem__(idx)
        return img
    
    def categories(self):
        id2cat = {c["id"]: c for c in self.lvis.dataset["categories"]}
        all_cats = sorted(list(id2cat.keys()))
        categories = {}
        for l in list(all_cats):
            categories[l] = id2cat[l]['name']
        return categories
    
    

categories_lvis = [{'id': 1, 'name': 'aerosol_can'}, {'id': 2, 'name': 'air_conditioner'}, {'id': 3, 'name': 'airplane'}, {'id': 4, 'name': 'alarm_clock'}, {'id': 5, 'name': 'alcohol'}, {'id': 6, 'name': 'alligator'}, {'id': 7, 'name': 'almond'}, {'id': 8, 'name': 'ambulance'}, {'id': 9, 'name': 'amplifier'}, {'id': 10, 'name': 'anklet'}, {'id': 11, 'name': 'antenna'}, {'id': 12, 'name': 'apple'}, {'id': 13, 'name': 'applesauce'}, {'id': 14, 'name': 'apricot'}, {'id': 15, 'name': 'apron'}, {'id': 16, 'name': 'aquarium'}, {'id': 17, 'name': 'arctic_(type_of_shoe)'}, {'id': 18, 'name': 'armband'}, {'id': 19, 'name': 'armchair'}, {'id': 20, 'name': 'armoire'}, {'id': 21, 'name': 'armor'}, {'id': 22, 'name': 'artichoke'}, {'id': 23, 'name': 'trash_can'}, {'id': 24, 'name': 'ashtray'}, {'id': 25, 'name': 'asparagus'}, {'id': 26, 'name': 'atomizer'}, {'id': 27, 'name': 'avocado'}, {'id': 28, 'name': 'award'}, {'id': 29, 'name': 'awning'}, {'id': 30, 'name': 'ax'}, {'id': 31, 'name': 'baboon'}, {'id': 32, 'name': 'baby_buggy'}, {'id': 33, 'name': 'basketball_backboard'}, {'id': 34, 'name': 'backpack'}, {'id': 35, 'name': 'handbag'}, {'id': 36, 'name': 'suitcase'}, {'id': 37, 'name': 'bagel'}, {'id': 38, 'name': 'bagpipe'}, {'id': 39, 'name': 'baguet'}, {'id': 40, 'name': 'bait'}, {'id': 41, 'name': 'ball'}, {'id': 42, 'name': 'ballet_skirt'}, {'id': 43, 'name': 'balloon'}, {'id': 44, 'name': 'bamboo'}, {'id': 45, 'name': 'banana'}, {'id': 46, 'name': 'Band_Aid'}, {'id': 47, 'name': 'bandage'}, {'id': 48, 'name': 'bandanna'}, {'id': 49, 'name': 'banjo'}, {'id': 50, 'name': 'banner'}, {'id': 51, 'name': 'barbell'}, {'id': 52, 'name': 'barge'}, {'id': 53, 'name': 'barrel'}, {'id': 54, 'name': 'barrette'}, {'id': 55, 'name': 'barrow'}, {'id': 56, 'name': 'baseball_base'}, {'id': 57, 'name': 'baseball'}, {'id': 58, 'name': 'baseball_bat'}, {'id': 59, 'name': 'baseball_cap'}, {'id': 60, 'name': 'baseball_glove'}, {'id': 61, 'name': 'basket'}, {'id': 62, 'name': 'basketball'}, {'id': 63, 'name': 'bass_horn'}, {'id': 64, 'name': 'bat_(animal)'}, {'id': 65, 'name': 'bath_mat'}, {'id': 66, 'name': 'bath_towel'}, {'id': 67, 'name': 'bathrobe'}, {'id': 68, 'name': 'bathtub'}, {'id': 69, 'name': 'batter_(food)'}, {'id': 70, 'name': 'battery'}, {'id': 71, 'name': 'beachball'}, {'id': 72, 'name': 'bead'}, {'id': 73, 'name': 'bean_curd'}, {'id': 74, 'name': 'beanbag'}, {'id': 75, 'name': 'beanie'}, {'id': 76, 'name': 'bear'}, {'id': 77, 'name': 'bed'}, {'id': 78, 'name': 'bedpan'}, {'id': 79, 'name': 'bedspread'}, {'id': 80, 'name': 'cow'}, {'id': 81, 'name': 'beef_(food)'}, {'id': 82, 'name': 'beeper'}, {'id': 83, 'name': 'beer_bottle'}, {'id': 84, 'name': 'beer_can'}, {'id': 85, 'name': 'beetle'}, {'id': 86, 'name': 'bell'}, {'id': 87, 'name': 'bell_pepper'}, {'id': 88, 'name': 'belt'}, {'id': 89, 'name': 'belt_buckle'}, {'id': 90, 'name': 'bench'}, {'id': 91, 'name': 'beret'}, {'id': 92, 'name': 'bib'}, {'id': 93, 'name': 'Bible'}, {'id': 94, 'name': 'bicycle'}, {'id': 95, 'name': 'visor'}, {'id': 96, 'name': 'billboard'}, {'id': 97, 'name': 'binder'}, {'id': 98, 'name': 'binoculars'}, {'id': 99, 'name': 'bird'}, {'id': 100, 'name': 'birdfeeder'}, {'id': 101, 'name': 'birdbath'}, {'id': 102, 'name': 'birdcage'}, {'id': 103, 'name': 'birdhouse'}, {'id': 104, 'name': 'birthday_cake'}, {'id': 105, 'name': 'birthday_card'}, {'id': 106, 'name': 'pirate_flag'}, {'id': 107, 'name': 'black_sheep'}, {'id': 108, 'name': 'blackberry'}, {'id': 109, 'name': 'blackboard'}, {'id': 110, 'name': 'blanket'}, {'id': 111, 'name': 'blazer'}, {'id': 112, 'name': 'blender'}, {'id': 113, 'name': 'blimp'}, {'id': 114, 'name': 'blinker'}, {'id': 115, 'name': 'blouse'}, {'id': 116, 'name': 'blueberry'}, {'id': 117, 'name': 'gameboard'}, {'id': 118, 'name': 'boat'}, {'id': 119, 'name': 'bob'}, {'id': 120, 'name': 'bobbin'}, {'id': 121, 'name': 'bobby_pin'}, {'id': 122, 'name': 'boiled_egg'}, {'id': 123, 'name': 'bolo_tie'}, {'id': 124, 'name': 'deadbolt'}, {'id': 125, 'name': 'bolt'}, {'id': 126, 'name': 'bonnet'}, {'id': 127, 'name': 'book'}, {'id': 128, 'name': 'bookcase'}, {'id': 129, 'name': 'booklet'}, {'id': 130, 'name': 'bookmark'}, {'id': 131, 'name': 'boom_microphone'}, {'id': 132, 'name': 'boot'}, {'id': 133, 'name': 'bottle'}, {'id': 134, 'name': 'bottle_opener'}, {'id': 135, 'name': 'bouquet'}, {'id': 136, 'name': 'bow_(weapon)'}, {'id': 137, 'name': 'bow_(decorative_ribbons)'}, {'id': 138, 'name': 'bow-tie'}, {'id': 139, 'name': 'bowl'}, {'id': 140, 'name': 'pipe_bowl'}, {'id': 141, 'name': 'bowler_hat'}, {'id': 142, 'name': 'bowling_ball'}, {'id': 143, 'name': 'box'}, {'id': 144, 'name': 'boxing_glove'}, {'id': 145, 'name': 'suspenders'}, {'id': 146, 'name': 'bracelet'}, {'id': 147, 'name': 'brass_plaque'}, {'id': 148, 'name': 'brassiere'}, {'id': 149, 'name': 'bread-bin'}, {'id': 150, 'name': 'bread'}, {'id': 151, 'name': 'breechcloth'}, {'id': 152, 'name': 'bridal_gown'}, {'id': 153, 'name': 'briefcase'}, {'id': 154, 'name': 'broccoli'}, {'id': 155, 'name': 'broach'}, {'id': 156, 'name': 'broom'}, {'id': 157, 'name': 'brownie'}, {'id': 158, 'name': 'brussels_sprouts'}, {'id': 159, 'name': 'bubble_gum'}, {'id': 160, 'name': 'bucket'}, {'id': 161, 'name': 'horse_buggy'}, {'id': 162, 'name': 'bull'}, {'id': 163, 'name': 'bulldog'}, {'id': 164, 'name': 'bulldozer'}, {'id': 165, 'name': 'bullet_train'}, {'id': 166, 'name': 'bulletin_board'}, {'id': 167, 'name': 'bulletproof_vest'}, {'id': 168, 'name': 'bullhorn'}, {'id': 169, 'name': 'bun'}, {'id': 170, 'name': 'bunk_bed'}, {'id': 171, 'name': 'buoy'}, {'id': 172, 'name': 'burrito'}, {'id': 173, 'name': 'bus_(vehicle)'}, {'id': 174, 'name': 'business_card'}, {'id': 175, 'name': 'butter'}, {'id': 176, 'name': 'butterfly'}, {'id': 177, 'name': 'button'}, {'id': 178, 'name': 'cab_(taxi)'}, {'id': 179, 'name': 'cabana'}, {'id': 180, 'name': 'cabin_car'}, {'id': 181, 'name': 'cabinet'}, {'id': 182, 'name': 'locker'}, {'id': 183, 'name': 'cake'}, {'id': 184, 'name': 'calculator'}, {'id': 185, 'name': 'calendar'}, {'id': 186, 'name': 'calf'}, {'id': 187, 'name': 'camcorder'}, {'id': 188, 'name': 'camel'}, {'id': 189, 'name': 'camera'}, {'id': 190, 'name': 'camera_lens'}, {'id': 191, 'name': 'camper_(vehicle)'}, {'id': 192, 'name': 'can'}, {'id': 193, 'name': 'can_opener'}, {'id': 194, 'name': 'candle'}, {'id': 195, 'name': 'candle_holder'}, {'id': 196, 'name': 'candy_bar'}, {'id': 197, 'name': 'candy_cane'}, {'id': 198, 'name': 'walking_cane'}, {'id': 199, 'name': 'canister'}, {'id': 200, 'name': 'canoe'}, {'id': 201, 'name': 'cantaloup'}, {'id': 202, 'name': 'canteen'}, {'id': 203, 'name': 'cap_(headwear)'}, {'id': 204, 'name': 'bottle_cap'}, {'id': 205, 'name': 'cape'}, {'id': 206, 'name': 'cappuccino'}, {'id': 207, 'name': 'car_(automobile)'}, {'id': 208, 'name': 'railcar_(part_of_a_train)'}, {'id': 209, 'name': 'elevator_car'}, {'id': 210, 'name': 'car_battery'}, {'id': 211, 'name': 'identity_card'}, {'id': 212, 'name': 'card'}, {'id': 213, 'name': 'cardigan'}, {'id': 214, 'name': 'cargo_ship'}, {'id': 215, 'name': 'carnation'}, {'id': 216, 'name': 'horse_carriage'}, {'id': 217, 'name': 'carrot'}, {'id': 218, 'name': 'tote_bag'}, {'id': 219, 'name': 'cart'}, {'id': 220, 'name': 'carton'}, {'id': 221, 'name': 'cash_register'}, {'id': 222, 'name': 'casserole'}, {'id': 223, 'name': 'cassette'}, {'id': 224, 'name': 'cast'}, {'id': 225, 'name': 'cat'}, {'id': 226, 'name': 'cauliflower'}, {'id': 227, 'name': 'cayenne_(spice)'}, {'id': 228, 'name': 'CD_player'}, {'id': 229, 'name': 'celery'}, {'id': 230, 'name': 'cellular_telephone'}, {'id': 231, 'name': 'chain_mail'}, {'id': 232, 'name': 'chair'}, {'id': 233, 'name': 'chaise_longue'}, {'id': 234, 'name': 'chalice'}, {'id': 235, 'name': 'chandelier'}, {'id': 236, 'name': 'chap'}, {'id': 237, 'name': 'checkbook'}, {'id': 238, 'name': 'checkerboard'}, {'id': 239, 'name': 'cherry'}, {'id': 240, 'name': 'chessboard'}, {'id': 241, 'name': 'chicken_(animal)'}, {'id': 242, 'name': 'chickpea'}, {'id': 243, 'name': 'chili_(vegetable)'}, {'id': 244, 'name': 'chime'}, {'id': 245, 'name': 'chinaware'}, {'id': 246, 'name': 'crisp_(potato_chip)'}, {'id': 247, 'name': 'poker_chip'}, {'id': 248, 'name': 'chocolate_bar'}, {'id': 249, 'name': 'chocolate_cake'}, {'id': 250, 'name': 'chocolate_milk'}, {'id': 251, 'name': 'chocolate_mousse'}, {'id': 252, 'name': 'choker'}, {'id': 253, 'name': 'chopping_board'}, {'id': 254, 'name': 'chopstick'}, {'id': 255, 'name': 'Christmas_tree'}, {'id': 256, 'name': 'slide'}, {'id': 257, 'name': 'cider'}, {'id': 258, 'name': 'cigar_box'}, {'id': 259, 'name': 'cigarette'}, {'id': 260, 'name': 'cigarette_case'}, {'id': 261, 'name': 'cistern'}, {'id': 262, 'name': 'clarinet'}, {'id': 263, 'name': 'clasp'}, {'id': 264, 'name': 'cleansing_agent'}, {'id': 265, 'name': 'cleat_(for_securing_rope)'}, {'id': 266, 'name': 'clementine'}, {'id': 267, 'name': 'clip'}, {'id': 268, 'name': 'clipboard'}, {'id': 269, 'name': 'clippers_(for_plants)'}, {'id': 270, 'name': 'cloak'}, {'id': 271, 'name': 'clock'}, {'id': 272, 'name': 'clock_tower'}, {'id': 273, 'name': 'clothes_hamper'}, {'id': 274, 'name': 'clothespin'}, {'id': 275, 'name': 'clutch_bag'}, {'id': 276, 'name': 'coaster'}, {'id': 277, 'name': 'coat'}, {'id': 278, 'name': 'coat_hanger'}, {'id': 279, 'name': 'coatrack'}, {'id': 280, 'name': 'cock'}, {'id': 281, 'name': 'cockroach'}, {'id': 282, 'name': 'cocoa_(beverage)'}, {'id': 283, 'name': 'coconut'}, {'id': 284, 'name': 'coffee_maker'}, {'id': 285, 'name': 'coffee_table'}, {'id': 286, 'name': 'coffeepot'}, {'id': 287, 'name': 'coil'}, {'id': 288, 'name': 'coin'}, {'id': 289, 'name': 'colander'}, {'id': 290, 'name': 'coleslaw'}, {'id': 291, 'name': 'coloring_material'}, {'id': 292, 'name': 'combination_lock'}, {'id': 293, 'name': 'pacifier'}, {'id': 294, 'name': 'comic_book'}, {'id': 295, 'name': 'compass'}, {'id': 296, 'name': 'computer_keyboard'}, {'id': 297, 'name': 'condiment'}, {'id': 298, 'name': 'cone'}, {'id': 299, 'name': 'control'}, {'id': 300, 'name': 'convertible_(automobile)'}, {'id': 301, 'name': 'sofa_bed'}, {'id': 302, 'name': 'cooker'}, {'id': 303, 'name': 'cookie'}, {'id': 304, 'name': 'cooking_utensil'}, {'id': 305, 'name': 'cooler_(for_food)'}, {'id': 306, 'name': 'cork_(bottle_plug)'}, {'id': 307, 'name': 'corkboard'}, {'id': 308, 'name': 'corkscrew'}, {'id': 309, 'name': 'edible_corn'}, {'id': 310, 'name': 'cornbread'}, {'id': 311, 'name': 'cornet'}, {'id': 312, 'name': 'cornice'}, {'id': 313, 'name': 'cornmeal'}, {'id': 314, 'name': 'corset'}, {'id': 315, 'name': 'costume'}, {'id': 316, 'name': 'cougar'}, {'id': 317, 'name': 'coverall'}, {'id': 318, 'name': 'cowbell'}, {'id': 319, 'name': 'cowboy_hat'}, {'id': 320, 'name': 'crab_(animal)'}, {'id': 321, 'name': 'crabmeat'}, {'id': 322, 'name': 'cracker'}, {'id': 323, 'name': 'crape'}, {'id': 324, 'name': 'crate'}, {'id': 325, 'name': 'crayon'}, {'id': 326, 'name': 'cream_pitcher'}, {'id': 327, 'name': 'crescent_roll'}, {'id': 328, 'name': 'crib'}, {'id': 329, 'name': 'crock_pot'}, {'id': 330, 'name': 'crossbar'}, {'id': 331, 'name': 'crouton'}, {'id': 332, 'name': 'crow'}, {'id': 333, 'name': 'crowbar'}, {'id': 334, 'name': 'crown'}, {'id': 335, 'name': 'crucifix'}, {'id': 336, 'name': 'cruise_ship'}, {'id': 337, 'name': 'police_cruiser'}, {'id': 338, 'name': 'crumb'}, {'id': 339, 'name': 'crutch'}, {'id': 340, 'name': 'cub_(animal)'}, {'id': 341, 'name': 'cube'}, {'id': 342, 'name': 'cucumber'}, {'id': 343, 'name': 'cufflink'}, {'id': 344, 'name': 'cup'}, {'id': 345, 'name': 'trophy_cup'}, {'id': 346, 'name': 'cupboard'}, {'id': 347, 'name': 'cupcake'}, {'id': 348, 'name': 'hair_curler'}, {'id': 349, 'name': 'curling_iron'}, {'id': 350, 'name': 'curtain'}, {'id': 351, 'name': 'cushion'}, {'id': 352, 'name': 'cylinder'}, {'id': 353, 'name': 'cymbal'}, {'id': 354, 'name': 'dagger'}, {'id': 355, 'name': 'dalmatian'}, {'id': 356, 'name': 'dartboard'}, {'id': 357, 'name': 'date_(fruit)'}, {'id': 358, 'name': 'deck_chair'}, {'id': 359, 'name': 'deer'}, {'id': 360, 'name': 'dental_floss'}, {'id': 361, 'name': 'desk'}, {'id': 362, 'name': 'detergent'}, {'id': 363, 'name': 'diaper'}, {'id': 364, 'name': 'diary'}, {'id': 365, 'name': 'die'}, {'id': 366, 'name': 'dinghy'}, {'id': 367, 'name': 'dining_table'}, {'id': 368, 'name': 'tux'}, {'id': 369, 'name': 'dish'}, {'id': 370, 'name': 'dish_antenna'}, {'id': 371, 'name': 'dishrag'}, {'id': 372, 'name': 'dishtowel'}, {'id': 373, 'name': 'dishwasher'}, {'id': 374, 'name': 'dishwasher_detergent'}, {'id': 375, 'name': 'dispenser'}, {'id': 376, 'name': 'diving_board'}, {'id': 377, 'name': 'Dixie_cup'}, {'id': 378, 'name': 'dog'}, {'id': 379, 'name': 'dog_collar'}, {'id': 380, 'name': 'doll'}, {'id': 381, 'name': 'dollar'}, {'id': 382, 'name': 'dollhouse'}, {'id': 383, 'name': 'dolphin'}, {'id': 384, 'name': 'domestic_ass'}, {'id': 385, 'name': 'doorknob'}, {'id': 386, 'name': 'doormat'}, {'id': 387, 'name': 'doughnut'}, {'id': 388, 'name': 'dove'}, {'id': 389, 'name': 'dragonfly'}, {'id': 390, 'name': 'drawer'}, {'id': 391, 'name': 'underdrawers'}, {'id': 392, 'name': 'dress'}, {'id': 393, 'name': 'dress_hat'}, {'id': 394, 'name': 'dress_suit'}, {'id': 395, 'name': 'dresser'}, {'id': 396, 'name': 'drill'}, {'id': 397, 'name': 'drone'}, {'id': 398, 'name': 'dropper'}, {'id': 399, 'name': 'drum_(musical_instrument)'}, {'id': 400, 'name': 'drumstick'}, {'id': 401, 'name': 'duck'}, {'id': 402, 'name': 'duckling'}, {'id': 403, 'name': 'duct_tape'}, {'id': 404, 'name': 'duffel_bag'}, {'id': 405, 'name': 'dumbbell'}, {'id': 406, 'name': 'dumpster'}, {'id': 407, 'name': 'dustpan'}, {'id': 408, 'name': 'eagle'}, {'id': 409, 'name': 'earphone'}, {'id': 410, 'name': 'earplug'}, {'id': 411, 'name': 'earring'}, {'id': 412, 'name': 'easel'}, {'id': 413, 'name': 'eclair'}, {'id': 414, 'name': 'eel'}, {'id': 415, 'name': 'egg'}, {'id': 416, 'name': 'egg_roll'}, {'id': 417, 'name': 'egg_yolk'}, {'id': 418, 'name': 'eggbeater'}, {'id': 419, 'name': 'eggplant'}, {'id': 420, 'name': 'electric_chair'}, {'id': 421, 'name': 'refrigerator'}, {'id': 422, 'name': 'elephant'}, {'id': 423, 'name': 'elk'}, {'id': 424, 'name': 'envelope'}, {'id': 425, 'name': 'eraser'}, {'id': 426, 'name': 'escargot'}, {'id': 427, 'name': 'eyepatch'}, {'id': 428, 'name': 'falcon'}, {'id': 429, 'name': 'fan'}, {'id': 430, 'name': 'faucet'}, {'id': 431, 'name': 'fedora'}, {'id': 432, 'name': 'ferret'}, {'id': 433, 'name': 'Ferris_wheel'}, {'id': 434, 'name': 'ferry'}, {'id': 435, 'name': 'fig_(fruit)'}, {'id': 436, 'name': 'fighter_jet'}, {'id': 437, 'name': 'figurine'}, {'id': 438, 'name': 'file_cabinet'}, {'id': 439, 'name': 'file_(tool)'}, {'id': 440, 'name': 'fire_alarm'}, {'id': 441, 'name': 'fire_engine'}, {'id': 442, 'name': 'fire_extinguisher'}, {'id': 443, 'name': 'fire_hose'}, {'id': 444, 'name': 'fireplace'}, {'id': 445, 'name': 'fireplug'}, {'id': 446, 'name': 'first-aid_kit'}, {'id': 447, 'name': 'fish'}, {'id': 448, 'name': 'fish_(food)'}, {'id': 449, 'name': 'fishbowl'}, {'id': 450, 'name': 'fishing_rod'}, {'id': 451, 'name': 'flag'}, {'id': 452, 'name': 'flagpole'}, {'id': 453, 'name': 'flamingo'}, {'id': 454, 'name': 'flannel'}, {'id': 455, 'name': 'flap'}, {'id': 456, 'name': 'flash'}, {'id': 457, 'name': 'flashlight'}, {'id': 458, 'name': 'fleece'}, {'id': 459, 'name': 'flip-flop_(sandal)'}, {'id': 460, 'name': 'flipper_(footwear)'}, {'id': 461, 'name': 'flower_arrangement'}, {'id': 462, 'name': 'flute_glass'}, {'id': 463, 'name': 'foal'}, {'id': 464, 'name': 'folding_chair'}, {'id': 465, 'name': 'food_processor'}, {'id': 466, 'name': 'football_(American)'}, {'id': 467, 'name': 'football_helmet'}, {'id': 468, 'name': 'footstool'}, {'id': 469, 'name': 'fork'}, {'id': 470, 'name': 'forklift'}, {'id': 471, 'name': 'freight_car'}, {'id': 472, 'name': 'French_toast'}, {'id': 473, 'name': 'freshener'}, {'id': 474, 'name': 'frisbee'}, {'id': 475, 'name': 'frog'}, {'id': 476, 'name': 'fruit_juice'}, {'id': 477, 'name': 'frying_pan'}, {'id': 478, 'name': 'fudge'}, {'id': 479, 'name': 'funnel'}, {'id': 480, 'name': 'futon'}, {'id': 481, 'name': 'gag'}, {'id': 482, 'name': 'garbage'}, {'id': 483, 'name': 'garbage_truck'}, {'id': 484, 'name': 'garden_hose'}, {'id': 485, 'name': 'gargle'}, {'id': 486, 'name': 'gargoyle'}, {'id': 487, 'name': 'garlic'}, {'id': 488, 'name': 'gasmask'}, {'id': 489, 'name': 'gazelle'}, {'id': 490, 'name': 'gelatin'}, {'id': 491, 'name': 'gemstone'}, {'id': 492, 'name': 'generator'}, {'id': 493, 'name': 'giant_panda'}, {'id': 494, 'name': 'gift_wrap'}, {'id': 495, 'name': 'ginger'}, {'id': 496, 'name': 'giraffe'}, {'id': 497, 'name': 'cincture'}, {'id': 498, 'name': 'glass_(drink_container)'}, {'id': 499, 'name': 'globe'}, {'id': 500, 'name': 'glove'}, {'id': 501, 'name': 'goat'}, {'id': 502, 'name': 'goggles'}, {'id': 503, 'name': 'goldfish'}, {'id': 504, 'name': 'golf_club'}, {'id': 505, 'name': 'golfcart'}, {'id': 506, 'name': 'gondola_(boat)'}, {'id': 507, 'name': 'goose'}, {'id': 508, 'name': 'gorilla'}, {'id': 509, 'name': 'gourd'}, {'id': 510, 'name': 'grape'}, {'id': 511, 'name': 'grater'}, {'id': 512, 'name': 'gravestone'}, {'id': 513, 'name': 'gravy_boat'}, {'id': 514, 'name': 'green_bean'}, {'id': 515, 'name': 'green_onion'}, {'id': 516, 'name': 'griddle'}, {'id': 517, 'name': 'grill'}, {'id': 518, 'name': 'grits'}, {'id': 519, 'name': 'grizzly'}, {'id': 520, 'name': 'grocery_bag'}, {'id': 521, 'name': 'guitar'}, {'id': 522, 'name': 'gull'}, {'id': 523, 'name': 'gun'}, {'id': 524, 'name': 'hairbrush'}, {'id': 525, 'name': 'hairnet'}, {'id': 526, 'name': 'hairpin'}, {'id': 527, 'name': 'halter_top'}, {'id': 528, 'name': 'ham'}, {'id': 529, 'name': 'hamburger'}, {'id': 530, 'name': 'hammer'}, {'id': 531, 'name': 'hammock'}, {'id': 532, 'name': 'hamper'}, {'id': 533, 'name': 'hamster'}, {'id': 534, 'name': 'hair_dryer'}, {'id': 535, 'name': 'hand_glass'}, {'id': 536, 'name': 'hand_towel'}, {'id': 537, 'name': 'handcart'}, {'id': 538, 'name': 'handcuff'}, {'id': 539, 'name': 'handkerchief'}, {'id': 540, 'name': 'handle'}, {'id': 541, 'name': 'handsaw'}, {'id': 542, 'name': 'hardback_book'}, {'id': 543, 'name': 'harmonium'}, {'id': 544, 'name': 'hat'}, {'id': 545, 'name': 'hatbox'}, {'id': 546, 'name': 'veil'}, {'id': 547, 'name': 'headband'}, {'id': 548, 'name': 'headboard'}, {'id': 549, 'name': 'headlight'}, {'id': 550, 'name': 'headscarf'}, {'id': 551, 'name': 'headset'}, {'id': 552, 'name': 'headstall_(for_horses)'}, {'id': 553, 'name': 'heart'}, {'id': 554, 'name': 'heater'}, {'id': 555, 'name': 'helicopter'}, {'id': 556, 'name': 'helmet'}, {'id': 557, 'name': 'heron'}, {'id': 558, 'name': 'highchair'}, {'id': 559, 'name': 'hinge'}, {'id': 560, 'name': 'hippopotamus'}, {'id': 561, 'name': 'hockey_stick'}, {'id': 562, 'name': 'hog'}, {'id': 563, 'name': 'home_plate_(baseball)'}, {'id': 564, 'name': 'honey'}, {'id': 565, 'name': 'fume_hood'}, {'id': 566, 'name': 'hook'}, {'id': 567, 'name': 'hookah'}, {'id': 568, 'name': 'hornet'}, {'id': 569, 'name': 'horse'}, {'id': 570, 'name': 'hose'}, {'id': 571, 'name': 'hot-air_balloon'}, {'id': 572, 'name': 'hotplate'}, {'id': 573, 'name': 'hot_sauce'}, {'id': 574, 'name': 'hourglass'}, {'id': 575, 'name': 'houseboat'}, {'id': 576, 'name': 'hummingbird'}, {'id': 577, 'name': 'hummus'}, {'id': 578, 'name': 'polar_bear'}, {'id': 579, 'name': 'icecream'}, {'id': 580, 'name': 'popsicle'}, {'id': 581, 'name': 'ice_maker'}, {'id': 582, 'name': 'ice_pack'}, {'id': 583, 'name': 'ice_skate'}, {'id': 584, 'name': 'igniter'}, {'id': 585, 'name': 'inhaler'}, {'id': 586, 'name': 'iPod'}, {'id': 587, 'name': 'iron_(for_clothing)'}, {'id': 588, 'name': 'ironing_board'}, {'id': 589, 'name': 'jacket'}, {'id': 590, 'name': 'jam'}, {'id': 591, 'name': 'jar'}, {'id': 592, 'name': 'jean'}, {'id': 593, 'name': 'jeep'}, {'id': 594, 'name': 'jelly_bean'}, {'id': 595, 'name': 'jersey'}, {'id': 596, 'name': 'jet_plane'}, {'id': 597, 'name': 'jewel'}, {'id': 598, 'name': 'jewelry'}, {'id': 599, 'name': 'joystick'}, {'id': 600, 'name': 'jumpsuit'}, {'id': 601, 'name': 'kayak'}, {'id': 602, 'name': 'keg'}, {'id': 603, 'name': 'kennel'}, {'id': 604, 'name': 'kettle'}, {'id': 605, 'name': 'key'}, {'id': 606, 'name': 'keycard'}, {'id': 607, 'name': 'kilt'}, {'id': 608, 'name': 'kimono'}, {'id': 609, 'name': 'kitchen_sink'}, {'id': 610, 'name': 'kitchen_table'}, {'id': 611, 'name': 'kite'}, {'id': 612, 'name': 'kitten'}, {'id': 613, 'name': 'kiwi_fruit'}, {'id': 614, 'name': 'knee_pad'}, {'id': 615, 'name': 'knife'}, {'id': 616, 'name': 'knitting_needle'}, {'id': 617, 'name': 'knob'}, {'id': 618, 'name': 'knocker_(on_a_door)'}, {'id': 619, 'name': 'koala'}, {'id': 620, 'name': 'lab_coat'}, {'id': 621, 'name': 'ladder'}, {'id': 622, 'name': 'ladle'}, {'id': 623, 'name': 'ladybug'}, {'id': 624, 'name': 'lamb_(animal)'}, {'id': 625, 'name': 'lamb-chop'}, {'id': 626, 'name': 'lamp'}, {'id': 627, 'name': 'lamppost'}, {'id': 628, 'name': 'lampshade'}, {'id': 629, 'name': 'lantern'}, {'id': 630, 'name': 'lanyard'}, {'id': 631, 'name': 'laptop_computer'}, {'id': 632, 'name': 'lasagna'}, {'id': 633, 'name': 'latch'}, {'id': 634, 'name': 'lawn_mower'}, {'id': 635, 'name': 'leather'}, {'id': 636, 'name': 'legging_(clothing)'}, {'id': 637, 'name': 'Lego'}, {'id': 638, 'name': 'legume'}, {'id': 639, 'name': 'lemon'}, {'id': 640, 'name': 'lemonade'}, {'id': 641, 'name': 'lettuce'}, {'id': 642, 'name': 'license_plate'}, {'id': 643, 'name': 'life_buoy'}, {'id': 644, 'name': 'life_jacket'}, {'id': 645, 'name': 'lightbulb'}, {'id': 646, 'name': 'lightning_rod'}, {'id': 647, 'name': 'lime'}, {'id': 648, 'name': 'limousine'}, {'id': 649, 'name': 'lion'}, {'id': 650, 'name': 'lip_balm'}, {'id': 651, 'name': 'liquor'}, {'id': 652, 'name': 'lizard'}, {'id': 653, 'name': 'log'}, {'id': 654, 'name': 'lollipop'}, {'id': 655, 'name': 'speaker_(stero_equipment)'}, {'id': 656, 'name': 'loveseat'}, {'id': 657, 'name': 'machine_gun'}, {'id': 658, 'name': 'magazine'}, {'id': 659, 'name': 'magnet'}, {'id': 660, 'name': 'mail_slot'}, {'id': 661, 'name': 'mailbox_(at_home)'}, {'id': 662, 'name': 'mallard'}, {'id': 663, 'name': 'mallet'}, {'id': 664, 'name': 'mammoth'}, {'id': 665, 'name': 'manatee'}, {'id': 666, 'name': 'mandarin_orange'}, {'id': 667, 'name': 'manger'}, {'id': 668, 'name': 'manhole'}, {'id': 669, 'name': 'map'}, {'id': 670, 'name': 'marker'}, {'id': 671, 'name': 'martini'}, {'id': 672, 'name': 'mascot'}, {'id': 673, 'name': 'mashed_potato'}, {'id': 674, 'name': 'masher'}, {'id': 675, 'name': 'mask'}, {'id': 676, 'name': 'mast'}, {'id': 677, 'name': 'mat_(gym_equipment)'}, {'id': 678, 'name': 'matchbox'}, {'id': 679, 'name': 'mattress'}, {'id': 680, 'name': 'measuring_cup'}, {'id': 681, 'name': 'measuring_stick'}, {'id': 682, 'name': 'meatball'}, {'id': 683, 'name': 'medicine'}, {'id': 684, 'name': 'melon'}, {'id': 685, 'name': 'microphone'}, {'id': 686, 'name': 'microscope'}, {'id': 687, 'name': 'microwave_oven'}, {'id': 688, 'name': 'milestone'}, {'id': 689, 'name': 'milk'}, {'id': 690, 'name': 'milk_can'}, {'id': 691, 'name': 'milkshake'}, {'id': 692, 'name': 'minivan'}, {'id': 693, 'name': 'mint_candy'}, {'id': 694, 'name': 'mirror'}, {'id': 695, 'name': 'mitten'}, {'id': 696, 'name': 'mixer_(kitchen_tool)'}, {'id': 697, 'name': 'money'}, {'id': 698, 'name': 'monitor_(computer_equipment) computer_monitor'}, {'id': 699, 'name': 'monkey'}, {'id': 700, 'name': 'motor'}, {'id': 701, 'name': 'motor_scooter'}, {'id': 702, 'name': 'motor_vehicle'}, {'id': 703, 'name': 'motorcycle'}, {'id': 704, 'name': 'mound_(baseball)'}, {'id': 705, 'name': 'mouse_(computer_equipment)'}, {'id': 706, 'name': 'mousepad'}, {'id': 707, 'name': 'muffin'}, {'id': 708, 'name': 'mug'}, {'id': 709, 'name': 'mushroom'}, {'id': 710, 'name': 'music_stool'}, {'id': 711, 'name': 'musical_instrument'}, {'id': 712, 'name': 'nailfile'}, {'id': 713, 'name': 'napkin'}, {'id': 714, 'name': 'neckerchief'}, {'id': 715, 'name': 'necklace'}, {'id': 716, 'name': 'necktie'}, {'id': 717, 'name': 'needle'}, {'id': 718, 'name': 'nest'}, {'id': 719, 'name': 'newspaper'}, {'id': 720, 'name': 'newsstand'}, {'id': 721, 'name': 'nightshirt'}, {'id': 722, 'name': 'nosebag_(for_animals)'}, {'id': 723, 'name': 'noseband_(for_animals)'}, {'id': 724, 'name': 'notebook'}, {'id': 725, 'name': 'notepad'}, {'id': 726, 'name': 'nut'}, {'id': 727, 'name': 'nutcracker'}, {'id': 728, 'name': 'oar'}, {'id': 729, 'name': 'octopus_(food)'}, {'id': 730, 'name': 'octopus_(animal)'}, {'id': 731, 'name': 'oil_lamp'}, {'id': 732, 'name': 'olive_oil'}, {'id': 733, 'name': 'omelet'}, {'id': 734, 'name': 'onion'}, {'id': 735, 'name': 'orange_(fruit)'}, {'id': 736, 'name': 'orange_juice'}, {'id': 737, 'name': 'ostrich'}, {'id': 738, 'name': 'ottoman'}, {'id': 739, 'name': 'oven'}, {'id': 740, 'name': 'overalls_(clothing)'}, {'id': 741, 'name': 'owl'}, {'id': 742, 'name': 'packet'}, {'id': 743, 'name': 'inkpad'}, {'id': 744, 'name': 'pad'}, {'id': 745, 'name': 'paddle'}, {'id': 746, 'name': 'padlock'}, {'id': 747, 'name': 'paintbrush'}, {'id': 748, 'name': 'painting'}, {'id': 749, 'name': 'pajamas'}, {'id': 750, 'name': 'palette'}, {'id': 751, 'name': 'pan_(for_cooking)'}, {'id': 752, 'name': 'pan_(metal_container)'}, {'id': 753, 'name': 'pancake'}, {'id': 754, 'name': 'pantyhose'}, {'id': 755, 'name': 'papaya'}, {'id': 756, 'name': 'paper_plate'}, {'id': 757, 'name': 'paper_towel'}, {'id': 758, 'name': 'paperback_book'}, {'id': 759, 'name': 'paperweight'}, {'id': 760, 'name': 'parachute'}, {'id': 761, 'name': 'parakeet'}, {'id': 762, 'name': 'parasail_(sports)'}, {'id': 763, 'name': 'parasol'}, {'id': 764, 'name': 'parchment'}, {'id': 765, 'name': 'parka'}, {'id': 766, 'name': 'parking_meter'}, {'id': 767, 'name': 'parrot'}, {'id': 768, 'name': 'passenger_car_(part_of_a_train)'}, {'id': 769, 'name': 'passenger_ship'}, {'id': 770, 'name': 'passport'}, {'id': 771, 'name': 'pastry'}, {'id': 772, 'name': 'patty_(food)'}, {'id': 773, 'name': 'pea_(food)'}, {'id': 774, 'name': 'peach'}, {'id': 775, 'name': 'peanut_butter'}, {'id': 776, 'name': 'pear'}, {'id': 777, 'name': 'peeler_(tool_for_fruit_and_vegetables)'}, {'id': 778, 'name': 'wooden_leg'}, {'id': 779, 'name': 'pegboard'}, {'id': 780, 'name': 'pelican'}, {'id': 781, 'name': 'pen'}, {'id': 782, 'name': 'pencil'}, {'id': 783, 'name': 'pencil_box'}, {'id': 784, 'name': 'pencil_sharpener'}, {'id': 785, 'name': 'pendulum'}, {'id': 786, 'name': 'penguin'}, {'id': 787, 'name': 'pennant'}, {'id': 788, 'name': 'penny_(coin)'}, {'id': 789, 'name': 'pepper'}, {'id': 790, 'name': 'pepper_mill'}, {'id': 791, 'name': 'perfume'}, {'id': 792, 'name': 'persimmon'}, {'id': 793, 'name': 'person'}, {'id': 794, 'name': 'pet'}, {'id': 795, 'name': 'pew_(church_bench)'}, {'id': 796, 'name': 'phonebook'}, {'id': 797, 'name': 'phonograph_record'}, {'id': 798, 'name': 'piano'}, {'id': 799, 'name': 'pickle'}, {'id': 800, 'name': 'pickup_truck'}, {'id': 801, 'name': 'pie'}, {'id': 802, 'name': 'pigeon'}, {'id': 803, 'name': 'piggy_bank'}, {'id': 804, 'name': 'pillow'}, {'id': 805, 'name': 'pin_(non_jewelry)'}, {'id': 806, 'name': 'pineapple'}, {'id': 807, 'name': 'pinecone'}, {'id': 808, 'name': 'ping-pong_ball'}, {'id': 809, 'name': 'pinwheel'}, {'id': 810, 'name': 'tobacco_pipe'}, {'id': 811, 'name': 'pipe'}, {'id': 812, 'name': 'pistol'}, {'id': 813, 'name': 'pita_(bread)'}, {'id': 814, 'name': 'pitcher_(vessel_for_liquid)'}, {'id': 815, 'name': 'pitchfork'}, {'id': 816, 'name': 'pizza'}, {'id': 817, 'name': 'place_mat'}, {'id': 818, 'name': 'plate'}, {'id': 819, 'name': 'platter'}, {'id': 820, 'name': 'playpen'}, {'id': 821, 'name': 'pliers'}, {'id': 822, 'name': 'plow_(farm_equipment)'}, {'id': 823, 'name': 'plume'}, {'id': 824, 'name': 'pocket_watch'}, {'id': 825, 'name': 'pocketknife'}, {'id': 826, 'name': 'poker_(fire_stirring_tool)'}, {'id': 827, 'name': 'pole'}, {'id': 828, 'name': 'polo_shirt'}, {'id': 829, 'name': 'poncho'}, {'id': 830, 'name': 'pony'}, {'id': 831, 'name': 'pool_table'}, {'id': 832, 'name': 'pop_(soda)'}, {'id': 833, 'name': 'postbox_(public)'}, {'id': 834, 'name': 'postcard'}, {'id': 835, 'name': 'poster'}, {'id': 836, 'name': 'pot'}, {'id': 837, 'name': 'flowerpot'}, {'id': 838, 'name': 'potato'}, {'id': 839, 'name': 'potholder'}, {'id': 840, 'name': 'pottery'}, {'id': 841, 'name': 'pouch'}, {'id': 842, 'name': 'power_shovel'}, {'id': 843, 'name': 'prawn'}, {'id': 844, 'name': 'pretzel'}, {'id': 845, 'name': 'printer'}, {'id': 846, 'name': 'projectile_(weapon)'}, {'id': 847, 'name': 'projector'}, {'id': 848, 'name': 'propeller'}, {'id': 849, 'name': 'prune'}, {'id': 850, 'name': 'pudding'}, {'id': 851, 'name': 'puffer_(fish)'}, {'id': 852, 'name': 'puffin'}, {'id': 853, 'name': 'pug-dog'}, {'id': 854, 'name': 'pumpkin'}, {'id': 855, 'name': 'puncher'}, {'id': 856, 'name': 'puppet'}, {'id': 857, 'name': 'puppy'}, {'id': 858, 'name': 'quesadilla'}, {'id': 859, 'name': 'quiche'}, {'id': 860, 'name': 'quilt'}, {'id': 861, 'name': 'rabbit'}, {'id': 862, 'name': 'race_car'}, {'id': 863, 'name': 'racket'}, {'id': 864, 'name': 'radar'}, {'id': 865, 'name': 'radiator'}, {'id': 866, 'name': 'radio_receiver'}, {'id': 867, 'name': 'radish'}, {'id': 868, 'name': 'raft'}, {'id': 869, 'name': 'rag_doll'}, {'id': 870, 'name': 'raincoat'}, {'id': 871, 'name': 'ram_(animal)'}, {'id': 872, 'name': 'raspberry'}, {'id': 873, 'name': 'rat'}, {'id': 874, 'name': 'razorblade'}, {'id': 875, 'name': 'reamer_(juicer)'}, {'id': 876, 'name': 'rearview_mirror'}, {'id': 877, 'name': 'receipt'}, {'id': 878, 'name': 'recliner'}, {'id': 879, 'name': 'record_player'}, {'id': 880, 'name': 'reflector'}, {'id': 881, 'name': 'remote_control'}, {'id': 882, 'name': 'rhinoceros'}, {'id': 883, 'name': 'rib_(food)'}, {'id': 884, 'name': 'rifle'}, {'id': 885, 'name': 'ring'}, {'id': 886, 'name': 'river_boat'}, {'id': 887, 'name': 'road_map'}, {'id': 888, 'name': 'robe'}, {'id': 889, 'name': 'rocking_chair'}, {'id': 890, 'name': 'rodent'}, {'id': 891, 'name': 'roller_skate'}, {'id': 892, 'name': 'Rollerblade'}, {'id': 893, 'name': 'rolling_pin'}, {'id': 894, 'name': 'root_beer'}, {'id': 895, 'name': 'router_(computer_equipment)'}, {'id': 896, 'name': 'rubber_band'}, {'id': 897, 'name': 'runner_(carpet)'}, {'id': 898, 'name': 'plastic_bag'}, {'id': 899, 'name': 'saddle_(on_an_animal)'}, {'id': 900, 'name': 'saddle_blanket'}, {'id': 901, 'name': 'saddlebag'}, {'id': 902, 'name': 'safety_pin'}, {'id': 903, 'name': 'sail'}, {'id': 904, 'name': 'salad'}, {'id': 905, 'name': 'salad_plate'}, {'id': 906, 'name': 'salami'}, {'id': 907, 'name': 'salmon_(fish)'}, {'id': 908, 'name': 'salmon_(food)'}, {'id': 909, 'name': 'salsa'}, {'id': 910, 'name': 'saltshaker'}, {'id': 911, 'name': 'sandal_(type_of_shoe)'}, {'id': 912, 'name': 'sandwich'}, {'id': 913, 'name': 'satchel'}, {'id': 914, 'name': 'saucepan'}, {'id': 915, 'name': 'saucer'}, {'id': 916, 'name': 'sausage'}, {'id': 917, 'name': 'sawhorse'}, {'id': 918, 'name': 'saxophone'}, {'id': 919, 'name': 'scale_(measuring_instrument)'}, {'id': 920, 'name': 'scarecrow'}, {'id': 921, 'name': 'scarf'}, {'id': 922, 'name': 'school_bus'}, {'id': 923, 'name': 'scissors'}, {'id': 924, 'name': 'scoreboard'}, {'id': 925, 'name': 'scraper'}, {'id': 926, 'name': 'screwdriver'}, {'id': 927, 'name': 'scrubbing_brush'}, {'id': 928, 'name': 'sculpture'}, {'id': 929, 'name': 'seabird'}, {'id': 930, 'name': 'seahorse'}, {'id': 931, 'name': 'seaplane'}, {'id': 932, 'name': 'seashell'}, {'id': 933, 'name': 'sewing_machine'}, {'id': 934, 'name': 'shaker'}, {'id': 935, 'name': 'shampoo'}, {'id': 936, 'name': 'shark'}, {'id': 937, 'name': 'sharpener'}, {'id': 938, 'name': 'Sharpie'}, {'id': 939, 'name': 'shaver_(electric)'}, {'id': 940, 'name': 'shaving_cream'}, {'id': 941, 'name': 'shawl'}, {'id': 942, 'name': 'shears'}, {'id': 943, 'name': 'sheep'}, {'id': 944, 'name': 'shepherd_dog'}, {'id': 945, 'name': 'sherbert'}, {'id': 946, 'name': 'shield'}, {'id': 947, 'name': 'shirt'}, {'id': 948, 'name': 'shoe'}, {'id': 949, 'name': 'shopping_bag'}, {'id': 950, 'name': 'shopping_cart'}, {'id': 951, 'name': 'short_pants'}, {'id': 952, 'name': 'shot_glass'}, {'id': 953, 'name': 'shoulder_bag'}, {'id': 954, 'name': 'shovel'}, {'id': 955, 'name': 'shower_head'}, {'id': 956, 'name': 'shower_cap'}, {'id': 957, 'name': 'shower_curtain'}, {'id': 958, 'name': 'shredder_(for_paper)'}, {'id': 959, 'name': 'signboard'}, {'id': 960, 'name': 'silo'}, {'id': 961, 'name': 'sink'}, {'id': 962, 'name': 'skateboard'}, {'id': 963, 'name': 'skewer'}, {'id': 964, 'name': 'ski'}, {'id': 965, 'name': 'ski_boot'}, {'id': 966, 'name': 'ski_parka'}, {'id': 967, 'name': 'ski_pole'}, {'id': 968, 'name': 'skirt'}, {'id': 969, 'name': 'skullcap'}, {'id': 970, 'name': 'sled'}, {'id': 971, 'name': 'sleeping_bag'}, {'id': 972, 'name': 'sling_(bandage)'}, {'id': 973, 'name': 'slipper_(footwear)'}, {'id': 974, 'name': 'smoothie'}, {'id': 975, 'name': 'snake'}, {'id': 976, 'name': 'snowboard'}, {'id': 977, 'name': 'snowman'}, {'id': 978, 'name': 'snowmobile'}, {'id': 979, 'name': 'soap'}, {'id': 980, 'name': 'soccer_ball'}, {'id': 981, 'name': 'sock'}, {'id': 982, 'name': 'sofa'}, {'id': 983, 'name': 'softball'}, {'id': 984, 'name': 'solar_array'}, {'id': 985, 'name': 'sombrero'}, {'id': 986, 'name': 'soup'}, {'id': 987, 'name': 'soup_bowl'}, {'id': 988, 'name': 'soupspoon'}, {'id': 989, 'name': 'sour_cream'}, {'id': 990, 'name': 'soya_milk'}, {'id': 991, 'name': 'space_shuttle'}, {'id': 992, 'name': 'sparkler_(fireworks)'}, {'id': 993, 'name': 'spatula'}, {'id': 994, 'name': 'spear'}, {'id': 995, 'name': 'spectacles'}, {'id': 996, 'name': 'spice_rack'}, {'id': 997, 'name': 'spider'}, {'id': 998, 'name': 'crawfish'}, {'id': 999, 'name': 'sponge'}, {'id': 1000, 'name': 'spoon'}, {'id': 1001, 'name': 'sportswear'}, {'id': 1002, 'name': 'spotlight'}, {'id': 1003, 'name': 'squid_(food)'}, {'id': 1004, 'name': 'squirrel'}, {'id': 1005, 'name': 'stagecoach'}, {'id': 1006, 'name': 'stapler_(stapling_machine)'}, {'id': 1007, 'name': 'starfish'}, {'id': 1008, 'name': 'statue_(sculpture)'}, {'id': 1009, 'name': 'steak_(food)'}, {'id': 1010, 'name': 'steak_knife'}, {'id': 1011, 'name': 'steering_wheel'}, {'id': 1012, 'name': 'stepladder'}, {'id': 1013, 'name': 'step_stool'}, {'id': 1014, 'name': 'stereo_(sound_system)'}, {'id': 1015, 'name': 'stew'}, {'id': 1016, 'name': 'stirrer'}, {'id': 1017, 'name': 'stirrup'}, {'id': 1018, 'name': 'stool'}, {'id': 1019, 'name': 'stop_sign'}, {'id': 1020, 'name': 'brake_light'}, {'id': 1021, 'name': 'stove'}, {'id': 1022, 'name': 'strainer'}, {'id': 1023, 'name': 'strap'}, {'id': 1024, 'name': 'straw_(for_drinking)'}, {'id': 1025, 'name': 'strawberry'}, {'id': 1026, 'name': 'street_sign'}, {'id': 1027, 'name': 'streetlight'}, {'id': 1028, 'name': 'string_cheese'}, {'id': 1029, 'name': 'stylus'}, {'id': 1030, 'name': 'subwoofer'}, {'id': 1031, 'name': 'sugar_bowl'}, {'id': 1032, 'name': 'sugarcane_(plant)'}, {'id': 1033, 'name': 'suit_(clothing)'}, {'id': 1034, 'name': 'sunflower'}, {'id': 1035, 'name': 'sunglasses'}, {'id': 1036, 'name': 'sunhat'}, {'id': 1037, 'name': 'surfboard'}, {'id': 1038, 'name': 'sushi'}, {'id': 1039, 'name': 'mop'}, {'id': 1040, 'name': 'sweat_pants'}, {'id': 1041, 'name': 'sweatband'}, {'id': 1042, 'name': 'sweater'}, {'id': 1043, 'name': 'sweatshirt'}, {'id': 1044, 'name': 'sweet_potato'}, {'id': 1045, 'name': 'swimsuit'}, {'id': 1046, 'name': 'sword'}, {'id': 1047, 'name': 'syringe'}, {'id': 1048, 'name': 'Tabasco_sauce'}, {'id': 1049, 'name': 'table-tennis_table'}, {'id': 1050, 'name': 'table'}, {'id': 1051, 'name': 'table_lamp'}, {'id': 1052, 'name': 'tablecloth'}, {'id': 1053, 'name': 'tachometer'}, {'id': 1054, 'name': 'taco'}, {'id': 1055, 'name': 'tag'}, {'id': 1056, 'name': 'taillight'}, {'id': 1057, 'name': 'tambourine'}, {'id': 1058, 'name': 'army_tank'}, {'id': 1059, 'name': 'tank_(storage_vessel)'}, {'id': 1060, 'name': 'tank_top_(clothing)'}, {'id': 1061, 'name': 'tape_(sticky_cloth_or_paper)'}, {'id': 1062, 'name': 'tape_measure'}, {'id': 1063, 'name': 'tapestry'}, {'id': 1064, 'name': 'tarp'}, {'id': 1065, 'name': 'tartan'}, {'id': 1066, 'name': 'tassel'}, {'id': 1067, 'name': 'tea_bag'}, {'id': 1068, 'name': 'teacup'}, {'id': 1069, 'name': 'teakettle'}, {'id': 1070, 'name': 'teapot'}, {'id': 1071, 'name': 'teddy_bear'}, {'id': 1072, 'name': 'telephone'}, {'id': 1073, 'name': 'telephone_booth'}, {'id': 1074, 'name': 'telephone_pole'}, {'id': 1075, 'name': 'telephoto_lens'}, {'id': 1076, 'name': 'television_camera'}, {'id': 1077, 'name': 'television_set'}, {'id': 1078, 'name': 'tennis_ball'}, {'id': 1079, 'name': 'tennis_racket'}, {'id': 1080, 'name': 'tequila'}, {'id': 1081, 'name': 'thermometer'}, {'id': 1082, 'name': 'thermos_bottle'}, {'id': 1083, 'name': 'thermostat'}, {'id': 1084, 'name': 'thimble'}, {'id': 1085, 'name': 'thread'}, {'id': 1086, 'name': 'thumbtack'}, {'id': 1087, 'name': 'tiara'}, {'id': 1088, 'name': 'tiger'}, {'id': 1089, 'name': 'tights_(clothing)'}, {'id': 1090, 'name': 'timer'}, {'id': 1091, 'name': 'tinfoil'}, {'id': 1092, 'name': 'tinsel'}, {'id': 1093, 'name': 'tissue_paper'}, {'id': 1094, 'name': 'toast_(food)'}, {'id': 1095, 'name': 'toaster'}, {'id': 1096, 'name': 'toaster_oven'}, {'id': 1097, 'name': 'toilet'}, {'id': 1098, 'name': 'toilet_tissue'}, {'id': 1099, 'name': 'tomato'}, {'id': 1100, 'name': 'tongs'}, {'id': 1101, 'name': 'toolbox'}, {'id': 1102, 'name': 'toothbrush'}, {'id': 1103, 'name': 'toothpaste'}, {'id': 1104, 'name': 'toothpick'}, {'id': 1105, 'name': 'cover'}, {'id': 1106, 'name': 'tortilla'}, {'id': 1107, 'name': 'tow_truck'}, {'id': 1108, 'name': 'towel'}, {'id': 1109, 'name': 'towel_rack'}, {'id': 1110, 'name': 'toy'}, {'id': 1111, 'name': 'tractor_(farm_equipment)'}, {'id': 1112, 'name': 'traffic_light'}, {'id': 1113, 'name': 'dirt_bike'}, {'id': 1114, 'name': 'trailer_truck'}, {'id': 1115, 'name': 'train_(railroad_vehicle)'}, {'id': 1116, 'name': 'trampoline'}, {'id': 1117, 'name': 'tray'}, {'id': 1118, 'name': 'trench_coat'}, {'id': 1119, 'name': 'triangle_(musical_instrument)'}, {'id': 1120, 'name': 'tricycle'}, {'id': 1121, 'name': 'tripod'}, {'id': 1122, 'name': 'trousers'}, {'id': 1123, 'name': 'truck'}, {'id': 1124, 'name': 'truffle_(chocolate)'}, {'id': 1125, 'name': 'trunk'}, {'id': 1126, 'name': 'vat'}, {'id': 1127, 'name': 'turban'}, {'id': 1128, 'name': 'turkey_(food)'}, {'id': 1129, 'name': 'turnip'}, {'id': 1130, 'name': 'turtle'}, {'id': 1131, 'name': 'turtleneck_(clothing)'}, {'id': 1132, 'name': 'typewriter'}, {'id': 1133, 'name': 'umbrella'}, {'id': 1134, 'name': 'underwear'}, {'id': 1135, 'name': 'unicycle'}, {'id': 1136, 'name': 'urinal'}, {'id': 1137, 'name': 'urn'}, {'id': 1138, 'name': 'vacuum_cleaner'}, {'id': 1139, 'name': 'vase'}, {'id': 1140, 'name': 'vending_machine'}, {'id': 1141, 'name': 'vent'}, {'id': 1142, 'name': 'vest'}, {'id': 1143, 'name': 'videotape'}, {'id': 1144, 'name': 'vinegar'}, {'id': 1145, 'name': 'violin'}, {'id': 1146, 'name': 'vodka'}, {'id': 1147, 'name': 'volleyball'}, {'id': 1148, 'name': 'vulture'}, {'id': 1149, 'name': 'waffle'}, {'id': 1150, 'name': 'waffle_iron'}, {'id': 1151, 'name': 'wagon'}, {'id': 1152, 'name': 'wagon_wheel'}, {'id': 1153, 'name': 'walking_stick'}, {'id': 1154, 'name': 'wall_clock'}, {'id': 1155, 'name': 'wall_socket'}, {'id': 1156, 'name': 'wallet'}, {'id': 1157, 'name': 'walrus'}, {'id': 1158, 'name': 'wardrobe'}, {'id': 1159, 'name': 'washbasin'}, {'id': 1160, 'name': 'automatic_washer'}, {'id': 1161, 'name': 'watch'}, {'id': 1162, 'name': 'water_bottle'}, {'id': 1163, 'name': 'water_cooler'}, {'id': 1164, 'name': 'water_faucet'}, {'id': 1165, 'name': 'water_heater'}, {'id': 1166, 'name': 'water_jug'}, {'id': 1167, 'name': 'water_gun'}, {'id': 1168, 'name': 'water_scooter'}, {'id': 1169, 'name': 'water_ski'}, {'id': 1170, 'name': 'water_tower'}, {'id': 1171, 'name': 'watering_can'}, {'id': 1172, 'name': 'watermelon'}, {'id': 1173, 'name': 'weathervane'}, {'id': 1174, 'name': 'webcam'}, {'id': 1175, 'name': 'wedding_cake'}, {'id': 1176, 'name': 'wedding_ring'}, {'id': 1177, 'name': 'wet_suit'}, {'id': 1178, 'name': 'wheel'}, {'id': 1179, 'name': 'wheelchair'}, {'id': 1180, 'name': 'whipped_cream'}, {'id': 1181, 'name': 'whistle'}, {'id': 1182, 'name': 'wig'}, {'id': 1183, 'name': 'wind_chime'}, {'id': 1184, 'name': 'windmill'}, {'id': 1185, 'name': 'window_box_(for_plants)'}, {'id': 1186, 'name': 'windshield_wiper'}, {'id': 1187, 'name': 'windsock'}, {'id': 1188, 'name': 'wine_bottle'}, {'id': 1189, 'name': 'wine_bucket'}, {'id': 1190, 'name': 'wineglass'}, {'id': 1191, 'name': 'blinder_(for_horses)'}, {'id': 1192, 'name': 'wok'}, {'id': 1193, 'name': 'wolf'}, {'id': 1194, 'name': 'wooden_spoon'}, {'id': 1195, 'name': 'wreath'}, {'id': 1196, 'name': 'wrench'}, {'id': 1197, 'name': 'wristband'}, {'id': 1198, 'name': 'wristlet'}, {'id': 1199, 'name': 'yacht'}, {'id': 1200, 'name': 'yogurt'}, {'id': 1201, 'name': 'yoke_(animal_equipment)'}, {'id': 1202, 'name': 'zebra'}, {'id': 1203, 'name': 'zucchini'}]

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
# id_map_coco = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 11, 11: 13, 12: 14, 13: 15, 14: 16, 15: 17, 16: 18, 17: 19, 18: 20, 19: 21, 20: 22, 21: 23, 22: 24, 23: 25, 24: 27, 25: 28, 26: 31, 27: 32, 28: 33, 29: 34, 30: 35, 31: 36, 32: 37, 33: 38, 34: 39, 35: 40, 36: 41, 37: 42, 38: 43, 39: 44, 40: 46,
#                   41: 47, 42: 48, 43: 49, 44: 50, 45: 51, 46: 52, 47: 53, 48: 54, 49: 55, 50: 56, 51: 57, 52: 58, 53: 59, 54: 60, 55: 61, 56: 62, 57: 63, 58: 64, 59: 65, 60: 67, 61: 70, 62: 72, 63: 73, 64: 74, 65: 75, 66: 76, 67: 77, 68: 78, 69: 79, 70: 80, 71: 81, 72: 82, 73: 84, 74: 85, 75: 86, 76: 87, 77: 88, 78: 89, 79: 90}

id_map_lvis = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 11, 11: 12, 12: 13, 13: 14, 14: 15, 15: 16, 16: 17, 17: 18, 18: 19, 19: 20, 20: 21, 21: 22, 22: 23, 23: 24, 24: 25, 25: 26, 26: 27, 27: 28, 28: 29, 29: 30, 30: 31, 31: 32, 32: 33, 33: 34, 34: 35, 35: 36, 36: 37, 37: 38, 38: 39, 39: 40, 40: 41, 41: 42, 42: 43, 43: 44, 44: 45, 45: 46, 46: 47, 47: 48, 48: 49, 49: 50, 50: 51, 51: 52, 52: 53, 53: 54, 54: 55, 55: 56, 56: 57, 57: 58, 58: 59, 59: 60, 60: 61, 61: 62, 62: 63, 63: 64, 64: 65, 65: 66, 66: 67, 67: 68, 68: 69, 69: 70, 70: 71, 71: 72, 72: 73, 73: 74, 74: 75, 75: 76, 76: 77, 77: 78, 78: 79, 79: 80, 80: 81, 81: 82, 82: 83, 83: 84, 84: 85, 85: 86, 86: 87, 87: 88, 88: 89, 89: 90, 90: 91, 91: 92, 92: 93, 93: 94, 94: 95, 95: 96, 96: 97, 97: 98, 98: 99, 99: 100, 100: 101, 101: 102, 102: 103, 103: 104, 104: 105, 105: 106, 106: 107, 107: 108, 108: 109, 109: 110, 110: 111, 111: 112, 112: 113, 113: 114, 114: 115, 115: 116, 116: 117, 117: 118, 118: 119, 119: 120, 120: 121, 121: 122, 122: 123, 123: 124, 124: 125, 125: 126, 126: 127, 127: 128, 128: 129, 129: 130, 130: 131, 131: 132, 132: 133, 133: 134, 134: 135, 135: 136, 136: 137, 137: 138, 138: 139, 139: 140, 140: 141, 141: 142, 142: 143, 143: 144, 144: 145, 145: 146, 146: 147, 147: 148, 148: 149, 149: 150, 150: 151, 151: 152, 152: 153, 153: 154, 154: 155, 155: 156, 156: 157, 157: 158, 158: 159, 159: 160, 160: 161, 161: 162, 162: 163, 163: 164, 164: 165, 165: 166, 166: 167, 167: 168, 168: 169, 169: 170, 170: 171, 171: 172, 172: 173, 173: 174, 174: 175, 175: 176, 176: 177, 177: 178, 178: 179, 179: 180, 180: 181, 181: 182, 182: 183, 183: 184, 184: 185, 185: 186, 186: 187, 187: 188, 188: 189, 189: 190, 190: 191, 191: 192, 192: 193, 193: 194, 194: 195, 195: 196, 196: 197, 197: 198, 198: 199, 199: 200, 200: 201, 201: 202, 202: 203, 203: 204, 204: 205, 205: 206, 206: 207, 207: 208, 208: 209, 209: 210, 210: 211, 211: 212, 212: 213, 213: 214, 214: 215, 215: 216, 216: 217, 217: 218, 218: 219, 219: 220, 220: 221, 221: 222, 222: 223, 223: 224, 224: 225, 225: 226, 226: 227, 227: 228, 228: 229, 229: 230, 230: 231, 231: 232, 232: 233, 233: 234, 234: 235, 235: 236, 236: 237, 237: 238, 238: 239, 239: 240, 240: 241, 241: 242, 242: 243, 243: 244, 244: 245, 245: 246, 246: 247, 247: 248, 248: 249, 249: 250, 250: 251, 251: 252, 252: 253, 253: 254, 254: 255, 255: 256, 256: 257, 257: 258, 258: 259, 259: 260, 260: 261, 261: 262, 262: 263, 263: 264, 264: 265, 265: 266, 266: 267, 267: 268, 268: 269, 269: 270, 270: 271, 271: 272, 272: 273, 273: 274, 274: 275, 275: 276, 276: 277, 277: 278, 278: 279, 279: 280, 280: 281, 281: 282, 282: 283, 283: 284, 284: 285, 285: 286, 286: 287, 287: 288, 288: 289, 289: 290, 290: 291, 291: 292, 292: 293, 293: 294, 294: 295, 295: 296, 296: 297, 297: 298, 298: 299, 299: 300, 300: 301, 301: 302, 302: 303, 303: 304, 304: 305, 305: 306, 306: 307, 307: 308, 308: 309, 309: 310, 310: 311, 311: 312, 312: 313, 313: 314, 314: 315, 315: 316, 316: 317, 317: 318, 318: 319, 319: 320, 320: 321, 321: 322, 322: 323, 323: 324, 324: 325, 325: 326, 326: 327, 327: 328, 328: 329, 329: 330, 330: 331, 331: 332, 332: 333, 333: 334, 334: 335, 335: 336, 336: 337, 337: 338, 338: 339, 339: 340, 340: 341, 341: 342, 342: 343, 343: 344, 344: 345, 345: 346, 346: 347, 347: 348, 348: 349, 349: 350, 350: 351, 351: 352, 352: 353, 353: 354, 354: 355, 355: 356, 356: 357, 357: 358, 358: 359, 359: 360, 360: 361, 361: 362, 362: 363, 363: 364, 364: 365, 365: 366, 366: 367, 367: 368, 368: 369, 369: 370, 370: 371, 371: 372, 372: 373, 373: 374, 374: 375, 375: 376, 376: 377, 377: 378, 378: 379, 379: 380, 380: 381, 381: 382, 382: 383, 383: 384, 384: 385, 385: 386, 386: 387, 387: 388, 388: 389, 389: 390, 390: 391, 391: 392, 392: 393, 393: 394, 394: 395, 395: 396, 396: 397, 397: 398, 398: 399, 399: 400, 400: 401, 401: 402, 402: 403, 403: 404, 404: 405, 405: 406, 406: 407, 407: 408, 408: 409, 409: 410, 410: 411, 411: 412, 412: 413, 413: 414, 414: 415, 415: 416, 416: 417, 417: 418, 418: 419, 419: 420, 420: 421, 421: 422, 422: 423, 423: 424, 424: 425, 425: 426, 426: 427, 427: 428, 428: 429, 429: 430, 430: 431, 431: 432, 432: 433, 433: 434, 434: 435, 435: 436, 436: 437, 437: 438, 438: 439, 439: 440, 440: 441, 441: 442, 442: 443, 443: 444, 444: 445, 445: 446, 446: 447, 447: 448, 448: 449, 449: 450, 450: 451, 451: 452, 452: 453, 453: 454, 454: 455, 455: 456, 456: 457, 457: 458, 458: 459, 459: 460, 460: 461, 461: 462, 462: 463, 463: 464, 464: 465, 465: 466, 466: 467, 467: 468, 468: 469, 469: 470, 470: 471, 471: 472, 472: 473, 473: 474, 474: 475, 475: 476, 476: 477, 477: 478, 478: 479, 479: 480, 480: 481, 481: 482, 482: 483, 483: 484, 484: 485, 485: 486, 486: 487, 487: 488, 488: 489, 489: 490, 490: 491, 491: 492, 492: 493, 493: 494, 494: 495, 495: 496, 496: 497, 497: 498, 498: 499, 499: 500, 500: 501, 501: 502, 502: 503, 503: 504, 504: 505, 505: 506, 506: 507, 507: 508, 508: 509, 509: 510, 510: 511, 511: 512, 512: 513, 513: 514, 514: 515, 515: 516, 516: 517, 517: 518, 518: 519, 519: 520, 520: 521, 521: 522, 522: 523, 523: 524, 524: 525, 525: 526, 526: 527, 527: 528, 528: 529, 529: 530, 530: 531, 531: 532, 532: 533, 533: 534, 534: 535, 535: 536, 536: 537, 537: 538, 538: 539, 539: 540, 540: 541, 541: 542, 542: 543, 543: 544, 544: 545, 545: 546, 546: 547, 547: 548, 548: 549, 549: 550, 550: 551, 551: 552, 552: 553, 553: 554, 554: 555, 555: 556, 556: 557, 557: 558, 558: 559, 559: 560, 560: 561, 561: 562, 562: 563, 563: 564, 564: 565, 565: 566, 566: 567, 567: 568, 568: 569, 569: 570, 570: 571, 571: 572, 572: 573, 573: 574, 574: 575, 575: 576, 576: 577, 577: 578, 578: 579, 579: 580, 580: 581, 581: 582, 582: 583, 583: 584, 584: 585, 585: 586, 586: 587, 587: 588, 588: 589, 589: 590, 590: 591, 591: 592, 592: 593, 593: 594, 594: 595, 595: 596, 596: 597, 597: 598, 598: 599, 599: 600, 600: 601, 601: 602, 602: 603, 603: 604, 604: 605, 605: 606, 606: 607, 607: 608, 608: 609, 609: 610, 610: 611, 611: 612, 612: 613, 613: 614, 614: 615, 615: 616, 616: 617, 617: 618, 618: 619, 619: 620, 620: 621, 621: 622, 622: 623, 623: 624, 624: 625, 625: 626, 626: 627, 627: 628, 628: 629, 629: 630, 630: 631, 631: 632, 632: 633, 633: 634, 634: 635, 635: 636, 636: 637, 637: 638, 638: 639, 639: 640, 640: 641, 641: 642, 642: 643, 643: 644, 644: 645, 645: 646, 646: 647, 647: 648, 648: 649, 649: 650, 650: 651, 651: 652, 652: 653, 653: 654, 654: 655, 655: 656, 656: 657, 657: 658, 658: 659, 659: 660, 660: 661, 661: 662, 662: 663, 663: 664, 664: 665, 665: 666, 666: 667, 667: 668, 668: 669, 669: 670, 670: 671, 671: 672, 672: 673, 673: 674, 674: 675, 675: 676, 676: 677, 677: 678, 678: 679, 679: 680, 680: 681, 681: 682, 682: 683, 683: 684, 684: 685, 685: 686, 686: 687, 687: 688, 688: 689, 689: 690, 690: 691, 691: 692, 692: 693, 693: 694, 694: 695, 695: 696, 696: 697, 697: 698, 698: 699, 699: 700, 700: 701, 701: 702, 702: 703, 703: 704, 704: 705, 705: 706, 706: 707, 707: 708, 708: 709, 709: 710, 710: 711, 711: 712, 712: 713, 713: 714, 714: 715, 715: 716, 716: 717, 717: 718, 718: 719, 719: 720, 720: 721, 721: 722, 722: 723, 723: 724, 724: 725, 725: 726, 726: 727, 727: 728, 728: 729, 729: 730, 730: 731, 731: 732, 732: 733, 733: 734, 734: 735, 735: 736, 736: 737, 737: 738, 738: 739, 739: 740, 740: 741, 741: 742, 742: 743, 743: 744, 744: 745, 745: 746, 746: 747, 747: 748, 748: 749, 749: 750, 750: 751, 751: 752, 752: 753, 753: 754, 754: 755, 755: 756, 756: 757, 757: 758, 758: 759, 759: 760, 760: 761, 761: 762, 762: 763, 763: 764, 764: 765, 765: 766, 766: 767, 767: 768, 768: 769, 769: 770, 770: 771, 771: 772, 772: 773, 773: 774, 774: 775, 775: 776, 776: 777, 777: 778, 778: 779, 779: 780, 780: 781, 781: 782, 782: 783, 783: 784, 784: 785, 785: 786, 786: 787, 787: 788, 788: 789, 789: 790, 790: 791, 791: 792, 792: 793, 793: 794, 794: 795, 795: 796, 796: 797, 797: 798, 798: 799, 799: 800, 800: 801, 801: 802, 802: 803, 803: 804, 804: 805, 805: 806, 806: 807, 807: 808, 808: 809, 809: 810, 810: 811, 811: 812, 812: 813, 813: 814, 814: 815, 815: 816, 816: 817, 817: 818, 818: 819, 819: 820, 820: 821, 821: 822, 822: 823, 823: 824, 824: 825, 825: 826, 826: 827, 827: 828, 828: 829, 829: 830, 830: 831, 831: 832, 832: 833, 833: 834, 834: 835, 835: 836, 836: 837, 837: 838, 838: 839, 839: 840, 840: 841, 841: 842, 842: 843, 843: 844, 844: 845, 845: 846, 846: 847, 847: 848, 848: 849, 849: 850, 850: 851, 851: 852, 852: 853, 853: 854, 854: 855, 855: 856, 856: 857, 857: 858, 858: 859, 859: 860, 860: 861, 861: 862, 862: 863, 863: 864, 864: 865, 865: 866, 866: 867, 867: 868, 868: 869, 869: 870, 870: 871, 871: 872, 872: 873, 873: 874, 874: 875, 875: 876, 876: 877, 877: 878, 878: 879, 879: 880, 880: 881, 881: 882, 882: 883, 883: 884, 884: 885, 885: 886, 886: 887, 887: 888, 888: 889, 889: 890, 890: 891, 891: 892, 892: 893, 893: 894, 894: 895, 895: 896, 896: 897, 897: 898, 898: 899, 899: 900, 900: 901, 901: 902, 902: 903, 903: 904, 904: 905, 905: 906, 906: 907, 907: 908, 908: 909, 909: 910, 910: 911, 911: 912, 912: 913, 913: 914, 914: 915, 915: 916, 916: 917, 917: 918, 918: 919, 919: 920, 920: 921, 921: 922, 922: 923, 923: 924, 924: 925, 925: 926, 926: 927, 927: 928, 928: 929, 929: 930, 930: 931, 931: 932, 932: 933, 933: 934, 934: 935, 935: 936, 936: 937, 937: 938, 938: 939, 939: 940, 940: 941, 941: 942, 942: 943, 943: 944, 944: 945, 945: 946, 946: 947, 947: 948, 948: 949, 949: 950, 950: 951, 951: 952, 952: 953, 953: 954, 954: 955, 955: 956, 956: 957, 957: 958, 958: 959, 959: 960, 960: 961, 961: 962, 962: 963, 963: 964, 964: 965, 965: 966, 966: 967, 967: 968, 968: 969, 969: 970, 970: 971, 971: 972, 972: 973, 973: 974, 974: 975, 975: 976, 976: 977, 977: 978, 978: 979, 979: 980, 980: 981, 981: 982, 982: 983, 983: 984, 984: 985, 985: 986, 986: 987, 987: 988, 988: 989, 989: 990, 990: 991, 991: 992, 992: 993, 993: 994, 994: 995, 995: 996, 996: 997, 997: 998, 998: 999, 999: 1000, 1000: 1001, 1001: 1002, 1002: 1003, 1003: 1004, 1004: 1005, 1005: 1006, 1006: 1007, 1007: 1008, 1008: 1009, 1009: 1010, 1010: 1011, 1011: 1012, 1012: 1013, 1013: 1014, 1014: 1015, 1015: 1016, 1016: 1017, 1017: 1018, 1018: 1019, 1019: 1020, 1020: 1021, 1021: 1022, 1022: 1023, 1023: 1024, 1024: 1025, 1025: 1026, 1026: 1027, 1027: 1028, 1028: 1029, 1029: 1030, 1030: 1031, 1031: 1032, 1032: 1033, 1033: 1034, 1034: 1035, 1035: 1036, 1036: 1037, 1037: 1038, 1038: 1039, 1039: 1040, 1040: 1041, 1041: 1042, 1042: 1043, 1043: 1044, 1044: 1045, 1045: 1046, 1046: 1047, 1047: 1048, 1048: 1049, 1049: 1050, 1050: 1051, 1051: 1052, 1052: 1053, 1053: 1054, 1054: 1055, 1055: 1056, 1056: 1057, 1057: 1058, 1058: 1059, 1059: 1060, 1060: 1061, 1061: 1062, 1062: 1063, 1063: 1064, 1064: 1065, 1065: 1066, 1066: 1067, 1067: 1068, 1068: 1069, 1069: 1070, 1070: 1071, 1071: 1072, 1072: 1073, 1073: 1074, 1074: 1075, 1075: 1076, 1076: 1077, 1077: 1078, 1078: 1079, 1079: 1080, 1080: 1081, 1081: 1082, 1082: 1083, 1083: 1084, 1084: 1085, 1085: 1086, 1086: 1087, 1087: 1088, 1088: 1089, 1089: 1090, 1090: 1091, 1091: 1092, 1092: 1093, 1093: 1094, 1094: 1095, 1095: 1096, 1096: 1097, 1097: 1098, 1098: 1099, 1099: 1100, 1100: 1101, 1101: 1102, 1102: 1103, 1103: 1104, 1104: 1105, 1105: 1106, 1106: 1107, 1107: 1108, 1108: 1109, 1109: 1110, 1110: 1111, 1111: 1112, 1112: 1113, 1113: 1114, 1114: 1115, 1115: 1116, 1116: 1117, 1117: 1118, 1118: 1119, 1119: 1120, 1120: 1121, 1121: 1122, 1122: 1123, 1123: 1124, 1124: 1125, 1125: 1126, 1126: 1127, 1127: 1128, 1128: 1129, 1129: 1130, 1130: 1131, 1131: 1132, 1132: 1133, 1133: 1134, 1134: 1135, 1135: 1136, 1136: 1137, 1137: 1138, 1138: 1139, 1139: 1140, 1140: 1141, 1141: 1142, 1142: 1143, 1143: 1144, 1144: 1145, 1145: 1146, 1146: 1147, 1147: 1148, 1148: 1149, 1149: 1150, 1150: 1151, 1151: 1152, 1152: 1153, 1153: 1154, 1154: 1155, 1155: 1156, 1156: 1157, 1157: 1158, 1158: 1159, 1159: 1160, 1160: 1161, 1161: 1162, 1162: 1163, 1163: 1164, 1164: 1165, 1165: 1166, 1166: 1167, 1167: 1168, 1168: 1169, 1169: 1170, 1170: 1171, 1171: 1172, 1172: 1173, 1173: 1174, 1174: 1175, 1175: 1176, 1176: 1177, 1177: 1178, 1178: 1179, 1179: 1180, 1180: 1181, 1181: 1182, 1182: 1183, 1183: 1184, 1184: 1185, 1185: 1186, 1186: 1187, 1187: 1188, 1188: 1189, 1189: 1190, 1190: 1191, 1191: 1192, 1192: 1193, 1193: 1194, 1194: 1195, 1195: 1196, 1196: 1197, 1197: 1198, 1198: 1199, 1199: 1200, 1200: 1201, 1201: 1202, 1202: 1203}

class PostProcessCocoGrounding(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""

    def __init__(self, num_select=300, tokenlizer=None) -> None:
        super().__init__()
        self.num_select = num_select

        # assert coco_api is not None
        # category_dict = coco_api.dataset['categories']
        category_dict = categories_lvis
    

        cat_list = [item['name'] for item in category_dict]
        # print(cat_list)
        captions, cat2tokenspan = build_captions_and_token_span(cat_list, True)
        

        # print(captions)
        # print(cat2tokenspan)
        # exit()
        tokenspanlist = [cat2tokenspan[cat.lower()] for cat in cat_list]
        positive_map = create_positive_map_from_span(
            tokenlizer(captions), tokenspanlist, max_text_len=4500)  # 80, 256. normed
        
        # with open("tokenspan_lvis.pkl",'wb') as f:
        #     pickle.dump(tokenspanlist, f,-1)        
        # print(tokenspanlist)
        # exit()
        
        

        
        id_map = {} 
        max_real_id = 0

        for key in id_map_lvis.keys():
            id_map[key] = id_map_lvis[key] 
            max_real_id = max(max_real_id, id_map[key])
        offset = len(id_map.keys())




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
    dataset = LvisDetection(
        args.image_dir, args.anno_path, transforms=transform)
    data_loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)

    # build post processor
    tokenlizer = get_tokenlizer.get_tokenlizer(cfg.text_encoder_type)
    postprocessor = PostProcessCocoGrounding(
         tokenlizer=tokenlizer)

    # build evaluator
    evaluator = LvisEvaluator(
        dataset.lvis, iou_types=("bbox",))

    # build captions
    # category_dict = dataset.coco.dataset['categories']
    category_dict = categories_lvis
    cat_list = [item['name'] for item in category_dict]
    
    size = len(cat_list) // 9
    remainder = len(cat_list) % 9

    result = []
    start = 0

    for i in range(8):  # 前5个列表
        end = start + size
        result.append(cat_list[start:end])
        start = end

    # 剩余的元素放到最后一个列表
    result.append(cat_list[start:])
    
    
    
    captions = []
    for r in result:
        # print(len(r))
        caption = " . ".join(r) + ' .'
        captions.append(caption)
    caption = " . ".join(cat_list) + ' .'
    


    # cat_list = [item['name'] for item in categories_aerial] + [item['name'] for item in categories_coco]
    
    
    print("Input text prompt:", captions)



    legal_task_id = ['aerial','aqua', 'cotton', 'egohand', 'mushroom', 'package', 'pascalvoc','pistol', 'pothole', 'raccoon',   'shellfish', 'thermal',   'vehicle']
    
    with open("subtasks_prompt_wo_coco_5_shot.pkl", 'rb') as f:
        subtasks_prompt_data = pickle.load(f)
        
    with open("subtasks_lora_wo_coco_1_shot.pkl", 'rb') as f:
        subtasks_lora_data = pickle.load(f)
        
        
        
    with open("task_image_feat_bank_1_shot/mean_feat/task_feats.pkl", 'rb') as f:
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
    load = False
    for i, (images, targets) in enumerate(data_loader):
        # get images and captions
        
        if not load:
            images = images.tensors.to(args.device)
            bs = images.shape[0]
        
            input_captions = [captions] * bs
        
            
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
            
            # with open("tmp_results_lvis/{}.pkl".format(i),'wb') as f:
            #     pickle.dump(results, f, -1)
        else:
            with open("tmp_results_lvis/{}.pkl".format(i),'rb') as f:
                results = pickle.load(f)      
        

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
    # use_prompt_name = args.checkpoint_path.split("/")[1]
    os.makedirs(args.save_path, exist_ok=True)
 
    with open(os.path.join(args.save_path, "results.json"), 'w') as f:
        json.dump({'result':evaluator.coco_eval["bbox"].stats.tolist()}, f)
        
    torch.save(model.state_dict(),os.path.join(args.save_path, "model.pth"))


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
    parser.add_argument("--test_name",  type=str,
                        help="number of workers for dataloader")
    parser.add_argument("--save_path",  type=str,
                        help="number of workers for dataloader")  
    args = parser.parse_args()

    main(args)

