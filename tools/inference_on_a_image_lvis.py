import argparse
import os
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import sys
# please make sure https://github.com/IDEA-Research/GroundingDINO is installed correctly.
import groundingdino.datasets.transforms as T
# from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from groundingdino.util.vl_utils import create_positive_map_from_span
import pickle

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

               
# def plot_boxes_to_image(image_pil, tgt):
#     H, W = tgt["size"]
#     boxes = tgt["boxes"]
#     labels = tgt["labels"]
#     assert len(boxes) == len(labels), "boxes and labels must have same length"

#     draw = ImageDraw.Draw(image_pil)
#     mask = Image.new("L", image_pil.size, 0)
#     mask_draw = ImageDraw.Draw(mask)

#     # draw boxes and masks
#     for box, label in zip(boxes, labels):
#         # from 0..1 to 0..W, 0..H
#         box = box * torch.Tensor([W, H, W, H])
#         # from xywh to xyxy
#         box[:2] -= box[2:] / 2
#         box[2:] += box[:2]
#         # random color
#         color = tuple(np.random.randint(0, 255, size=3).tolist())
#         # draw
#         x0, y0, x1, y1 = box
#         x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

#         draw.rectangle([x0, y0, x1, y1], outline=color, width=6)
#         # draw.text((x0, y0), str(label), fill=color)

#         font = ImageFont.load_default()
#         if hasattr(font, "getbbox"):
#             bbox = draw.textbbox((x0, y0), str(label), font)
#         else:
#             w, h = draw.textsize(str(label), font)
#             bbox = (x0, y0, w + x0, y0 + h)
#         # bbox = draw.textbbox((x0, y0), str(label))
#         draw.rectangle(bbox, fill=color)
#         draw.text((x0, y0), str(label), fill="white")

#         mask_draw.rectangle([x0, y0, x1, y1], fill=255, width=6)

#     return image_pil, mask


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
        
        if labal_name=='apricot':
            continue
        
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
    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold=None, with_logits=True, cpu_only=False, token_spans=None):
    assert text_threshold is not None or token_spans is not None, "text_threshould and token_spans should not be None at the same time!"
    # caption = caption.lower()
    # caption = caption.strip()
    # # print(len(caption.split(".")), caption.split("."))
    
    # # # print(len(caption),caption)
    # # exit()
    # if not caption.endswith("."):
    #     caption = caption + "."
    device = "cuda" if not cpu_only else "cpu"
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        if not args.use_zira:
            outputs = model(image[None], captions=[caption])
        else:
            outputs,_,_ = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].sigmoid()[0]  # (nq, 256)
  
    boxes = outputs["pred_boxes"][0]  # (nq, 4)

    class_name_cat = ""
    for i in range(len(caption)):
        class_name_cat+=caption[i]
            
    # filter output
    if token_spans is None:
        logits_filt = logits.cpu().clone()
        boxes_filt = boxes.cpu().clone()
        filt_mask = logits_filt.max(dim=1)[0] > box_threshold
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4

        # get phrase
        tokenlizer = model.tokenizer
        


        # class_name = class_name_cat.split(".")
        tokenized = tokenlizer(class_name_cat)
        # build pred
        pred_phrases = []
        for logit, box in zip(logits_filt, boxes_filt):
            # print(logit.shape)
            # exit()
            pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer, right_idx=4500)
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
            phrase = ' '.join([class_name_cat[_s:_e] for (_s, _e) in token_span])
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
        pred_phrases = all_phrases


    return boxes_filt, pred_phrases


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

    parser.add_argument("--box_threshold", type=float, default=0.3, help="box threshold")
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
    parser.add_argument("--coco_val_path", type=str, 
                        help="number of workers for dataloader") 
    args = parser.parse_args()

    # cfg
    config_file = args.config_file  # change the path of the model config file
    checkpoint_path = args.checkpoint_path  # change the path of the model
    image_path = args.image_path
    # text_prompt = args.text_prompt
    output_dir = args.output_dir
    box_threshold = args.box_threshold
    text_threshold = args.text_threshold
    # token_spans = args.token_spans
    
    with open("tokenspan_lvis.pkl",'rb') as f:
        token_spans = pickle.load(f)
    
    
    
    text_prompt = ['aerosol_can . air_conditioner . airplane . alarm_clock . alcohol . alligator . almond . ambulance . amplifier . anklet . antenna . apple . applesauce . apricot . apron . aquarium . arctic_(type_of_shoe) . armband . armchair . armoire . armor . artichoke . trash_can . ashtray . asparagus . atomizer . avocado . award . awning . ax . baboon . baby_buggy . basketball_backboard . backpack . handbag . suitcase . bagel . bagpipe . baguet . bait . ball . ballet_skirt . balloon . bamboo . banana . Band_Aid . bandage . bandanna . banjo . banner . barbell . barge . barrel . barrette . barrow . baseball_base . baseball . baseball_bat . baseball_cap . baseball_glove . basket . basketball . bass_horn . bat_(animal) . bath_mat . bath_towel . bathrobe . bathtub . batter_(food) . battery . beachball . bead . bean_curd . beanbag . beanie . bear . bed . bedpan . bedspread . cow . beef_(food) . beeper . beer_bottle . beer_can . beetle . bell . bell_pepper . belt . belt_buckle . bench . beret . bib . Bible . bicycle . visor . billboard . binder . binoculars . bird . birdfeeder . birdbath . birdcage . birdhouse . birthday_cake . birthday_card . pirate_flag . black_sheep . blackberry . blackboard . blanket . blazer . blender . blimp . blinker . blouse . blueberry . gameboard . boat . bob . bobbin . bobby_pin . boiled_egg . bolo_tie . deadbolt . bolt . bonnet . book . bookcase . booklet . bookmark . boom_microphone . boot . bottle .', 'bottle_opener . bouquet . bow_(weapon) . bow_(decorative_ribbons) . bow-tie . bowl . pipe_bowl . bowler_hat . bowling_ball . box . boxing_glove . suspenders . bracelet . brass_plaque . brassiere . bread-bin . bread . breechcloth . bridal_gown . briefcase . broccoli . broach . broom . brownie . brussels_sprouts . bubble_gum . bucket . horse_buggy . bull . bulldog . bulldozer . bullet_train . bulletin_board . bulletproof_vest . bullhorn . bun . bunk_bed . buoy . burrito . bus_(vehicle) . business_card . butter . butterfly . button . cab_(taxi) . cabana . cabin_car . cabinet . locker . cake . calculator . calendar . calf . camcorder . camel . camera . camera_lens . camper_(vehicle) . can . can_opener . candle . candle_holder . candy_bar . candy_cane . walking_cane . canister . canoe . cantaloup . canteen . cap_(headwear) . bottle_cap . cape . cappuccino . car_(automobile) . railcar_(part_of_a_train) . elevator_car . car_battery . identity_card . card . cardigan . cargo_ship . carnation . horse_carriage . carrot . tote_bag . cart . carton . cash_register . casserole . cassette . cast . cat . cauliflower . cayenne_(spice) . CD_player . celery . cellular_telephone . chain_mail . chair . chaise_longue . chalice . chandelier . chap . checkbook . checkerboard . cherry . chessboard . chicken_(animal) . chickpea . chili_(vegetable) . chime . chinaware . crisp_(potato_chip) . poker_chip . chocolate_bar . chocolate_cake . chocolate_milk . chocolate_mousse . choker . chopping_board . chopstick . Christmas_tree . slide . cider . cigar_box . cigarette . cigarette_case . cistern . clarinet . clasp . cleansing_agent . cleat_(for_securing_rope) . clementine .', 'clip . clipboard . clippers_(for_plants) . cloak . clock . clock_tower . clothes_hamper . clothespin . clutch_bag . coaster . coat . coat_hanger . coatrack . cock . cockroach . cocoa_(beverage) . coconut . coffee_maker . coffee_table . coffeepot . coil . coin . colander . coleslaw . coloring_material . combination_lock . pacifier . comic_book . compass . computer_keyboard . condiment . cone . control . convertible_(automobile) . sofa_bed . cooker . cookie . cooking_utensil . cooler_(for_food) . cork_(bottle_plug) . corkboard . corkscrew . edible_corn . cornbread . cornet . cornice . cornmeal . corset . costume . cougar . coverall . cowbell . cowboy_hat . crab_(animal) . crabmeat . cracker . crape . crate . crayon . cream_pitcher . crescent_roll . crib . crock_pot . crossbar . crouton . crow . crowbar . crown . crucifix . cruise_ship . police_cruiser . crumb . crutch . cub_(animal) . cube . cucumber . cufflink . cup . trophy_cup . cupboard . cupcake . hair_curler . curling_iron . curtain . cushion . cylinder . cymbal . dagger . dalmatian . dartboard . date_(fruit) . deck_chair . deer . dental_floss . desk . detergent . diaper . diary . die . dinghy . dining_table . tux . dish . dish_antenna . dishrag . dishtowel . dishwasher . dishwasher_detergent . dispenser . diving_board . Dixie_cup . dog . dog_collar . doll . dollar . dollhouse . dolphin . domestic_ass . doorknob . doormat . doughnut . dove . dragonfly . drawer . underdrawers . dress . dress_hat . dress_suit . dresser . drill . drone . dropper . drum_(musical_instrument) .', 'drumstick . duck . duckling . duct_tape . duffel_bag . dumbbell . dumpster . dustpan . eagle . earphone . earplug . earring . easel . eclair . eel . egg . egg_roll . egg_yolk . eggbeater . eggplant . electric_chair . refrigerator . elephant . elk . envelope . eraser . escargot . eyepatch . falcon . fan . faucet . fedora . ferret . Ferris_wheel . ferry . fig_(fruit) . fighter_jet . figurine . file_cabinet . file_(tool) . fire_alarm . fire_engine . fire_extinguisher . fire_hose . fireplace . fireplug . first-aid_kit . fish . fish_(food) . fishbowl . fishing_rod . flag . flagpole . flamingo . flannel . flap . flash . flashlight . fleece . flip-flop_(sandal) . flipper_(footwear) . flower_arrangement . flute_glass . foal . folding_chair . food_processor . football_(American) . football_helmet . footstool . fork . forklift . freight_car . French_toast . freshener . frisbee . frog . fruit_juice . frying_pan . fudge . funnel . futon . gag . garbage . garbage_truck . garden_hose . gargle . gargoyle . garlic . gasmask . gazelle . gelatin . gemstone . generator . giant_panda . gift_wrap . ginger . giraffe . cincture . glass_(drink_container) . globe . glove . goat . goggles . goldfish . golf_club . golfcart . gondola_(boat) . goose . gorilla . gourd . grape . grater . gravestone . gravy_boat . green_bean . green_onion . griddle . grill . grits . grizzly . grocery_bag . guitar . gull . gun . hairbrush . hairnet . hairpin . halter_top . ham . hamburger . hammer . hammock . hamper .', 'hamster . hair_dryer . hand_glass . hand_towel . handcart . handcuff . handkerchief . handle . handsaw . hardback_book . harmonium . hat . hatbox . veil . headband . headboard . headlight . headscarf . headset . headstall_(for_horses) . heart . heater . helicopter . helmet . heron . highchair . hinge . hippopotamus . hockey_stick . hog . home_plate_(baseball) . honey . fume_hood . hook . hookah . hornet . horse . hose . hot-air_balloon . hotplate . hot_sauce . hourglass . houseboat . hummingbird . hummus . polar_bear . icecream . popsicle . ice_maker . ice_pack . ice_skate . igniter . inhaler . iPod . iron_(for_clothing) . ironing_board . jacket . jam . jar . jean . jeep . jelly_bean . jersey . jet_plane . jewel . jewelry . joystick . jumpsuit . kayak . keg . kennel . kettle . key . keycard . kilt . kimono . kitchen_sink . kitchen_table . kite . kitten . kiwi_fruit . knee_pad . knife . knitting_needle . knob . knocker_(on_a_door) . koala . lab_coat . ladder . ladle . ladybug . lamb_(animal) . lamb-chop . lamp . lamppost . lampshade . lantern . lanyard . laptop_computer . lasagna . latch . lawn_mower . leather . legging_(clothing) . Lego . legume . lemon . lemonade . lettuce . license_plate . life_buoy . life_jacket . lightbulb . lightning_rod . lime . limousine . lion . lip_balm . liquor . lizard . log . lollipop . speaker_(stero_equipment) . loveseat . machine_gun . magazine . magnet . mail_slot . mailbox_(at_home) . mallard . mallet . mammoth . manatee .', 'mandarin_orange . manger . manhole . map . marker . martini . mascot . mashed_potato . masher . mask . mast . mat_(gym_equipment) . matchbox . mattress . measuring_cup . measuring_stick . meatball . medicine . melon . microphone . microscope . microwave_oven . milestone . milk . milk_can . milkshake . minivan . mint_candy . mirror . mitten . mixer_(kitchen_tool) . money . monitor_(computer_equipment) computer_monitor . monkey . motor . motor_scooter . motor_vehicle . motorcycle . mound_(baseball) . mouse_(computer_equipment) . mousepad . muffin . mug . mushroom . music_stool . musical_instrument . nailfile . napkin . neckerchief . necklace . necktie . needle . nest . newspaper . newsstand . nightshirt . nosebag_(for_animals) . noseband_(for_animals) . notebook . notepad . nut . nutcracker . oar . octopus_(food) . octopus_(animal) . oil_lamp . olive_oil . omelet . onion . orange_(fruit) . orange_juice . ostrich . ottoman . oven . overalls_(clothing) . owl . packet . inkpad . pad . paddle . padlock . paintbrush . painting . pajamas . palette . pan_(for_cooking) . pan_(metal_container) . pancake . pantyhose . papaya . paper_plate . paper_towel . paperback_book . paperweight . parachute . parakeet . parasail_(sports) . parasol . parchment . parka . parking_meter . parrot . passenger_car_(part_of_a_train) . passenger_ship . passport . pastry . patty_(food) . pea_(food) . peach . peanut_butter . pear . peeler_(tool_for_fruit_and_vegetables) . wooden_leg . pegboard . pelican . pen . pencil . pencil_box . pencil_sharpener . pendulum . penguin . pennant . penny_(coin) . pepper . pepper_mill . perfume . persimmon . person . pet . pew_(church_bench) . phonebook . phonograph_record . piano .', 'pickle . pickup_truck . pie . pigeon . piggy_bank . pillow . pin_(non_jewelry) . pineapple . pinecone . ping-pong_ball . pinwheel . tobacco_pipe . pipe . pistol . pita_(bread) . pitcher_(vessel_for_liquid) . pitchfork . pizza . place_mat . plate . platter . playpen . pliers . plow_(farm_equipment) . plume . pocket_watch . pocketknife . poker_(fire_stirring_tool) . pole . polo_shirt . poncho . pony . pool_table . pop_(soda) . postbox_(public) . postcard . poster . pot . flowerpot . potato . potholder . pottery . pouch . power_shovel . prawn . pretzel . printer . projectile_(weapon) . projector . propeller . prune . pudding . puffer_(fish) . puffin . pug-dog . pumpkin . puncher . puppet . puppy . quesadilla . quiche . quilt . rabbit . race_car . racket . radar . radiator . radio_receiver . radish . raft . rag_doll . raincoat . ram_(animal) . raspberry . rat . razorblade . reamer_(juicer) . rearview_mirror . receipt . recliner . record_player . reflector . remote_control . rhinoceros . rib_(food) . rifle . ring . river_boat . road_map . robe . rocking_chair . rodent . roller_skate . Rollerblade . rolling_pin . root_beer . router_(computer_equipment) . rubber_band . runner_(carpet) . plastic_bag . saddle_(on_an_animal) . saddle_blanket . saddlebag . safety_pin . sail . salad . salad_plate . salami . salmon_(fish) . salmon_(food) . salsa . saltshaker . sandal_(type_of_shoe) . sandwich . satchel . saucepan . saucer . sausage . sawhorse . saxophone . scale_(measuring_instrument) . scarecrow . scarf . school_bus . scissors . scoreboard . scraper . screwdriver . scrubbing_brush . sculpture . seabird . seahorse . seaplane .', 'seashell . sewing_machine . shaker . shampoo . shark . sharpener . Sharpie . shaver_(electric) . shaving_cream . shawl . shears . sheep . shepherd_dog . sherbert . shield . shirt . shoe . shopping_bag . shopping_cart . short_pants . shot_glass . shoulder_bag . shovel . shower_head . shower_cap . shower_curtain . shredder_(for_paper) . signboard . silo . sink . skateboard . skewer . ski . ski_boot . ski_parka . ski_pole . skirt . skullcap . sled . sleeping_bag . sling_(bandage) . slipper_(footwear) . smoothie . snake . snowboard . snowman . snowmobile . soap . soccer_ball . sock . sofa . softball . solar_array . sombrero . soup . soup_bowl . soupspoon . sour_cream . soya_milk . space_shuttle . sparkler_(fireworks) . spatula . spear . spectacles . spice_rack . spider . crawfish . sponge . spoon . sportswear . spotlight . squid_(food) . squirrel . stagecoach . stapler_(stapling_machine) . starfish . statue_(sculpture) . steak_(food) . steak_knife . steering_wheel . stepladder . step_stool . stereo_(sound_system) . stew . stirrer . stirrup . stool . stop_sign . brake_light . stove . strainer . strap . straw_(for_drinking) . strawberry . street_sign . streetlight . string_cheese . stylus . subwoofer . sugar_bowl . sugarcane_(plant) . suit_(clothing) . sunflower . sunglasses . sunhat . surfboard . sushi . mop . sweat_pants . sweatband . sweater . sweatshirt . sweet_potato . swimsuit . sword . syringe . Tabasco_sauce . table-tennis_table . table . table_lamp . tablecloth . tachometer . taco . tag . taillight . tambourine . army_tank . tank_(storage_vessel) . tank_top_(clothing) . tape_(sticky_cloth_or_paper) . tape_measure . tapestry . tarp .', 'tartan . tassel . tea_bag . teacup . teakettle . teapot . teddy_bear . telephone . telephone_booth . telephone_pole . telephoto_lens . television_camera . television_set . tennis_ball . tennis_racket . tequila . thermometer . thermos_bottle . thermostat . thimble . thread . thumbtack . tiara . tiger . tights_(clothing) . timer . tinfoil . tinsel . tissue_paper . toast_(food) . toaster . toaster_oven . toilet . toilet_tissue . tomato . tongs . toolbox . toothbrush . toothpaste . toothpick . cover . tortilla . tow_truck . towel . towel_rack . toy . tractor_(farm_equipment) . traffic_light . dirt_bike . trailer_truck . train_(railroad_vehicle) . trampoline . tray . trench_coat . triangle_(musical_instrument) . tricycle . tripod . trousers . truck . truffle_(chocolate) . trunk . vat . turban . turkey_(food) . turnip . turtle . turtleneck_(clothing) . typewriter . umbrella . underwear . unicycle . urinal . urn . vacuum_cleaner . vase . vending_machine . vent . vest . videotape . vinegar . violin . vodka . volleyball . vulture . waffle . waffle_iron . wagon . wagon_wheel . walking_stick . wall_clock . wall_socket . wallet . walrus . wardrobe . washbasin . automatic_washer . watch . water_bottle . water_cooler . water_faucet . water_heater . water_jug . water_gun . water_scooter . water_ski . water_tower . watering_can . watermelon . weathervane . webcam . wedding_cake . wedding_ring . wet_suit . wheel . wheelchair . whipped_cream . whistle . wig . wind_chime . windmill . window_box_(for_plants) . windshield_wiper . windsock . wine_bottle . wine_bucket . wineglass . blinder_(for_horses) . wok . wolf . wooden_spoon . wreath . wrench . wristband . wristlet . yacht . yogurt . yoke_(animal_equipment) . zebra . zucchini .']

    # make dir
    os.makedirs(output_dir, exist_ok=True)
    # load image
    image_pil, image = load_image(image_path)
    # load model
    model = load_model(args, checkpoint_path, cpu_only=args.cpu_only)

    # visualize raw image
    image_pil.save(os.path.join(output_dir, "raw_image.jpg"))

    # set the text_threshold to None if token_spans is set.
    if token_spans is not None:
        text_threshold = None
        print("Using token_spans. Set the text_threshold to None.")


    # run model
    boxes_filt, pred_phrases = get_grounding_output(
        model, image, text_prompt, box_threshold, text_threshold, cpu_only=args.cpu_only, token_spans=token_spans
    )

    # visualize pred
    size = image_pil.size
    pred_dict = {
        "boxes": boxes_filt,
        "size": [size[1], size[0]],  # H,W
        "labels": pred_phrases,
    }
    print(pred_dict)
    # print(pred_dict)
    # exit()
    
    class_name_cat = ""
    
    for i in range(len(text_prompt)):
        class_name_cat+=text_prompt[i]
    class_name = class_name_cat.split(".")
    
    for i in range(len(class_name)):
        class_name[i] = class_name[i].strip()
    class_name = class_name[:-1]
    text_2_color_map = {}
    
    for i in range(len(class_name)):
        text_2_color_map[class_name[i]] = PALETTE[i%150]
 
    # print(class_name)
    # exit()
    image_with_box = plot_boxes_to_image(image_pil, pred_dict, text_2_color_map)[0]
    save_path = os.path.join(output_dir, "pred.jpg")
    image_with_box.save(save_path)
    print(f"\n======================\n{save_path} saved.\nThe program runs successfully!")
