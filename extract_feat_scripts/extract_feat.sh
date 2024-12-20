UDA_VISIBLE_DEVICES=0 \
python extract_feat_py/extract_feat.py \
 -c config/cfg_coco.py \
 -p groundingdino_swint_ogc.pth \
 --anno_path /ABSOLUTE/PATH/OF/YOUR_CUSTOM_COCO_STYLE_TRAIN_ANNO_FILE_PATH \
 --image_dir TRAIN_IMAGES_ROOT \
 --coco_val_path /ABSOLUTE/PATH/OF/YOUR_CUSTOM_COCO_STYLE_TRAIN_ANNO_FILE_PATH \
 --feature_save_path FEATRURE_SAVE_ROOT/TASK_NAME
