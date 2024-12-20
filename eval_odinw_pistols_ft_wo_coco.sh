CUDA_VISIBLE_DEVICES=1 \
python test_ap_on_odinw_pistols_wo_coco.py \
 -c config/cfg_odinw_coop_lora.py \
 -p output_logdir/odinw_pistol_coop_lora_from_ogc_10_shot/checkpoint.pth \
 --anno_path /data/dongbowen/data/odinw_13/pistols/export/val_annotations_without_background.json \
 --image_dir /data/dongbowen/data/odinw_13/pistols/export \
 --coco_val_path /data/dongbowen/data/odinw_13/pistols/export/val_annotations_without_background.json \
 --use_moe_lora \
 --use_coop \
 --use_retrieve 
#  --use_moe_lora \
#  --use_retrieve 
#  --use_retrieve
#  --use_adapter \
#  --use_coop \
#  --use_moe_lora \
#  --use_retrieve