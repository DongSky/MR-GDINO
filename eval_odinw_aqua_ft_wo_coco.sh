
CUDA_VISIBLE_DEVICES=5 \
python test_ap_on_odinw_aqua_wo_coco.py \
 -c config/cfg_odinw_coop_lora.py \
 -p output_logdir/odinw_aqua_coop_lora_from_ogc_10_shot/checkpoint.pth \
 --anno_path /data/dongbowen/data/odinw_13/Aquarium/Aquarium\ Combined.v2-raw-1024.coco/valid/annotations_without_background.json \
 --image_dir /data/dongbowen/data/odinw_13/Aquarium/Aquarium\ Combined.v2-raw-1024.coco/valid \
 --coco_val_path /data/dongbowen/data/odinw_13/Aquarium/Aquarium\ Combined.v2-raw-1024.coco/valid/annotations_without_background.json \
 --use_coop \
 --use_moe_lora \
 --use_retrieve 
#  --use_retrieve
#  --use_adapter
#  --use_coop \
#  --use_moe_lora \
#  --use_retrieve