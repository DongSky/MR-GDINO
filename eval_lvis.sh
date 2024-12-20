CUDA_VISIBLE_DEVICES=0 \
python test_ap_on_lvis_single.py \
 -c config/cfg_coco.py \
 -p output_logdir/odinw_vehicle_coop_lora_from_ogc_5_shot/checkpoint.pth \
 --anno_path /data/dongbowen/GroundingDINO-main/coco/lvis_v1_minival.json \
 --image_dir /data/dongbowen/GroundingDINO-main/coco/ \
 --coco_val_path /data/dongbowen/GroundingDINO-main/coco/lvis_v1_minival.json \
 --use_moe_lora \
 --use_coop \
#  --use_retrieve
#  --use_zira
#  --use_coop \
#  --use_moe_lora \
#  --use_retrieve