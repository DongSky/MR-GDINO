CUDA_VISIBLE_DEVICES=5 \
python test_ap_on_odinw_aerial_wo_coco.py \
 -c config/cfg_odinw_coop_lora.py \
 -p output_logdir/odinw_aerial_adapter_from_ogc_5_shot_cl_baseline/checkpoint.pth \
 --anno_path /data/dongbowen/data/odinw_13/AerialMaritimeDrone/large/valid/annotations_without_background.json \
 --image_dir /data/dongbowen/data/odinw_13/AerialMaritimeDrone/large/valid \
 --coco_val_path /data/dongbowen/data/odinw_13/AerialMaritimeDrone/large/valid/annotations_without_background.json \
 --save_path text_results/aerial_5_shot_adapter \
 --use_adapter \
#   --use_adapter \
#  --use_moe_lora \
#  --use_retrieve 
#  --use_coop \
#  --use_moe_lora \
#  --use_retrieve
