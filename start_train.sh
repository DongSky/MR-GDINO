CUDA_VISIBLE_DEVICES=0 \
bash train_dist.sh \
 1 \
 config/cfg_odinw_coop_lora.py \
 config/train_CUSTOM.json  \
 OUTPUT_DIR \
 ./groundingdino_swint_ogc.pth
