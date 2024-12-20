CUDA_VISIBLE_DEVICES=0 \
bash train_dist.sh \
 1 \
 config/cfg_odinw_coop_lora.py \
 config/train_CUSTOM_TASK1.json  \
 OUTPUT_DIR_TASK1 \
 ./groundingdino_swint_ogc.pth



CUDA_VISIBLE_DEVICES=0 \
bash train_dist.sh \
 1 \
 config/cfg_odinw_coop_lora.py \
 config/train_CUSTOM_TASK2.json  \
 OUTPUT_DIR_TASK2 \
 ./groundingdino_swint_ogc.pth

 ### 往后可以添加更多task的启动代码
