CUDA_VISIBLE_DEVICES=0 \
python tools/inference_on_a_image.py \
 -c config/cfg_coco.py \
 -p /PATH/OF/THA/LAST/CHECH_POINT \
 --image_path /ABSOLUTE/PATH/OF/YOUR_EVAL_IMAGE \
 --text_prompt "CLASS_NAME1 . CLASS_NAME2 . CLASS_NAME3 . ... ."\
 --base_model_path groundingdino_swint_ogc.pth \
 --use_double_model \
 --use_moe_lora \
 --use_coop \
 --use_retrieve \
 -o OUTPUT_DIR
