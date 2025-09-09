MAX_PIXELS=1003520 \
CUDA_VISIBLE_DEVICES=0 \
CUDA_LAUNCH_BLOCKING=1 \
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \

swift sft \
    --model /home/zhengzhentao/MegaCQA/fine_tuning_ChartMind_New/output/stage1/v2-20250908-192027/checkpoint-126 \
    --model_type qwen2_5_vl \
    --train_type lora \
    --dataset /home/zhengzhentao/MegaCQA/fine_tuning_ChartMind_New/train.jsonl \
    --split_dataset_ratio 0.3 \
    --torch_dtype bfloat16 \
    --attn_impl flash_attn \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --learning_rate 5e-5 \
    --gradient_accumulation_steps 8 \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 10 \
    --logging_steps 5 \
    --max_length 16384 \
    --output_dir output/stage2_lora \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 8 \
    --load_from_cache_file false \
    --truncation_strategy delete \
    --lora_rank 32 \
    --lora_alpha 64 \
    --lora_dropout 0.05 \
    --target_modules all-linear \
