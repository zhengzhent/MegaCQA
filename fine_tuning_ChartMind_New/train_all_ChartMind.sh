MAX_PIXELS=1003520 \
CUDA_VISIBLE_DEVICES=0 \
CUDA_LAUNCH_BLOCKING=1 \
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \

swift sft \
    --model /home/zhengzhentao/MegaCQA/fine_tuning_ChartMind_New/output/stage1/v2-20250908-192027/checkpoint-126 \
    --model_type qwen2_5_vl \
    --train_type full \
    --dataset /home/zhengzhentao/MegaCQA/fine_tuning_ChartMind_New/train.jsonl \
    --split_dataset_ratio 0.3 \
    --torch_dtype bfloat16 \
    --attn_impl flash_attn \
    --freeze_vit false \
    --freeze_llm false \
    --freeze_aligner false \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --learning_rate 5e-6 \
    --gradient_accumulation_steps 8 \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 10 \
    --logging_steps 5 \
    --max_length 8192 \
    --output_dir output/stage2 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 8 \
    --load_from_cache_file false \
    --truncation_strategy delete \
#    --deepspeed zero2