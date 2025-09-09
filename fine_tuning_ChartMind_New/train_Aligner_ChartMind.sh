MAX_PIXELS=1003520 \
CUDA_VISIBLE_DEVICES=0 \
CUDA_LAUNCH_BLOCKING=1 \
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \

swift sft \
    --model /home/zhengzhentao/MegaCQA/new_Qwen \
    --model_type qwen2_5_vl \
    --train_type full \
    --dataset /home/zhengzhentao/MegaCQA/fine_tuning_ChartMind_New/train.jsonl  \
    --split_dataset_ratio 0.3 \
    --torch_dtype bfloat16 \
    --freeze_vit true \
    --freeze_llm true \
    --freeze_aligner false \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --learning_rate 5e-6 \
    --gradient_accumulation_steps 8 \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 10 \
    --logging_steps 5 \
    --output_dir output/stage1  \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 8 \
    --max_length 24576 \
    --load_from_cache_file false \
    --truncation_strategy delete \
    --attn_impl flash_attn \
#    --resume_from_checkpoint output/stage1/v0-20250908-142545/checkpoint-50 \
#    --ignore_data_skip true \

