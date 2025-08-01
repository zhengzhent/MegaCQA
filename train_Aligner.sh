MAX_PIXELS=1003520 \
CUDA_VISIBLE_DEVICES=0 \
CUDA_LAUNCH_BLOCKING=1 \
swift sft \
    --model /home/zhengzhentao/MegaCQA/new_Qwen \
    --model_type qwen2_5_vl \
    --train_type full \
    --dataset /home/zhengzhentao/MegaCQA/visual_qa_dataset.json  \
    --split_dataset_ratio 0.01 \
    --torch_dtype bfloat16 \
    --freeze_vit true \
    --freeze_llm true \
    --freeze_aligner false \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --learning_rate 5e-6 \
    --gradient_accumulation_steps 8 \
    --eval_steps -1 \
    --save_steps 1000 \
    --save_total_limit 10 \
    --logging_steps 5 \
    --output_dir output \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 8 \
    --max_length 8192 \
    --attn_impl flash_attn \
#    --deepspeed zero2

