rm -rf Qwen3-0.6B-Base-dummy
WANDB_MODE="disabled" \
TORCH_DISTRIBUTED_DEBUG="info" \
CUDA_VISIBLE_DEVICES="0" \
python3.10 profiling.py \
--model_name_or_path "Qwen/Qwen3-0.6B-Base" \
--per_device_train_batch_size 8 \
--gradient_accumulation_steps 1 \
--output_dir Qwen3-0.6B-Base-dummy \
--bf16 --do_train --do_eval false --max_steps 2 \
--block_size 4096 \
--logging_steps 1 \
--learning_rate 2e-5 \
--warmup_steps 20 \
--save_steps 100 \
--save_total_limit 5 \
--gradient_checkpointing true \
--torch_dtype bfloat16 \
--ddp_find_unused_parameters false \
--dataloader_num_workers 5 \
--dataloader_prefetch_factor 5 \
--attn_implementation "flex_attention" \
--remove_unused_columns false