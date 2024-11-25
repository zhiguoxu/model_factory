export PYTHONPATH=$PYTHONPATH:/mnt/cache/xuzhiguo/workspace/

job_name=llama_factory_lora
scancel -n $job_name
log_file=sft.log
rm $log_file

nohup srun -p pat_rd --gres=gpu:1 --job-name=$job_name accelerate launch workflow.py \
    --do_train \
    --pretrained_model_name_or_path /mnt/lustrenew/share/qitianlong/models/internlm2-chat-7b \
    --train_data_path /mnt/cache/xuzhiguo/workspace/batch_test/data/tool_learn_eval-fcdata_zh_train_luban.json \
    --cutoff_len 4096 \
    --val_size 0.2 \
    --finetuning_type lora \
    --lora_target wqkv \
    --output_dir output_1 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --lr_scheduler_type cosine \
    --logging_steps 5 \
    --save_steps 10 \
    --learning_rate 5e-5 \
    --num_train_epochs 1.0 \
    --fp16 \
    --save_total_limit 3 \
    2>&1 | tee $log_file &

# overwrite_output_dir