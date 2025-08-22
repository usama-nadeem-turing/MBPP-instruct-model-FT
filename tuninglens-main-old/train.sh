MODELS=(
    #meta-llama/Llama-3.2-3B-Instruct
    google/gemma-2-2b-it
    # meta-llama/Llama-3.2-1B-Instruct
    # meta-llama/Llama-3.1-8B-Instruct
    # meta-llama/Llama-2-7b-chat-hf
    google/gemma-3-1b-it
    # Qwen/Qwen3-4B
    # Qwen/Qwen2.5-3B-Instruct
)
MODEL_TYPES=(
    #llama3_2
    gemma2
    # llama3_2
    # llama3_1
    # llama
    gemma3_text
    # qwen3
    # qwen2_5
)

gsutil cp gs://delivery-rnd-colab/models/2410-combined.jsonl /tmp/data.jsonl
DATASET_PATH=${DATASET_PATH:-/tmp/data.jsonl}

# For a quick smoke test, you can override before running:
#   DATASET_PATH=../one_example.jsonl bash train.sh

export PYTHONPATH=$PYTHONPATH:$(pwd)/plugins  
for i in "${!MODELS[@]}"; do
    MODEL="${MODELS[$i]}"
    MODEL_TYPE="${MODEL_TYPES[$i]}"
    nproc_per_node=1
    MODEL_NAME=$(basename "$MODEL")
    OUTPUT_DIR=models/${MODEL_NAME}/2410-dft
    SAVE_STEPS=10

    # Set attn_impl depending on MODEL_NAME
    if [[ "$MODEL_NAME" == *gemma* ]]; then
        ATTN_IMPL="eager"
    else
        ATTN_IMPL="flash_attn"
    fi

    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    CUDA_VISIBLE_DEVICES=0 \
    NPROC_PER_NODE=$nproc_per_node \
    TORCHDYNAMO_RECOMPILE_LIMIT=16 \
    TORCHDYNAMO_CACHE_SIZE_LIMIT=16 \
    TORCHDYNAMO_ACCUMULATED_RECOMPILE_LIMIT=256 \
    TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1 \
    swift sft \
        --model $MODEL \
        --model_type $MODEL_TYPE \
        --loss_type dft \
        --external_plugins plugins/dft_loss.py \
        --dataset "$DATASET_PATH" \
        --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
        --torch_dtype bfloat16 \
        --num_train_epochs 1 \
        --learning_rate 1e-4 \
        --lora_rank 32 \
        --lora_alpha 64 \
        --target_modules all-linear \
        --weight_decay 0.01 \
        --lr_scheduler_type constant \
        --warmup_ratio 0.05 \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 16 \
        --eval_steps $SAVE_STEPS \
        --save_steps $SAVE_STEPS \
        --save_only_model \
        --save_total_limit 50 \
        --logging_steps 5 \
        --max_length 15000 \
        --output_dir $OUTPUT_DIR \
        --dataloader_num_workers 4 \
        --seed 44 \
        --data_seed 44 \
        --split_dataset_ratio 0.05 \
        --report_to tensorboard \
        --logging_first_step true \
        --attn_impl $ATTN_IMPL

    python3 lora_to_full.py $OUTPUT_DIR --last
    
    CUDA_VISIBLE_DEVICES=0,1 evalplus.evaluate \
    --dataset mbpp \
    --model ${OUTPUT_DIR}-full \
    --root evalplus_results/ \
    --backend vllm \
    --tp 2 \
    --temperature 0.5 \
    --n-samples 50 \
    --parallel 64 
done