#!/bin/bash

# Fixed training script with better model handling and memory optimization
MODELS=(
    #meta-llama/Llama-3.2-3B-Instruct
    google/gemma-2-2b-it
    # meta-llama/Llama-3.2-1B-Instruct
    # meta-llama/Llama-3.1-8B-Instruct
    # meta-llama/Llama-2-7b-chat-hf
    # google/gemma-3-1b-it
    # Qwen/Qwen3-4B
    # Qwen/Qwen2.5-3B-Instruct
)
MODEL_TYPES=(
    #llama3_2
    gemma2
    # llama3_2
    # llama3_1
    # llama
    # gemma3_text
    # qwen3
    # qwen2_5
)

# Check if local model exists, otherwise use remote
check_and_download_model() {
    local model_name=$1
    local local_path="./gemma-2-2b-it-local"
    
    if [[ "$model_name" == "google/gemma-2-2b-it" && -d "$local_path" ]]; then
        echo "Using local model from $local_path"
        echo "$local_path"
    else
        echo "$model_name"
    fi
}

# Function to safely run lora_to_full.py
run_lora_to_full() {
    local output_dir=$1
    echo "Attempting to convert LoRA to full model from: $output_dir"
    
    if python3 lora_to_full.py "$output_dir" --last; then
        echo "LoRA to full conversion successful"
        return 0
    else
        echo "LoRA to full conversion failed, skipping evaluation"
        return 1
    fi
}

# Function to safely run evaluation
run_evaluation() {
    local output_dir=$1
    local full_model_path="${output_dir}-full"
    
    if [[ ! -d "$full_model_path" ]]; then
        echo "Full model not found at $full_model_path, skipping evaluation"
        return 1
    fi
    
    echo "Running evaluation on: $full_model_path"
    
    # Use absolute path for local model evaluation
    if [[ "$full_model_path" == *"-full" ]]; then
        full_model_path=$(realpath "$full_model_path")
    fi
    
    CUDA_VISIBLE_DEVICES=0,1 evalplus.evaluate \
    --dataset mbpp \
    --model "$full_model_path" \
    --root evalplus_results/ \
    --backend vllm \
    --tp 2 \
    --temperature 0.5 \
    --n-samples 50 \
    --parallel 64
}

gsutil cp gs://delivery-rnd-colab/models/2410-combined.jsonl /tmp/data.jsonl
DATASET_PATH=${DATASET_PATH:-/tmp/data.jsonl}

export PYTHONPATH=$PYTHONPATH:$(pwd)/plugins  

for i in "${!MODELS[@]}"; do
    MODEL="${MODELS[$i]}"
    MODEL_TYPE="${MODEL_TYPES[$i]}"
    nproc_per_node=2
    MODEL_NAME=$(basename "$MODEL")
    OUTPUT_DIR=models/${MODEL_NAME}/2410-dft
    SAVE_STEPS=10

    # Get the actual model path (local or remote)
    ACTUAL_MODEL_PATH=$(check_and_download_model "$MODEL")
    
    # Set attn_impl depending on MODEL_NAME
    if [[ "$MODEL_NAME" == *gemma* ]]; then
        ATTN_IMPL="eager"
    else
        ATTN_IMPL="flash_attn"
    fi

    echo "Training with model: $ACTUAL_MODEL_PATH"
    echo "Output directory: $OUTPUT_DIR"

    # Memory optimization: Reduce batch size and gradient accumulation
    # Also enable gradient checkpointing and memory efficient attention
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    CUDA_VISIBLE_DEVICES=0,1 \
    NPROC_PER_NODE=$nproc_per_node \
    TORCHDYNAMO_RECOMPILE_LIMIT=16 \
    TORCHDYNAMO_CACHE_SIZE_LIMIT=16 \
    TORCHDYNAMO_ACCUMULATED_RECOMPILE_LIMIT=256 \
    TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1 \
    swift sft \
        --model "$ACTUAL_MODEL_PATH" \
        --use_hf True \
        --train_type lora \
        --model_type $MODEL_TYPE \
        --loss_type dft \
        --external_plugins plugins/dft_loss.py \
        --dataset $DATASET_PATH \
        --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
        --torch_dtype bfloat16 \
        --num_train_epochs 1 \
        --learning_rate 1e-4 \
        --lora_rank 16 \
        --lora_alpha 32 \
        --target_modules all-linear \
        --weight_decay 0.01 \
        --lr_scheduler_type constant \
        --warmup_ratio 0.05 \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 8 \
        --eval_steps $SAVE_STEPS \
        --save_steps $SAVE_STEPS \
        --save_only_model \
        --save_total_limit 10 \
        --logging_steps 5 \
        --max_length 8000 \
        --output_dir $OUTPUT_DIR \
        --warmup_ratio 0.05 \
        --dataloader_num_workers 2 \
        --truncation_strategy 'right' \
        --torch_dtype bfloat16 \
        --seed 44 \
        --data_seed 44 \
        --split_dataset_ratio 0.05 \
        --dataset_shuffle true \
        --report_to tensorboard \
        --logging_first_step true \
        --logging_steps 1 \
        --attn_impl $ATTN_IMPL \
        --gradient_checkpointing true \
        --max_grad_norm 1.0

    # Check if training completed successfully
    if [[ $? -eq 0 ]]; then
        echo "Training completed successfully"
        
        # Try to convert LoRA to full model
        if run_lora_to_full "$OUTPUT_DIR"; then
            # Run evaluation if conversion was successful
            run_evaluation "$OUTPUT_DIR"
        fi
    else
        echo "Training failed, skipping conversion and evaluation"
    fi
done
