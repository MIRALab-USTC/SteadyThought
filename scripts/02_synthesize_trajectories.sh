#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/../code"

model="DeepSeek-R1-Distill-Qwen-1.5B"
dataset="omni_level237"
# Place the checkpoint in models/<model>, or set your local checkpoint path.
model_path="../models/$model"
output_dir="../outputs/$model/$dataset"
entropy_threshold="2.2"
segmented_path="$output_dir/entropy-$model-$dataset-dataset_segmented-$entropy_threshold.jsonl"
response_path="$output_dir/$model-$dataset-dataset.jsonl"
thought_path="$output_dir/$model-$dataset-tobesynthesized.jsonl"
method="ours"

export VLLM_USE_V1=0
export PYTHONUTF8=1

# Prepare question + current-thought inputs.
python prepare_thoughts_inputs.py \
    --json_path "$segmented_path" \
    --json_path_2 "$response_path" \
    --output_path "$thought_path"

# Generate and evaluate controlled continuations.
CUDA_VISIBLE_DEVICES=4,5 python run_inference.py \
    --model_name_or_path "$model_path" \
    --data_path "$thought_path" \
    --dataset "${dataset}_thought_inputs" \
    --completion True \
    --save_dir "$output_dir/trajectory_synthesis" \
    --max_tokens 12000 \
    --temp 0.6 \
    --top_p 0.95 \
    --tensor_parallel_size 2 \
    --logit_adjustment \
    --logit_adjustment_tokens "$method" \
    --logit_adjustment_value -1000000 \
    --logit_adjustment_max_len 20000
