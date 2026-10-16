#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/../code"

model="DeepSeek-R1-Distill-Qwen-1.5B"
dataset="omni_level237"
# Place the checkpoint in models/<model>, or set your local checkpoint path.
model_path="../models/$model"
data_path="../data/omni_math/omni_math_level237.jsonl"
output_dir="../outputs/$model/$dataset"
response_path="$output_dir/$model-$dataset-dataset.jsonl"
entropy_path="$output_dir/entropy-$model-$dataset-dataset.jsonl"
entropy_threshold="2.2"

export VLLM_USE_V1=0
export PYTHONUTF8=1

# Generate and evaluate the original responses.
CUDA_VISIBLE_DEVICES=4,5 python run_inference.py \
    --model_name_or_path "$model_path" \
    --data_path "$data_path" \
    --dataset "$dataset" \
    --use_chat_format \
    --save_dir "$output_dir/vanilla" \
    --max_tokens 14336 \
    --temp 0.6 \
    --top_p 0.95 \
    --tensor_parallel_size 2

# Combine references and evaluated responses.
python prepare_responses.py \
    --data_path "$data_path" \
    --result_path "$output_dir/vanilla/math_eval_rewrite.jsonl" \
    --output_path "$response_path"

# Compute token entropy and identify thought switching points.
CUDA_VISIBLE_DEVICES=5 python compute_token_entropy.py \
    --dataset_path "$response_path" \
    --model_path "$model_path" \
    --output_path "$entropy_path"

python segment_thoughts.py \
    --base_path "$entropy_path" \
    --entropy_threshold "$entropy_threshold"
