#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/../code"

model="DeepSeek-R1-Distill-Qwen-1.5B"
dataset="omni_level237"
model_path="/data/zypeng/models/$model"
data_path="../outputs/$model/$dataset/$model-$dataset-tobesynthesized.jsonl"
result_path="../outputs/$model/$dataset/trajectory_synthesis/logit-adjustment/ours_value_-1000000.0_first20000/math_eval_rewrite.jsonl"
output_path="../outputs/$model/$dataset/$model-$dataset-stpo.jsonl"

export PYTHONUTF8=1

# Do not combine labels or continuations from different samples.
python - "$data_path" "$result_path" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as source:
    thoughts = [json.loads(line) for line in source]
with open(sys.argv[2], encoding="utf-8") as source:
    predictions = [json.loads(line) for line in source]
if len(thoughts) != len(predictions):
    raise ValueError("Thought inputs and synthesized predictions have different lengths")
for index, (thought, prediction) in enumerate(zip(thoughts, predictions)):
    if thought["question"] != prediction["prompt"]:
        raise ValueError(f"Thought/synthesis prompt mismatch at row {index}")
PY

python build_stpo_dataset.py \
    --json_path "$data_path" \
    --json_path_2 "$result_path" \
    --output_path "$output_path" \
    --model_path "$model_path"

if [[ ! -s "$output_path" ]]; then
    printf 'No preference pairs were written. Inspect synthesis correctness and isthink labels.\n' >&2
    exit 1
fi
printf 'Final dataset: %s\n' "$output_path"
