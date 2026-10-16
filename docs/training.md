# STPO training

## What this snapshot includes

The data-construction workflow produces JSONL records with `instruction`, `chosen`, and `rejected`. The optimization objective described in the paper is **Steady Thought Preference Optimization (STPO)**, applied to continuations conditioned on the question and current thought.

The author used an unmodified LLaMA-Factory installation for training. That framework is an external dependency, not copied into this repository. This snapshot does not yet contain the original training YAML, its exact framework commit, DeepSpeed configuration, or checkpoint-selection recipe. It therefore does not claim to reproduce training with one command.

## Register the dataset

Keep LLaMA-Factory in a separate training checkout and environment. Copy the generated `stpo_pairs.jsonl` into its data directory, then merge the following entry into its existing `data/dataset_info.json`:

```json
{
  "steadythought_stpo": {
    "file_name": "stpo_pairs.jsonl",
    "ranking": true,
    "columns": {
      "prompt": "instruction",
      "chosen": "chosen",
      "rejected": "rejected"
    }
  }
}
```

This is a dataset-registration example. Do not replace the entire existing `dataset_info.json` with it. Verify the registration against the LLaMA-Factory version used for your training run.

Current `instruction` strings already include serialized model-specific chat tokens and the selected thought, and continuations contain EOS tokens. The training template and special-token behavior must preserve this exact shared context rather than silently adding a second chat template. The appropriate original configuration is still required; registering the fields alone does not guarantee identical training behavior.

## Reported settings

The paper's implementation appendix gives:

| Model | Learning rate | Beta | Gamma | Adaptation |
|---|---:|---:|---|---|
| DeepSeek-R1-Distill-Qwen-1.5B | 5e-7 | 2.0 | 1.0 | LoRA not specified for this model |
| Qwen3-8B | 3e-5 | 2.0 | Not explicitly specified in the appendix paragraph | LoRA |
| DeepSeek-R1-Distill-Qwen-14B | 13e-5 | 3.0 | Not explicitly specified in the appendix paragraph | LoRA |
| Qwen3-32B | 3e-5 | 2.0 | Not explicitly specified in the appendix paragraph | LoRA |

Shared reported settings include batch size 256, context length 6144, Adam, warm-up ratio 0.1, cosine learning-rate scheduling, and DeepSpeed ZeRO-3 with CPU offload.

This table summarizes the manuscript; it is not a complete training recipe. LoRA rank/targets, epochs, accumulation, precision, exact preference-loss settings, and model selection still need the author's original configurations. Do not infer missing values from the table.

## Resources and acknowledgements

- LLaMA-Factory: https://github.com/hiyouga/LLaMA-Factory
- Dataset registration format: https://github.com/hiyouga/LLaMA-Factory/blob/main/data/README.md

These upstream resources are pointers, not substitutes for recording the original training version.
