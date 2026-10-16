# Data construction

[English homepage](../README.md) · [中文首页](../README.zh-CN.md) · [Data formats](data_formats.md)

Use the three entry points from the repository root:

```bash
bash scripts/01_identify_thoughts.sh
bash scripts/02_synthesize_trajectories.sh
bash scripts/03_build_stpo_dataset.sh
```

Or run `bash scripts/run_all.sh`. Parameters are set directly in each script, not in a shared config file.

| Stage | Original operations | Main artifacts |
|---|---|---|
| Identify switching points | Original inference, response preparation, token entropy, segmentation | `vanilla/math_eval_rewrite.jsonl`, model-named dataset, entropy and `_segmented-2.2.jsonl` files |
| Synthesize trajectories | Thought-input preparation and controlled inference | Model-named `tobesynthesized.jsonl`, `trajectory_synthesis/logit-adjustment/ours_value_-1000000.0_first20000/math_eval_rewrite.jsonl` |
| Build preference data | Alignment checks and preference-pair construction | `stpo_pairs.jsonl` |

Artifacts remain under `outputs/<model>/<dataset>/`. Their existing names and the order of all seven Python commands are preserved. Stage 1 uses one source-question file for both generation and metadata preparation; change its `data_path` to select a test or full dataset.

Use the same model, dataset, entropy threshold and output paths in subsequent stages. If synthesis decoding settings change the result directory name, update `result_path` in stage 3 accordingly.

Existing outputs are overwritten. Run stage 3 only after stage 2 has completed answer evaluation. Its row-count and exact-prompt checks prevent mismatched labels and continuations from being paired.

The previous seven-stage scripts were backed up outside the release directory before replacement. Python algorithms and model/data files were not changed by the consolidation. See [compatibility notes](compatibility.md) for inherited constraints and [training](training.md) for the separate training setup.
