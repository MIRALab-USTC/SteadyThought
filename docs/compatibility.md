# Compatibility and validation

## Current entry points

The supported interface is three shell stages under `scripts/`: thought identification, trajectory synthesis, and preference-data construction. Each script declares its own parameters and switches to `code/` before calling Python. There is no shared shell configuration dependency.

The old seven-stage entry points were backed up to `/data/zypeng/github/SteadyThought-backup-7steps-20261016` before replacement. This integration does not modify the Python algorithms, model weights, existing experiment outputs, or raw questions. `source_manifest.json` records the initial release copy, not every later author edit. Current integration checks are recorded in `release_validation.json`.

## Important constraints

- **Working directory:** inference loads the bundled exact-match metric relative to `code/`. The stage scripts handle this automatically.
- **Source consistency:** generation and response preparation share the same source-question file in stage 1. Do not independently reorder intermediate files. Stage 3 checks that thought inputs and synthesis results have equal row counts and identical prompts before building pairs.
- **Correctness labels:** `prepare_responses.py` writes original correctness as `mv_eval`. Thought inputs retain this original-answer label; the separate synthesis evaluation has the new continuation label. The two must not be interchanged.
- **Model paths:** the existing Python templates/EOS handling relies on `DeepSeek` or `Qwen` in model/input paths. Keep recognizable model names in the output directory. Other families need a separate adaptation.
- **Segmentation filenames:** the segmentation CLI uses `--base_path` and derives `_segmented-<threshold>.jsonl` automatically. Keep the threshold spelling consistent across stages 1 and 2, such as `2.2`.
- **Entropy coverage:** the entropy implementation truncates model input to 14,336 tokens while slicing paragraphs from the original text. Tail records can have incomplete arrays. Independently tokenized paragraph boundaries can also differ from full-text tokenization. Neither behavior is repaired here.
- **Entropy labels:** the math entropy branch reads `graded_list`; do not use its historical `result` field as original correctness. Use evaluated predictions or prepared responses instead.
- **Boundary handling:** short-paragraph handling and the original `think differenly` marker spelling are unchanged.
- **Bias duration:** the active vLLM path uses a static `logit_bias` map. `--logit_adjustment_max_len` affects the result folder name but does not activate the commented length-aware logits processor.
- **CLI booleans:** use `--completion True` for prepared thought prompts. Omit it for normal solving; passing the string `False` can still be treated as true.
- **Pair selection:** the existing original-correct penultimate-thought exclusion, successful-synthesis condition, `isthink` condition, and 6144-token branch limits remain. No preference-selection rule was changed.
- **Overwrite behavior:** stages overwrite outputs; they do not automatically resume incomplete files. `run_all.sh` stops on failure.
- **Historical launcher:** `code/conduct_stpo_dataset.sh` is retained historical material, not the supported entry point.

## Answer extraction

`--data_path` selects the actual JSONL. `--dataset` is also passed to answer extraction; preserve historical labels when comparing old experiments. Generic math labels usually fall back to the last number when no supported final-answer marker is present. The literal `omni-math` label disables this fallback, so changing labels can change accuracy.

## Validation scope

The three-stage chain was exercised using synthetic fixtures, real shell orchestration and the current processing functions. Model generation, entropy values, answer grading and tokenizer lengths were mocked. Tests cover command order, GPU/decoding arguments, paths with spaces, independent stage reruns, prompt/length mismatch protection, and stopping after failure.

No new real-GPU generation, clean GPU installation, preference training, or paper-score reproduction is claimed. Complete training recipes and final license/provenance checks remain separate release tasks.
