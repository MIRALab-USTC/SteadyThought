# Data formats

All dataset files in this workflow use **JSON Lines**: one complete JSON object per line, with no enclosing list. Ordering matters because the current preparation scripts align files by row position.

## Source questions

`data/omni_math/omni_math_level237.jsonl` has these core fields:

| Field | Meaning |
|---|---|
| `problem` | Original mathematical question |
| `answer` | Reference final answer |
| `solution` | Reference solution |
| `level` | Selected difficulty level |
| Other metadata | Domain, difficulty, source, and other supplied attributes |

The local-file inference loader accepts `question` instead of `problem` when present. It requires `answer` and `solution` in either case. For thought inputs, `question` is the already-formatted continuation prompt.

## Evaluated predictions

`math_eval_rewrite.jsonl` retains the question, prompt, references, and generations, and adds grading fields:

| Field | Meaning |
|---|---|
| `prompt` | Exact input passed to generation |
| `model_generation` | List of generated continuation strings; the tutorial requests one |
| `all_pred` | Extracted answers for the generations |
| `all_eval` | Correctness labels for the extracted answers |
| `mv_pred` | Selected majority-vote answer |
| `mv_eval` | Correctness of the selected answer |
| `mv_index` | Index of the selected generation |

With one generation, `mv_eval` represents that generation's extracted-answer correctness. It is a mathematical grading result, not an independent judge's claim that every reasoning step is correct.

## Prepared responses

`prepare_responses.new_dataset` writes:

- `prompt`, `model_generation`, `problem`, `answer`, `solution`, `level`;
- `<answer>`: the generated text following `</think>`, if available;
- `isthink`: whether the generated response contains `</think>`;
- `mv_eval`: original response correctness, copied from evaluation `mv_eval`.

The current CLI writes `mv_eval` directly. Historical files may use `result`, but no alias conversion is needed for the current three-stage pipeline. The original label remains attached to thought inputs; it is not the correctness of their newly synthesized continuations.

## Token entropy records

`compute_token_entropy.py` writes one record per paragraph, not one per problem:

| Field | Meaning |
|---|---|
| `original_content_index` | Zero-based index of the source response |
| `paragraph_index` | Zero-based paragraph index within that response |
| `paragraph_text` | Paragraph text, retaining the delimiter where applicable |
| `token_entropies` | Predictive token-distribution entropies assigned to the paragraph |
| `input_ids` | Corresponding token IDs, subject to the current truncation behavior |
| `problem` | Original question |
| `result` | Historical auxiliary label; not populated from prepared-response `result` by the current math branch |

Do not use entropy-record `result` to recover original accuracy. Use prepared responses or evaluated predictions.

## Segmented thoughts

`segment_thoughts.py` writes one record per grouped response:

- `split_segments`: ordered thought strings;
- `steps_split_indices`: paragraph indices grouped into each thought;
- `switch_count`: **number of thoughts**, despite the historical field name;
- `problem` and `result`: carried metadata.

If a response has `n` thoughts, the number of boundaries between thoughts is `n - 1`, not `switch_count`.

## Thought inputs

`prepare_thoughts_inputs.py` uses `process_json_file_new_dataset_3` and writes:

| Field | Meaning |
|---|---|
| `question` | Serialized question + selected thought; directly used as the continuation prompt |
| `question_index` | Zero-based originating response index |
| `split_index` | Zero-based selected thought index |
| `segments` | All thought strings from the original response |
| `mv_eval` | Original full-response correctness, not the synthesized continuation's correctness |
| `answer`, `solution`, `problem`, `level` | Source references and metadata |
| `<answer>`, `isthink` | Original generated final-answer text and reasoning-close status |

The script does not emit the final thought or any entries for a single-thought response. For later thoughts it reconstructs the question template and includes only the selected thought, not earlier thoughts.

## STPO pairs

```json
{"instruction":"<formatted question and current thought>","chosen":"<correct synthesized continuation and EOS>","rejected":"<original later thoughts and final answer and EOS>","split_index":0,"answer":"<reference answer>","level":2}
```

This is a schematic schema example, not a literal training record.

| Field | Meaning |
|---|---|
| `instruction` | Shared context: question + current thought |
| `chosen` | Correct synthesized continuation, with the model-specific EOS token appended |
| `rejected` | Original continuation after the selected thought, plus its final-answer portion and EOS |
| `split_index` | Selected thought index |
| `answer` | Reference answer metadata |
| `level` | Difficulty-level metadata |

The preference is about continuation behavior, not simply “correct whole answer versus incorrect whole answer.” An original response can be correct yet contain unnecessary switching.

The builder requires successful synthesis and `isthink == true`, excludes the original-correct penultimate-thought case, and filters either branch if instruction + continuation exceeds 6144 tokens. See [compatibility notes](compatibility.md) for source constraints.
