# Source data

`omni_math/omni_math_level237.jsonl` is the supplied Omni-MATH difficulty-level 2/3/7 subset used as the source pool in this construction workflow. The snapshot contains 1,051 JSONL records.

Each row retains the supplied `problem`, `answer`, `solution`, `level`, and other metadata. Data is copied unchanged, not re-fetched or re-filtered during release preparation.

The exact upstream revision and subset-filtering recipe should be recorded before claiming a fully reconstructible data release. Preserve the original ordering when pairing questions with evaluated responses.

Upstream resources:

- Omni-MATH project: https://github.com/KbsdJames/Omni-MATH
- Dataset: https://huggingface.co/datasets/KbsdJames/Omni-MATH

Confirm the applicable source-data and derived-data distribution terms before public release. Small intermediate examples are under `examples/`; runtime outputs should be written under `outputs/`.
