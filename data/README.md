# Source data

- `omni_math/omni_math_level237.jsonl`: 1,051 Omni-MATH problems from difficulty levels 2, 3, and 7.
- `omni_math/omni_math_level237_test.jsonl`: the first 10 problems, for a small-scale test.

Both files use JSON Lines: one problem per line, with `problem`, `answer`, `solution`, `level`, and source metadata. Preserve row order when matching problems with evaluated responses.

Select the input using `data_path` in `scripts/01_identify_thoughts.sh`. Generated files are saved under `outputs/<model>/<dataset>/`, which is excluded from Git.

Source: [Omni-MATH project](https://github.com/KbsdJames/Omni-MATH) · [Original dataset](https://huggingface.co/datasets/KbsdJames/Omni-MATH).
