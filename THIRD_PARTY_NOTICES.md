# Third-party provenance and license status

This snapshot preserves the author's current code without removing third-party attribution or altering bundled license files. No project-wide license has been assigned during documentation preparation.

## Bundled components

| Component | Evidence in this snapshot | Release status |
|---|---|---|
| `code/evaluate-main/` | Upstream README, AUTHORS, and Apache-2.0 LICENSE retained | Preserve its license and notices |
| `code/eval_math_rule/` | README identifies Omni-MATH-Rule and acknowledges Qwen2.5-MATH evaluation code | Exact source commit and applicable license still need confirmation |
| Mathematical grader | `evaluation/grader.py` credits MATH, CRITIC, PRM800K, ToRA, and DeepSeek-Math utilities | Preserve source comments; verify inherited terms before public distribution |
| Inference implementation | Author describes the workflow as derived from SEAL-based inference code | Exact upstream repository, revision, and applicable license still need confirmation |
| Omni-MATH subset | Supplied source question records retained unchanged | Verify upstream revision, filtering provenance, and data distribution terms |
| Training integration | External unmodified LLaMA-Factory installation | Do not vendor the whole framework; record version and original training configs |

The table is a provenance inventory, not a legal determination that every component is cleared for redistribution.

## Upstream references

- Hugging Face Evaluate: https://github.com/huggingface/evaluate
- Omni-MATH: https://github.com/KbsdJames/Omni-MATH
- Qwen2.5-MATH: https://github.com/QwenLM/Qwen2.5-Math
- LLaMA-Factory: https://github.com/hiyouga/LLaMA-Factory

The documentation organization was inspired by the [LogicTree repository](https://github.com/MIRALab-USTC/LogicTree-Improving-Complex-Reasoning-of-LLMs-via-Instantiated-Multi-step-Synthetic-Logical-Data). No LogicTree source code or assets were copied.

## Before publishing

Choose a license for original contributions, verify the licenses and versions of inherited components and source data, and add the required notices. Do not label the entire repository MIT or Apache-2.0 solely because one bundled dependency uses that license.
