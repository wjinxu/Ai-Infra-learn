# DSpark

来源：`~/projects/DeepSpec/deepspec/`

## eval/

| 文件 | 重点 |
|------|------|
| `base_evaluator.py` | `generate_decoding_sample()`，`verify_draft_tokens()` |
| `evaluator.py` | `_propose()` / `_update()` hook |
| `draft_ops.py` | `forward_dspark_draft_block()`，`build_dspark_proposal()` |

## modeling/

| 文件 | 重点 |
|------|------|
| `markov_head.py` | `VanillaMarkov.sample_block_tokens()` |
| `common.py` | `extract_context_feature()` |
| `qwen3/modeling.py` | `Qwen3DSparkModel._forward_backbone()` |
| `gemma4/modeling.py` | `Gemma4DSparkModel._forward_backbone()` |
| `gemma4/config.py` | Gemma4 draft 配置生成 |

## utils/

| 文件 | 重点 |
|------|------|
| `sampling.py` | rejection sampling 工具函数 |
