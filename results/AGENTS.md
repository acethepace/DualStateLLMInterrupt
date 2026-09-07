# Evaluation Results Structure

This directory stores all benchmark outputs from the dual-state LLM architecture evaluations.

## Directory Layout
The folder structure is strictly parameterized to support robust ablation studies for the research paper:
`results/[model_name]/[dataset_name]/[version_configuration]/`

For example: `results/unsloth_gemma-4-12b-it/flexi/v2_prefill/flexi_results.json`

## Versions
- **v1**: Unrestricted generation. Allows the model to use `max_new_tokens=15` to output its internal chain of thought before providing a semantic classification. Yields high accuracy but unacceptable latency (~2500ms).
- **v2**: Standard 1-token override. Restricts generation to 1 token via prompt engineering ("output STOP as your VERY FIRST token"). Guaranteed to fail on instruct models like Gemma because they cannot bypass their `<|channel>` tokens.
- **v2_prefill**: KV-Cache Exploit. Injects a fake, pre-closed thought block into the KV cache, tricking the model into skipping reasoning. Achieves real-time latency (~400ms) but drops accuracy significantly since the model is forced to hallucinate without reasoning context.

*Note: All results feed directly into NotebookLM analysis for the Dual-State Architecture EB-2 NIW paper.*
