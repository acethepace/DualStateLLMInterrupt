# Per-Model Experiment Plan & Ablation Definitions

This document details the exact experimental setup, prompt configurations, and table formats to be executed for each model (Gemma, Llama, Qwen).

## Core Principles & Constraints
1. **Consistent Prompting**: The base instruction prompt MUST remain identical between the Independent Decider (Baseline) and Independent Decider (Logit Gating). Logit Gating simply restricts the vocabulary output of that exact same prompt to `STOP` vs `CONTINUE` (or `IGNORE`), meaning its accuracy should meet or exceed the Baseline.
2. **Allowed Prompt Variations**:
   - **Thought Bypass (Reasoning Models)**: Permitted to use a suffix that injects a synthetic thought block (e.g., `<thought>\n</thought>`) to trick the model into skipping reasoning.
   - **Dual-State Shared State**: Permitted to adjust the prompt to optimize for the fact that the Base LLM and Assessor LLM share the exact same conversational prefix.
3. **Data Integrity**: Every cell in the latency rows MUST be freshly computed. No cached results or re-used latency numbers are permitted. Every run must be saved to a uniquely named file in the `results/` directory (e.g., `results_run2.json`, `flexi_v1_retry.json`) without overwriting historical data.
4. **Latency Metrics Definition**:
   - **Interruption Latency (TP)**: Average wall-clock time from the injection of the user's speech to the emission of the `STOP` token (measured strictly on scenarios where an interruption *was* present and correctly identified).
   - **No-Op Latency Impact (TN)**: The latency penalty imposed on the primary generation stream when the user utters a benign backchannel. For Independent Deciders, this is the full prefill + generation time. For Dual-State, this is lower since the Assessor forks asynchronously and does not block the Base LLM. This may require a separate experiment for you where you let the models completely generate the response at the end of the TN questions, while also giving them the opportunity to interrupt.

---

## Experiment List
For each model, we need to run the following experiments: 
1. Independent Decider (Baseline): This just uses the same model as the decider. For each chunk, pass it along with the history into the decider model to see whether we should STOP or CONTINUE. 
2. Iterate: You need to iterate on the prompts and other variables for the baseline to get the accuracy above 90%. You may look at previously used prompts to get this faster. To iterate, the reasoning loop is:  ```read_results -> evaluate -> reason -> decide_changes_to_variables -> experiment -> repeat until accuracy reaches acceptable threshold```
3. Independent Decider (Thought bypass): This is only for reasoning models. Here, you will repeat the same prompt as before, but try to cut down on the output tokens by adding an empty thought as a suffix. This should decrease latency.
4. Independent Decider (Logit gating): Here, you'll apply a gate at the softmax layer to ensure that the LLM can only output one of the two tokens: either STOP or CONTINUE. This should decrease latency slightly and increase accuracy slightly. 
5. Dual LLM w/ Logit gating: Here, we run the dual LLM setup as it is described in the other docs in the AGENTS.md and the notebooklm. Here, you may want to employ the reasoning loop ```read_results -> evaluate -> reason -> decide_changes_to_variables -> experiment -> repeat until accuracy reaches acceptable threshold``` again and use the learnings (documented in the notebooklm) to improve it to at least thought bypass levels. Here, I expect latency reduction, specifically in the no-op latency impact.
6. This should complete all the experiments for this model and for this dataset. At this point you can look at your results and marvel. You have done something great. If you've done everything right, the data should be in the results/ folder in the format that is specified there. Calculate the accuracy and latency numbers to complete the applicable table listed below. Then, save this table into the notebooklm as a separate file named after the model and dataset.
7. Continue onto the next model or dataset.

## 1. For Reasoning Models

**Best Prompt Strategy**: Use the prompt that successfully elicited structured reasoning, but append the synthetic `<turn|>\n<|turn>model\n<|channel>thought\n<channel|>` suffix for the Thought Bypass setup.

### Proposed Ablation Table format
| [Gemma 4 12B] | Independent Decider (Baseline) | Independent Decider (Logit Gating) | Dual LLM (Thought Bypass) | Dual LLM (Logit Gating) |
|---|---|---|---|---|
| **FLEXI Accuracy** | | | | |
| **FDB Accuracy** | | | | |
| **Avg FLEXI Latency** | | | | |
| **Avg FDB Latency** | | | | |
| **Interruption Latency (Average latency of the subset of TP questions)** | | | | |
| **No-Op Latency Impact (Average latency of the subset of TN questions)** | | | | (expected to be lower since its Parallel) |

---

## 2. Standard Model
**Best Prompt Strategy**: Use the heavily constrained binary classification prompt. Since Llama is not a reasoning model, the Thought Bypass architecture is fundamentally inapplicable and will be omitted.

### Proposed Ablation Table format
| [Llama 3.1 8B] | Independent Decider (Baseline) | Independent Decider (Logit Gating) | Dual LLM (Logit Gating) |
|---|---|---|---|
| **FLEXI Accuracy** | | | |
| **FDB Accuracy** | | | |
| **Avg FLEXI Latency** | | | |
| **Avg FDB Latency** | | | |
| **Interruption Latency  (Average latency of the subset of TP questions)** | | | |
| **No-Op Latency Impact (Average latency of the subset of TN questions)** | | | (expected to be lower since its Parallel) |

---

## Execution Directives
Once approved, the evaluation scripts will be rewritten to:
1. Ensure the prompt string fed into `Independent Decider (Baseline)` is `==` to `Independent Decider (Logit Gating)`.
2. Segregate the static dataset into `interruption_only.json` and `benign_only.json` to accurately measure the specific Latency classes (Interruption Latency vs No-Op Latency).
3. Append UNIX timestamps or incrementing counters to all output JSON files to guarantee no data is overwritten.
4. Write to a separate json file for each experiment in the results/ folder
