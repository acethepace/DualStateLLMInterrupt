# Empirical Validation: V2 (1-Token Override)

## Overview
Following the V1 evaluation, the V2 experiment attempted to bypass the latency penalties of the model's internal reasoning by strictly controlling the generation output. The goal was to force the model to output a single interruption token instantly.

## Methodology
Prompt engineering was employed to strictly ban the model from reasoning. The model was given the system instruction: "Your VERY FIRST token MUST be exactly 'STOP'". 
The model's generation was artificially capped at `max_new_tokens=1` to enforce immediate evaluation and prevent multi-token reasoning chains.

## Results
- **Latency**: The Assessor latency dropped drastically to **~250ms**, well within the sub-500ms target.
- **Accuracy**: The accuracy plummeted to **0/3**. 

## Analysis
The 1-token override failed because the model's instruct-weights violently overrode the explicit system prompt. The model still attempted to output its internal `<|channel>` reasoning token as its first action. Because generation was capped at 1 token, the Assessor exclusively outputted `<|channel>`, never reaching the actual `STOP` evaluation.

### The Pre-Fill Exploit
To circumvent this, a final test was conducted by manually pre-filling the KV cache with the mandatory `<|channel>thought\n<channel|>` block. By tricking the model into believing it had already finished reasoning, its true first generated token became the semantic evaluation. This exploit successfully restored **100% accuracy** at a 1-token latency of **~350-500ms**.

**Conclusion**: While the pre-fill exploit proves that 1-token Semantic VAD is mechanically possible zero-shot, it relies on brittle template manipulation. These findings empirically confirm that for robust, production-grade systems, **Supervised Fine-Tuning (SFT) is strictly required** to natively unlearn the reasoning loops and map semantic barge-ins directly to a single output token.
