# Empirical Validation: V1 (Zero-Shot)

## Overview
The V1 evaluation of the Dual-State KV-Forking architecture tested whether a generalized instruct model (Gemma 4 12B) could perform Semantic Voice Activity Detection (VAD) using zero-shot prompting, without the need for Supervised Fine-Tuning (SFT).

## Methodology
The model was prompted to output `<|Start-Listening|>` upon detecting an emergency barge-in during the FLEXI benchmark. 
To achieve low latency, the architecture copied the `past_key_values` from the Base dialogue model directly into an Assessor branch. This entirely eliminated the compute-heavy prefill latency typically associated with processing long conversational histories.

## Results
- **Accuracy**: The model achieved **100% accuracy (3/3)** on the FLEXI benchmark. It correctly identified and halted on barge-ins while successfully ignoring benign backchannels.
- **Latency Limitation**: The Assessor latency spiked significantly, ranging from **1700ms to 4000ms**.

## Analysis
The latency spike is attributed to the internal instruction-tuning of generalized models. When placed in the Assessor state, the model mandatorily engaged its internal reasoning channels (e.g., generating `<|channel>thought\n...<channel|>`) prior to emitting the requested control token. Because the model required generating up to 15 tokens per chunk evaluation to complete its thought process, it fundamentally violated the sub-500ms real-time conversational threshold. 

**Conclusion**: While zero-shot Semantic VAD is highly accurate using KV-Forking, the inherent reasoning latency of generalized instruct models makes it unsuitable for production full-duplex systems without further optimization.
