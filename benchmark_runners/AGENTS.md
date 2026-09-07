# Benchmark Runners

This directory contains dedicated runner scripts for each major full-duplex benchmark (e.g., FLEXI, Full-Duplex-Bench, Duplex-UltraChat).

## Architecture
- **Runners (Wrappers)**: Each Python script in this folder acts as a specialized parser and executor for a specific dataset. Because each dataset formats overlapping speech, timestamps, and interruptions differently, the runner is responsible for translating the dataset's native format into a standard stream of tokens/chunks.
- **Core Logic**: The runners pass the formatted chunks into the central `dual_state_interruptor.py` core logic, which handles the KV-cache forking, model inference, and semantic evaluation (V1/V2).

## Strict Engineering Requirements for All Runners
To ensure stability, transparency, and data integrity during long-running background evaluations (which can take hours), EVERY runner in this directory **MUST** implement the following four features:

### 1. Checkpointing & Resumption
- **Iterative Saving**: Test case results must be appended/written to the JSON file incrementally after *every single iteration* (or periodically saved). Do not store massive arrays in memory and only save at the very end.
- **Resumption**: Before starting, the runner must read the existing `results.json`. Any test case `id` already completed and saved in the file should be skipped. This ensures that if an OOM crash occurs on scenario 300, restarting the runner resumes at 301.

### 2. Live Logging
- **Progress Visibility**: Runners must include `print()` statements for every scenario processed (e.g., `print(f"Processed {idx}/{total}: {scenario_id} - {result} (Latency: {ms}ms)")`). This prevents the terminal/background task from being silent for hours.

### 3. Output Format Standardization
- **Pathing**: The output format MUST match the global results structure:
  `../results/[model_name]/[dataset_name]/[version_configuration]/results.json`
- Example: `results/unsloth_gemma-4-12b-it/flexi/v2_prefill/results.json`

### 4. GPU Mutex / Concurrency Lock
- **GPU Protection**: Only ONE test case utilizing the local GPU should be running at any given time. If multiple runners or bash scripts are executed in parallel (e.g. FDB vs FLEXI), they must implement a file lock (e.g. `gpu.lock`) or gracefully abort to prevent catastrophic Out-Of-Memory (OOM) crashes in the background.
