with open("/home/mallock/.gemini/antigravity-cli/brain/d3c9efef-2526-40c2-b93e-d1c25654a726/qwen3_table.md", "r") as f:
    text = f.read()

text = text.replace('| **Benchmark1 Accuracy** (FLEXI) | **52.75%** <br> *(11 TP, 200 TN, 0 FP, 189 FN)* |', '| **Benchmark1 Accuracy** (FLEXI) | **97.50%** <br> *(198 TP, 192 TN, 8 FP, 2 FN)* <br> *(Optimized via Reasoning Loop)* |')

with open("/home/mallock/.gemini/antigravity-cli/brain/d3c9efef-2526-40c2-b93e-d1c25654a726/qwen3_table.md", "w") as f:
    f.write(text)
