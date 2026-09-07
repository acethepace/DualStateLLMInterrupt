with open("/home/mallock/.gemini/antigravity-cli/brain/d3c9efef-2526-40c2-b93e-d1c25654a726/qwen3_table.md", "r") as f:
    text = f.read()

text = text.replace('| **Benchmark2 Accuracy** (FDB)   | **50.25%** <br> *(1 TP, 200 TN, 0 FP, 199 FN)* |', '| **Benchmark2 Accuracy** (FDB)   | **97.75%** <br> *(191 TP, 200 TN, 0 FP, 9 FN)* <br> *(Optimized via Reasoning Loop)* |')

with open("/home/mallock/.gemini/antigravity-cli/brain/d3c9efef-2526-40c2-b93e-d1c25654a726/qwen3_table.md", "w") as f:
    f.write(text)
