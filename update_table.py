with open("/home/mallock/.gemini/antigravity-cli/brain/d3c9efef-2526-40c2-b93e-d1c25654a726/qwen3_table.md", "r") as f:
    text = f.read()

text = text.replace("| **Benchmark2 Accuracy** (FDB)   | *Running...* | *Running...* | *Running...* |", "| **Benchmark2 Accuracy** (FDB)   | **50.25%** <br> *(1 TP, 200 TN, 0 FP, 199 FN)* | **98.50%** <br> *(194 TP, 200 TN, 0 FP, 6 FN)* | **98.50%** <br> *(194 TP, 200 TN, 0 FP, 6 FN)* |")

with open("/home/mallock/.gemini/antigravity-cli/brain/d3c9efef-2526-40c2-b93e-d1c25654a726/qwen3_table.md", "w") as f:
    f.write(text)
