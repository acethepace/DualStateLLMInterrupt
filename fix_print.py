with open("eval_qwen_subset.py", "r") as f:
    lines = f.readlines()
for i, line in enumerate(lines):
    if "Pred: {pred}" in line:
        lines[i] = '    print(f"User: {item[\'user_interruption\']} | Pred: {pred} | Expected: {expected} | Probs: STOP={probs[0]:.2f}, IGNORE={probs[1]:.2f}")\n'
with open("eval_qwen_subset.py", "w") as f:
    f.writelines(lines)
