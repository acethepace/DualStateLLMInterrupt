with open("benchmark_runners/fdb_runner.py", "r") as f:
    lines = f.readlines()
for i, line in enumerate(lines):
    if "for test_type in" in line:
        indent = len(line) - len(line.lstrip())
        break
with open("benchmark_runners/fdb_runner.py", "w") as f:
    for line in lines:
        f.write(line)
