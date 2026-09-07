import re

for filename in ["benchmark_runners/flexi_runner.py", "benchmark_runners/fdb_runner.py"]:
    with open(filename, "r") as f:
        content = f.read()
    
    content = re.sub(r'config\["max_new_tokens"\] = 15', 'config["max_new_tokens"] = 35', content)
    content = re.sub(r'stop_v1 = "STOP"', 'stop_v1 = "[HALT]"', content)
    content = re.sub(r'cont_v1 = "CONTINUE"', 'cont_v1 = "[PROCEED]"', content)
    
    with open(filename, "w") as f:
        f.write(content)
print("Patched variables!")
