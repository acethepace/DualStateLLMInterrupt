import re

prompt = """You are a VAD. Output JSON: {{"t": "<1-word reason>", "a": "<[H] or [P]>"}}. If intentional interruption, use [H]. If benign backchannel, use [P]."""

for filename in ["benchmark_runners/flexi_runner.py", "benchmark_runners/fdb_runner.py"]:
    with open(filename, "r") as f:
        lines = f.readlines()
        
    for i in range(len(lines)):
        if lines[i].strip().startswith("base_v1 ="):
            lines[i] = f'    base_v1 = """{prompt}"""\n'
            
    with open(filename, "w") as f:
        f.writelines(lines)
        
    with open(filename, "r") as f:
        content = f.read()
    
    content = re.sub(r'stop_v1, cont_v1, stop_v2 = "H", "P", "HALT"', 'stop_v1, cont_v1, stop_v2 = "[H]", "[P]", "HALT"', content)
    
    with open(filename, "w") as f:
        f.write(content)
print("Fixed H and P!")
