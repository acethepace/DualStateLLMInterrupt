import re

with open("benchmark_runners/flexi_runner.py", "r") as f:
    content = f.read()

content = content.replace('["v1", "v2", "v3", "v2_prefill"]', '["v1", "v2", "v3", "v4", "v2_prefill"]')
content = content.replace('version in ["v2", "v3", "v2_prefill"]', 'version in ["v2", "v3", "v4", "v2_prefill"]')

with open("benchmark_runners/flexi_runner.py", "w") as f:
    f.write(content)
