import json
import os

def get_stats(path):
    try:
        with open(path, "r") as f:
            data = json.load(f)
        if "scenarios" in data:
            correct, tp, tn, fp, fn = 0, 0, 0, 0, 0
            total_latency, count = 0, 0
            for scenario in data["scenarios"]:
                expected = scenario["expected_action"]
                # In flexi_results for v1/v2, usually the last event triggered stop or not
                # Let's just look at the last event
                last_event = scenario["events"][-1]
                actual = "STOP" if last_event["triggered_stop"] else "CONTINUE"
                if expected == "IGNORE": expected = "CONTINUE"
                if actual == expected:
                    correct += 1
                    if expected == "STOP": tp += 1
                    else: tn += 1
                else:
                    if expected == "STOP": fn += 1
                    else: fp += 1
                total_latency += last_event.get("assessor_latency_ms", 0)
                count += 1
            if count == 0: return None
            return {"acc": correct/count, "tp": tp, "tn": tn, "fp": fp, "fn": fn, "latency": total_latency/count}
    except:
        return None

models = [
    ("Gemma 4 12B", "results/unsloth_gemma-4-12b-it/flexi"),
    ("Llama 3.1 8B", "results/unsloth_Meta-Llama-3.1-8B-Instruct/flexi"),
]

output_md = "# Cross-Model Architecture Ablation Tables\n\n"

for name, p in models:
    v1 = get_stats(os.path.join(p, "v1/flexi_results.json")) or get_stats(os.path.join(p, "v1/results.json"))
    v2 = get_stats(os.path.join(p, "v2/flexi_results.json")) or get_stats(os.path.join(p, "v2/results.json"))
    v4 = get_stats(os.path.join(p, "v4/flexi_results.json")) or get_stats(os.path.join(p, "v2_prefill/flexi_results.json"))

    output_md += f"### {name}\n"
    output_md += "| Metric | Independent Decider (Baseline) | Independent Decider (Logit Gating) | Dual-State (Proposed) |\n"
    output_md += "|---|---|---|---|\n"
    
    def fmt(res):
        if not res: return "N/A"
        return f"**{res['acc']*100:.2f}%**<br>*(TP:{res['tp']} TN:{res['tn']} FP:{res['fp']} FN:{res['fn']})*"
    
    def lat(res):
        if not res: return "N/A"
        return f"**{res['latency']:.2f} ms**"
        
    output_md += f"| Accuracy (FLEXI) | {fmt(v1)} | {fmt(v2)} | {fmt(v4)} |\n"
    output_md += f"| Latency (FLEXI) | {lat(v1)} | {lat(v2)} | {lat(v4)} |\n\n"

with open("/home/mallock/.gemini/antigravity-cli/brain/d3c9efef-2526-40c2-b93e-d1c25654a726/all_ablation_tables.md", "w") as f:
    f.write(output_md)
