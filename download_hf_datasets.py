from datasets import load_dataset
import os

print("Downloading Duplex-UltraChat...")
try:
    ds = load_dataset("xinrongzhang2022/Duplex-UltraChat")
    ds.save_to_disk("datasets/Duplex-UltraChat")
    print("Saved Duplex-UltraChat.")
except Exception as e:
    print("Error:", e)

# Also try to grab the others if they have an exact HF name
print("Downloading EffiBench-X...")
try:
    ds = load_dataset("EffiBench/effibench-x")
    ds.save_to_disk("datasets/EffiBench-X-HF")
    print("Saved EffiBench-X-HF.")
except Exception as e:
    print("Error:", e)
