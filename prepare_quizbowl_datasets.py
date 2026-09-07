import json
import os
from datasets import load_dataset

def clean_and_save():
    os.makedirs('datasets/quizbowl', exist_ok=True)
    
    final_dataset = []
    
    # 1. AdvQA
    print("Loading AdvQA...")
    try:
        advqa = load_dataset("qanta-challenge/AdvQA", split="train")
        for i, item in enumerate(advqa):
            if i >= 100:  # Take 100 for subset
                break
            final_dataset.append({
                "id": f"advqa_{i}",
                "question": item['question'],
                "answer": item['answer']
            })
    except Exception as e:
        print(f"Failed to load AdvQA: {e}")

    # 2. Protobowl
    print("Loading Protobowl...")
    try:
        protobowl = load_dataset("mgor/protobowl-11-13", "progressive-clues", split="train")
        for i, item in enumerate(protobowl):
            if i >= 100: # Take 100 for subset
                break
            final_dataset.append({
                "id": f"protobowl_{i}",
                "question": item['question'],
                "answer": item['answer']
            })
    except Exception as e:
        print(f"Failed to load Protobowl: {e}")

    # Save small subset
    with open('datasets/quizbowl/subset.json', 'w') as f:
        json.dump(final_dataset[:20], f, indent=2)
        
    # Save larger set
    with open('datasets/quizbowl/full.json', 'w') as f:
        json.dump(final_dataset, f, indent=2)
        
    print(f"Saved {len(final_dataset[:20])} to subset.json and {len(final_dataset)} to full.json")

if __name__ == "__main__":
    clean_and_save()
