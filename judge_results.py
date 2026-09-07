import json
import sys

def judge_results(filename):
    with open(filename, 'r') as f:
        results = json.load(f)
        
    total = len(results)
    interrupted_count = sum(1 for r in results if r['interrupted'])
    accurate_interrupts = 0
    total_time_to_halt = 0
    total_redundant_tokens = 0
    total_consensus_time = 0
    
    for r in results:
        total_consensus_time += r.get('total_time_to_consensus', 0)
        if r['interrupted']:
            total_time_to_halt += r.get('time_to_halt', 0)
            total_redundant_tokens += r.get('redundant_tokens_processed', 0)
            
            # Simple substring matching judge
            guess = r.get('guess', '').lower().strip()
            answer = str(r.get('actual_answer', '')).lower().strip()
            
            if guess in answer or answer in guess or len(set(guess.split()) & set(answer.split())) > 0:
                accurate_interrupts += 1
                
    passive_rate = (total - interrupted_count) / total * 100 if total > 0 else 0
    acc_rate = (accurate_interrupts / interrupted_count * 100) if interrupted_count > 0 else 0
    avg_halt = (total_time_to_halt / interrupted_count) if interrupted_count > 0 else 0
    avg_redundant = (total_redundant_tokens / interrupted_count) if interrupted_count > 0 else 0
    avg_consensus = total_consensus_time / total if total > 0 else 0
    
    print(f"Results for {filename}:")
    print(f"  Total Scenarios: {total}")
    print(f"  Passive Listening Rate (Failed to interrupt): {passive_rate:.1f}%")
    print(f"  Interruption Accuracy: {acc_rate:.1f}%")
    print(f"  Avg Time-to-Halt: {avg_halt:.4f} sec")
    print(f"  Avg Redundant Tokens (Compute Waste): {avg_redundant:.1f} tokens")
    print(f"  Avg Time to Consensus: {avg_consensus:.4f} sec\n")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            try:
                judge_results(arg)
            except Exception as e:
                print(f"Could not judge {arg}: {e}")
    else:
        print("Please provide filenames.")
