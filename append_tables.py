with open("notebooklm/Agentic_Interrupt/Ablation_Matrix.md", "a") as f:
    f.write("\n\n## Experimental Results\n\n")
    
    with open("/home/mallock/.gemini/antigravity-cli/brain/d3c9efef-2526-40c2-b93e-d1c25654a726/all_ablation_tables.md", "r") as f_all:
        tables = f_all.read().replace("# Cross-Model Architecture Ablation Tables\n\n", "")
        f.write(tables)
        
    f.write("\n\n")
    
    with open("/home/mallock/.gemini/antigravity-cli/brain/d3c9efef-2526-40c2-b93e-d1c25654a726/qwen3_table.md", "r") as f_qwen:
        qwen = f_qwen.read()
        f.write(qwen)

