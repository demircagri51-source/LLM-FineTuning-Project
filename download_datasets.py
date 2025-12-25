
from datasets import load_dataset
import json
import os

os.makedirs("data", exist_ok=True)

datasets_to_get = {
    "deep": "Naholav/CodeGen-Deep-5K",
    "diverse": "Naholav/CodeGen-Diverse-5K"
}

for name, repo in datasets_to_get.items():
    print(f"Loading {repo} ...")
    ds = load_dataset(repo)  

    
    print("Available splits:", ds.keys())

   
    split = "train" if "train" in ds else list(ds.keys())[0]
    print(f"Using split: {split}")

    out_file = f"data/{name}_solution_only.jsonl"
    with open(out_file, "w", encoding="utf-8") as fout:
        count = 0
        for item in ds[split]:
            
            solution = item.get("solution", None)
            input_field = item.get("input", None)
           
            if solution is None:
                continue
            
            fout.write(json.dumps({"input": input_field, "solution": solution}, ensure_ascii=False) + "\n")
            count += 1
    print(f"Saved {count} examples to {out_file}\n")
