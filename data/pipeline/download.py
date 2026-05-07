import os
from datasets import load_dataset
import json

def main():
    print("Downloading ShareGPT dataset from HuggingFace...")
    # Load dataset
    dataset = load_dataset("anon8231489123/ShareGPT_Vicuna_unfiltered", data_files="ShareGPT_V3_unfiltered_cleaned_split.json")
    
    # Define outputs
    output_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "sharegpt_raw.json")
    
    # Save raw json
    data = dataset['train'].to_list()
    
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
        
    print(f"Saved {len(data)} items to {out_path}")

if __name__ == "__main__":
    main()
