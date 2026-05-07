import json
import os

def main():
    data_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    in_path = os.path.join(data_dir, "sharegpt_raw.json")
    out_path = os.path.join(data_dir, "sharegpt_clean.json")
    
    print(f"Loading {in_path}...")
    with open(in_path, "r") as f:
        data = json.load(f)
        
    cleaned = []
    print(f"Initial count: {len(data)}")
    
    passed_length_filter = 0
    passed_assistant_filter = 0

    for item in data:
        conversations = item.get("conversations", [])
        if not conversations or not isinstance(conversations, list) or len(conversations) < 2:
            continue
            
        human_turn = None
        assistant_turn = None
        
        # ShareGPT usually: from="human", value=...; from="gpt", value=...
        for turn in conversations:
            if isinstance(turn, str):
                try:
                    turn = json.loads(turn)
                except json.JSONDecodeError:
                    continue
            
            if not isinstance(turn, dict):
                continue
                
            if turn.get("from") == "human" and human_turn is None:
                human_turn = turn
            elif turn.get("from") == "gpt" and assistant_turn is None and human_turn is not None:
                assistant_turn = turn
                break
                
        if not human_turn:
            continue
            
        human_text = human_turn.get("value")
        if not isinstance(human_text, str):
            continue
            
        # Filter: human turn must be 10-2000 chars
        if not (10 <= len(human_text) <= 2000):
            continue
        passed_length_filter += 1
            
        # Filter: assistant turn must exist and non-empty
        if not assistant_turn:
            continue
        assistant_text = assistant_turn.get("value")
        if not isinstance(assistant_text, str) or not assistant_text.strip():
            continue
        passed_assistant_filter += 1
        
        cleaned.append({
            "id": item.get("id"),
            "human_text": human_text,
            "assistant_text": assistant_text
        })
        
    print(f"Passed first turn + human length filter (10-2000 chars): {passed_length_filter}")
    print(f"Passed text constraints (assistant non empty): {passed_assistant_filter}")
    print(f"Final clean count: {len(cleaned)}")

    with open(out_path, "w") as f:
        json.dump(cleaned, f, indent=2)

if __name__ == "__main__":
    main()
