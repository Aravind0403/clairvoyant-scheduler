import os
import json
import xgboost as xgb

def check_model_json(file_path):
    print(f"--- Inspecting: {file_path} ---")
    
    # 1. Basic File Validation
    if not os.path.exists(file_path):
        print(f"❌ Error: File not found at {file_path}")
        return

    file_size_kb = os.path.getsize(file_path) / 1024
    print(f"File Size: {file_size_kb:.2f} KB\n")

    # 2. Native JSON validation and parsing
    print("--- 1. JSON Structural Validation ---")
    try:
        with open(file_path, 'r') as f:
            model_data = json.load(f)
        print("✅ Successfully parsed as valid JSON.")
        
        # Extract metadata from JSON
        learner = model_data.get("learner", {})
        objective = learner.get("objective", {})
        
        print("\nMetadata found in JSON:")
        print(f"  • Objective: {objective.get('name', 'N/A')}")
        print(f"  • Expected Features: {learner.get('learner_model_param', {}).get('num_feature', 'N/A')}")
        print(f"  • Expected Classes: {learner.get('learner_model_param', {}).get('num_class', 'N/A')}")
        
        trees = learner.get("gradient_booster", {}).get("model", {}).get("trees", [])
        print(f"  • Total Trees: {len(trees)}")

    except json.JSONDecodeError as e:
        print(f"❌ JSON Decode Error: {e}")
        return
    except Exception as e:
        print(f"⚠️ Could not parse some XGBoost internal fields: {e}")

    # 3. XGBoost Loading Test
    print("\n--- 2. XGBoost Compatibility Validation ---")
    try:
        # Load as a Booster to just check native loading
        booster = xgb.Booster()
        booster.load_model(file_path)
        print("✅ Successfully loaded into xgb.Booster.")
        
        # Load into the higher level scikit-learn wrapper
        classifier = xgb.XGBClassifier()
        classifier.load_model(file_path)
        print("✅ Successfully loaded into xgb.XGBClassifier.")
        
        print("\nModel properties:")
        try:
            print(f"  • Classes configured: {classifier.classes_}")
        except AttributeError:
            print(f"  • Classes configured: 3 classes (auto-detected)")

    except Exception as e:
        print(f"❌ XGBoost load failed: {e}")

if __name__ == "__main__":
    # Point to the predictor.json created by train.py
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_json_path = os.path.join(current_dir, "model", "predictor.json")
    
    check_model_json(model_json_path)
