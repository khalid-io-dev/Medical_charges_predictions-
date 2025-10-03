import pandas as pd
import joblib

def prepare_single_input(raw_input, feature_names):
    """Prepare a single input for prediction."""
    has_children = 1 if raw_input.get("children", 0) > 0 else 0
    smoker_flag = 1 if str(raw_input.get("smoker", "no")).lower() == "yes" else 0
    age_smoker = raw_input.get("age", 0) * smoker_flag
    bmi_smoker = raw_input.get("bmi", 0.0) * smoker_flag
    smoker_region_value = f"{raw_input.get('smoker', 'no')}_{raw_input.get('region', '')}"
    
    row = {col: 0 for col in feature_names}
    
    if "age" in row: row["age"] = raw_input.get("age", 0)
    if "bmi" in row: row["bmi"] = raw_input.get("bmi", 0.0)
    if "children" in row: row["children"] = raw_input.get("children", 0)
    if "has_children" in row: row["has_children"] = has_children
    if "age_smoker" in row: row["age_smoker"] = age_smoker
    if "bmi_smoker" in row: row["bmi_smoker"] = bmi_smoker
    if "sex_male" in row:
        row["sex_male"] = 1 if str(raw_input.get("sex", "female")).lower() == "male" else 0
    if "smoker_yes" in row:
        row["smoker_yes"] = smoker_flag
    region_col = f"region_{raw_input.get('region', '')}"
    if region_col in row:
        row[region_col] = 1
    smoker_region_col = f"smoker_region_{smoker_region_value}"
    if smoker_region_col in row:
        row[smoker_region_col] = 1
    
    return pd.DataFrame([row], columns=feature_names)

def predict_charges(raw_input, model_path="models/final_model.joblib", feature_names_path="models/feature_names.joblib"):
    """Predict charges for a single input."""
    model = joblib.load(model_path)
    feature_names = joblib.load(feature_names_path)
    sample = prepare_single_input(raw_input, feature_names)
    pred = model.predict(sample)
    return float(pred[0])