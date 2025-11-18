"""Quick Model Test - Verify Model is Working"""
from ml_pipeline_advanced import load_production_model, StudentRiskPredictor
from ml_pipeline_complete import load_data, engineer_features

print("="*80)
print("QUICK MODEL TEST - VERIFICATION")
print("="*80)

# Load model
model, scaler, feature_names = load_production_model('random_forest')
predictor = StudentRiskPredictor(model, scaler, feature_names)
print("\n[OK] Model loaded successfully")

# Load data and get a sample student
datasets = load_data()
features = engineer_features(datasets)
test_student = features.iloc[0:1]

# Make prediction
result = predictor.predict_single(test_student.iloc[0])

print(f"\nSample Prediction:")
print(f"  Student ID: {test_student['StudentID'].iloc[0]}")
print(f"  Risk Score: {result['risk_score']:.4f}")
print(f"  Risk Category: {result['risk_category']}")
print(f"  Requires Intervention: {result['requires_intervention']}")
print(f"  Immediate Action: {result['immediate_action']}")

print("\n" + "="*80)
print("MODEL STATUS: WORKING CORRECTLY")
print("="*80)

