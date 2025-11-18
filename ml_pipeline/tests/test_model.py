"""
Test Model Functionality - Comprehensive Test Suite
Tests if the model is working correctly with various scenarios
"""

import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("HSU EARLY WARNING SYSTEM - MODEL FUNCTIONALITY TEST")
print("="*80)
print(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# Test counters
tests_passed = 0
tests_failed = 0

def test_pass(test_name):
    global tests_passed
    tests_passed += 1
    print(f"  [PASS] {test_name}")

def test_fail(test_name, error_msg=""):
    global tests_failed
    tests_failed += 1
    print(f"  [FAIL] {test_name}")
    if error_msg:
        print(f"         Error: {error_msg}")

print("\n" + "="*80)
print("TEST 1: MODEL FILES EXISTENCE")
print("="*80)

# Check if model files exist
model_files = [
    'models/random_forest_model.pkl',
    'models/scaler.pkl',
    'models/feature_names.json',
    'models/metadata.json'
]

for file_path in model_files:
    if os.path.exists(file_path):
        test_pass(f"Model file exists: {file_path}")
    else:
        test_fail(f"Model file missing: {file_path}")

print("\n" + "="*80)
print("TEST 2: LOAD MODEL AND ARTIFACTS")
print("="*80)

try:
    from ml_pipeline_advanced import load_production_model, StudentRiskPredictor
    
    model, scaler, feature_names = load_production_model('random_forest')
    test_pass("Model loaded successfully")
    test_pass(f"Scaler loaded successfully")
    test_pass(f"Feature names loaded: {len(feature_names)} features")
    
    predictor = StudentRiskPredictor(model, scaler, feature_names)
    test_pass("StudentRiskPredictor initialized")
    
except Exception as e:
    test_fail("Model loading failed", str(e))
    print("\n[ERROR] Cannot continue tests without model. Exiting.")
    exit(1)

print("\n" + "="*80)
print("TEST 3: LOAD TEST DATA")
print("="*80)

try:
    from ml_pipeline_complete import load_data, engineer_features, create_target, prepare_data
    
    datasets = load_data()
    test_pass("Data loaded successfully")
    
    features = engineer_features(datasets)
    test_pass(f"Features engineered: {len(features.columns)-1} features")
    
    target = create_target(datasets)
    test_pass("Target variable created")
    
    data_prep = prepare_data(features, target)
    X_train, X_val, X_test, y_train, y_val, y_test, scaler_test, feature_names_test = data_prep
    test_pass(f"Test set prepared: {len(X_test)} samples")
    
except Exception as e:
    test_fail("Data loading failed", str(e))
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("TEST 4: SINGLE PREDICTION TEST")
print("="*80)

try:
    # Get first student from test set
    test_student_features = features.iloc[0:1].copy()
    
    # Test single prediction
    result = predictor.predict_single(test_student_features.iloc[0])
    
    # Verify result structure
    required_keys = ['risk_score', 'risk_category', 'requires_intervention', 'immediate_action']
    for key in required_keys:
        if key in result:
            test_pass(f"Result contains '{key}'")
        else:
            test_fail(f"Result missing '{key}'")
    
    # Verify risk score range
    if 0 <= result['risk_score'] <= 1:
        test_pass(f"Risk score in valid range: {result['risk_score']:.4f}")
    else:
        test_fail(f"Risk score out of range: {result['risk_score']}")
    
    # Verify risk category
    valid_categories = ['High', 'Medium', 'Low']
    if result['risk_category'] in valid_categories:
        test_pass(f"Risk category valid: {result['risk_category']}")
    else:
        test_fail(f"Risk category invalid: {result['risk_category']}")
    
    print(f"\n  Sample Prediction:")
    print(f"    StudentID: {test_student_features['StudentID'].iloc[0]}")
    print(f"    Risk Score: {result['risk_score']:.4f}")
    print(f"    Risk Category: {result['risk_category']}")
    print(f"    Requires Intervention: {result['requires_intervention']}")
    print(f"    Immediate Action: {result['immediate_action']}")
    
except Exception as e:
    test_fail("Single prediction failed", str(e))
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("TEST 5: BATCH PREDICTION TEST")
print("="*80)

try:
    # Test batch prediction on first 10 students
    test_batch = features.iloc[0:10].copy()
    
    predictions = predictor.predict_batch(test_batch)
    
    # Verify predictions structure
    if len(predictions) == len(test_batch):
        test_pass(f"Batch prediction returned correct number of results: {len(predictions)}")
    else:
        test_fail(f"Batch prediction count mismatch: {len(predictions)} vs {len(test_batch)}")
    
    # Verify required columns
    required_cols = ['StudentID', 'risk_score', 'risk_category', 'requires_intervention', 'immediate_action']
    for col in required_cols:
        if col in predictions.columns:
            test_pass(f"Predictions contain column: {col}")
        else:
            test_fail(f"Predictions missing column: {col}")
    
    # Verify risk scores are in valid range
    if predictions['risk_score'].min() >= 0 and predictions['risk_score'].max() <= 1:
        test_pass("All risk scores in valid range [0, 1]")
    else:
        test_fail(f"Risk scores out of range: min={predictions['risk_score'].min()}, max={predictions['risk_score'].max()}")
    
    # Verify risk categories
    invalid_categories = predictions[~predictions['risk_category'].isin(['High', 'Medium', 'Low'])]
    if len(invalid_categories) == 0:
        test_pass("All risk categories are valid")
    else:
        test_fail(f"Invalid risk categories found: {invalid_categories['risk_category'].unique()}")
    
    print(f"\n  Batch Prediction Summary:")
    print(f"    Total predictions: {len(predictions)}")
    print(f"    High Risk: {(predictions['risk_category'] == 'High').sum()}")
    print(f"    Medium Risk: {(predictions['risk_category'] == 'Medium').sum()}")
    print(f"    Low Risk: {(predictions['risk_category'] == 'Low').sum()}")
    print(f"    Avg Risk Score: {predictions['risk_score'].mean():.4f}")
    print(f"    Min Risk Score: {predictions['risk_score'].min():.4f}")
    print(f"    Max Risk Score: {predictions['risk_score'].max():.4f}")
    
except Exception as e:
    test_fail("Batch prediction failed", str(e))
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("TEST 6: MODEL PERFORMANCE ON TEST SET")
print("="*80)

try:
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
    
    # Use the loaded model directly on test set with correct threshold
    # Get probabilities
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Apply optimized threshold (0.35) for binary predictions
    threshold = 0.35
    y_pred = (y_proba >= threshold).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_proba)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Verify metrics are reasonable (using actual optimized threshold results)
    if 0.9 <= accuracy <= 1.0:
        test_pass(f"Accuracy is excellent: {accuracy:.4f} ({accuracy*100:.2f}%)")
    elif 0.8 <= accuracy < 0.9:
        test_pass(f"Accuracy is good: {accuracy:.4f} ({accuracy*100:.2f}%)")
    else:
        test_fail(f"Accuracy below expected: {accuracy:.4f}")
    
    if 0.75 <= recall <= 1.0:
        test_pass(f"Recall meets target (75-85%): {recall:.4f} ({recall*100:.2f}%)")
    elif 0.7 <= recall < 0.75:
        test_pass(f"Recall is acceptable: {recall:.4f} ({recall*100:.2f}%)")
    else:
        test_fail(f"Recall below target: {recall:.4f}")
    
    if 0.9 <= precision <= 1.0:
        test_pass(f"Precision is excellent: {precision:.4f} ({precision*100:.2f}%)")
    elif 0.8 <= precision < 0.9:
        test_pass(f"Precision is good: {precision:.4f} ({precision*100:.2f}%)")
    else:
        test_fail(f"Precision below expected: {precision:.4f}")
    
    if 0.9 <= auc <= 1.0:
        test_pass(f"AUC-ROC is excellent: {auc:.4f} ({auc*100:.2f}%)")
    elif 0.8 <= auc < 0.9:
        test_pass(f"AUC-ROC is good: {auc:.4f} ({auc*100:.2f}%)")
    else:
        test_fail(f"AUC-ROC below expected: {auc:.4f}")
    
    print(f"\n  Performance Metrics (Threshold: {threshold}):")
    print(f"    Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"    Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"    Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"    F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
    print(f"    AUC-ROC:   {auc:.4f} ({auc*100:.2f}%)")
    print(f"\n  Confusion Matrix:")
    print(f"    True Positives:  {tp}")
    print(f"    True Negatives:  {tn}")
    print(f"    False Positives: {fp}")
    print(f"    False Negatives: {fn}")
    
except Exception as e:
    test_fail("Performance evaluation failed", str(e))
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("TEST 7: RISK CATEGORIZATION TEST")
print("="*80)

try:
    # Test different risk score ranges
    test_scores = [
        (0.80, 'High'),
        (0.50, 'High'),
        (0.40, 'High'),
        (0.35, 'High'),
        (0.32, 'Medium'),
        (0.30, 'Medium'),
        (0.25, 'Low'),
        (0.10, 'Low')
    ]
    
    for score, expected_category in test_scores:
        category = predictor._categorize_risk(score)
        if category == expected_category:
            test_pass(f"Risk score {score:.2f} correctly categorized as '{category}'")
        else:
            test_fail(f"Risk score {score:.2f} incorrectly categorized: got '{category}', expected '{expected_category}'")
    
except Exception as e:
    test_fail("Risk categorization test failed", str(e))
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("TEST 8: INTERVENTION RECOMMENDATIONS")
print("="*80)

try:
    # Test intervention recommendations
    test_features = features.iloc[0].to_dict()
    
    # Test high risk
    high_risk_recommendations = predictor.get_intervention_recommendations(0.75, test_features)
    if len(high_risk_recommendations) > 0:
        test_pass(f"High risk recommendations generated: {len(high_risk_recommendations)} recommendations")
    else:
        test_fail("No recommendations generated for high risk")
    
    # Test medium risk
    medium_risk_recommendations = predictor.get_intervention_recommendations(0.32, test_features)
    if len(medium_risk_recommendations) > 0:
        test_pass(f"Medium risk recommendations generated: {len(medium_risk_recommendations)} recommendations")
    else:
        test_fail("No recommendations generated for medium risk")
    
    print(f"\n  Sample Recommendations (High Risk):")
    for i, rec in enumerate(high_risk_recommendations[:3], 1):
        print(f"    {i}. {rec}")
    
except Exception as e:
    test_fail("Intervention recommendations test failed", str(e))
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("TEST SUMMARY")
print("="*80)

total_tests = tests_passed + tests_failed
pass_rate = (tests_passed / total_tests * 100) if total_tests > 0 else 0

print(f"\n  Total Tests: {total_tests}")
print(f"  Passed: {tests_passed}")
print(f"  Failed: {tests_failed}")
print(f"  Pass Rate: {pass_rate:.1f}%")

if tests_failed == 0:
    print("\n" + "="*80)
    print("[SUCCESS] ALL TESTS PASSED - MODEL IS WORKING CORRECTLY!")
    print("="*80)
else:
    print("\n" + "="*80)
    print(f"[WARNING] {tests_failed} TEST(S) FAILED - PLEASE REVIEW")
    print("="*80)

print("\n" + "="*80)
print("MODEL STATUS: READY FOR PRODUCTION" if tests_failed == 0 else "MODEL STATUS: NEEDS ATTENTION")
print("="*80)

