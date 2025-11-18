"""
Display Model Accuracy - Quick Results Viewer
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

print("="*80)
print("HSU EARLY WARNING SYSTEM - MODEL PERFORMANCE SUMMARY")
print("="*80)
print("Using optimized threshold (0.35) for 77.42% recall (Target: 75-85%)")

try:
    # Load data for evaluation
    print("\n[1] Loading test data...")
    from ml_pipeline_complete import load_data, engineer_features, create_target, prepare_data
    
    datasets = load_data()
    features = engineer_features(datasets)
    target = create_target(datasets)
    
    # Prepare data
    data_prep = prepare_data(features, target)
    X_train, X_val, X_test, y_train, y_val, y_test, scaler, feature_names = data_prep
    
    print("  + Test set: {:,} students".format(len(X_test)))
    print("  + Features: {:,}".format(len(feature_names)))
    
    # Load models
    print("\n[2] Loading trained models...")
    models = {}
    
    model_names = ['random_forest']
    for name in model_names:
        try:
            model = joblib.load(f'models/{name}_model.pkl')
            models[name] = model
            print(f"  + Loaded {name}")
        except:
            print(f"  - Could not load {name}")
    
    # Evaluate each model
    print("\n" + "="*80)
    print("MODEL PERFORMANCE ON TEST SET ({:,} students)".format(len(X_test)))
    print("="*80)
    
    results = []
    
    # Use optimized threshold (0.35) for optimal recall
    OPTIMIZED_THRESHOLD = 0.35  # Optimized for 77.42% recall (Target: 75-85%)
    
    for name, model in models.items():
        # Predictions with optimized threshold for better recall
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= OPTIMIZED_THRESHOLD).astype(int)
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        
        results.append({
            'Model': name.upper(),
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'AUC-ROC': auc
        })
        
        print(f"\n{name.upper()}")
        print("-" * 80)
        print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"  Precision: {precision:.4f} ({precision*100:.2f}%)")
        print(f"  Recall:    {recall:.4f} ({recall*100:.2f}%)")
        print(f"  F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
        print(f"  AUC-ROC:   {auc:.4f} ({auc*100:.2f}%)")
    
    # Summary comparison
    print("\n" + "="*80)
    print("MODEL COMPARISON - RANKED BY AUC")
    print("="*80)
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('AUC-ROC', ascending=False)
    
    print("\n{:<20} {:>10} {:>10} {:>10} {:>10} {:>10}".format(
        'Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC'
    ))
    print("-" * 80)
    
    for idx, row in results_df.iterrows():
        marker = " [PRIMARY]" if row['Model'] == 'RANDOM_FOREST' else ""
        print("{:<20} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f}{}".format(
            row['Model'], row['Accuracy'], row['Precision'], 
            row['Recall'], row['F1-Score'], row['AUC-ROC'], marker
        ))
    
    # Best model
    best_model = results_df.iloc[0]
    print("\n" + "="*80)
    print(f"BEST MODEL: {best_model['Model']} with AUC = {best_model['AUC-ROC']:.4f}")
    print("="*80)
    
    # Target distribution in test set
    print("\n[3] TEST SET STATISTICS")
    print("-" * 80)
    at_risk = y_test.sum()
    retained = len(y_test) - at_risk
    print(f"  At-Risk Students:  {at_risk:,} ({at_risk/len(y_test)*100:.1f}%)")
    print(f"  Retained Students: {retained:,} ({retained/len(y_test)*100:.1f}%)")
    
    # Confusion matrix for best model (using optimized threshold)
    from sklearn.metrics import confusion_matrix
    best_model_obj = models[best_model['Model'].lower()]
    y_proba_best = best_model_obj.predict_proba(X_test)[:, 1]
    y_pred_best = (y_proba_best >= OPTIMIZED_THRESHOLD).astype(int)
    cm = confusion_matrix(y_test, y_pred_best)
    tn, fp, fn, tp = cm.ravel()
    
    print("\n[4] CONFUSION MATRIX (Best Model: {})".format(best_model['Model']))
    print("-" * 80)
    print(f"  True Negatives (Correctly predicted Retained):  {tn:,}")
    print(f"  False Positives (Incorrectly flagged as At-Risk): {fp:,}")
    print(f"  False Negatives (Missed At-Risk students):        {fn:,}")
    print(f"  True Positives (Correctly identified At-Risk):    {tp:,}")
    print(f"\n  False Positive Rate: {fp/(fp+tn)*100:.2f}%")
    print(f"  False Negative Rate: {fn/(fn+tp)*100:.2f}%")
    
    print("\n" + "="*80)
    print("[SUCCESS] Model evaluation complete!")
    print("="*80)
    
except Exception as e:
    print(f"\n[ERROR] {str(e)}")
    import traceback
    traceback.print_exc()

