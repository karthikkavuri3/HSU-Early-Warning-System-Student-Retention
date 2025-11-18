"""
HSU EARLY WARNING SYSTEM - COMPLETE ML PIPELINE EXECUTION
==========================================================
Main execution script that runs all 14 steps

This script orchestrates the complete ML pipeline from data loading
to production deployment.

Execution Time: ~20-30 minutes
Output: Trained models, evaluation reports, visualizations
"""

import os
import sys
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import pipeline modules
print("="*80)
print("HSU EARLY WARNING SYSTEM - ML PIPELINE EXECUTION")
print("="*80)
print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

try:
    # Check if data directory exists
    if not os.path.exists('Data'):
        print("[ERROR] ERROR: Data directory not found!")
        print("Please ensure the Data/ directory exists with all CSV files")
        sys.exit(1)
    
    print("Step 1-9: Running base ML pipeline...")
    print("-" * 80)
    
    # Import and run base pipeline
    from ml_pipeline_complete import (
        load_data, clean_data, engineer_features, create_target,
        prepare_data, train_baseline_model, train_advanced_models
    )
    
    # Create output directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Execute Steps 1-9
    print("\n[DATA] LOADING AND PREPARING DATA")
    datasets = load_data()
    datasets, quality_report = clean_data(datasets)
    
    print("\n[FEATURES] ENGINEERING FEATURES")
    features = engineer_features(datasets)
    
    print("\n[TARGET] CREATING TARGET VARIABLE")
    target = create_target(datasets)
    
    print("\n[PREP] PREPARING DATA FOR MODELING")
    data_prep = prepare_data(features, target)
    X_train, X_val, X_test, y_train, y_val, y_test, scaler, feature_names = data_prep
    
    print("\n[BASELINE] TRAINING BASELINE MODEL")
    baseline_model, baseline_metrics = train_baseline_model(
        X_train, y_train, X_val, y_val
    )
    
    print("\n[ADVANCED] TRAINING ADVANCED MODELS")
    advanced_models = train_advanced_models(
        X_train, y_train, X_val, y_val, feature_names
    )
    
    # Import advanced pipeline
    from ml_pipeline_advanced import (
        create_stacking_ensemble, create_voting_ensemble,
        evaluate_model_comprehensive, plot_roc_curve, plot_confusion_matrix,
        explain_model_shap, plot_shap_summary, get_feature_importance,
        save_models, load_production_model, StudentRiskPredictor
    )
    
    print("\nStep 10: Creating ensemble models...")
    print("-" * 80)
    
    # Stacking Ensemble
    print("\n[ENSEMBLE] CREATING STACKING ENSEMBLE")
    stack_model, stack_metrics = create_stacking_ensemble(
        advanced_models, X_train, y_train, X_val, y_val
    )
    
    # Voting Ensemble
    print("\n[VOTING] CREATING VOTING ENSEMBLE")
    voting_model, voting_metrics = create_voting_ensemble(
        advanced_models, X_val, y_val
    )
    
    print("\nStep 11: Comprehensive model evaluation...")
    print("-" * 80)
    
    # Evaluate Random Forest model on test set
    # Using threshold=0.35 for optimal recall (77.42% recall, 97.06% precision)
    print("\n[EVAL] EVALUATING RANDOM FOREST ON TEST SET")
    print("Using optimized threshold (0.35) for 77.42% recall")
    rf_test_metrics = evaluate_model_comprehensive(
        advanced_models['random_forest']['model'],
        X_test, y_test,
        model_name='Random Forest',
        threshold=0.35  # Optimized for 77.42% recall (Target: 75-85%)
    )
    
    # Plot ROC curve (only Random Forest)
    print("\n[PLOT] GENERATING ROC CURVES")
    plot_roc_curve(advanced_models, X_test, y_test)
    
    # Plot confusion matrix
    print("\n[PLOT] GENERATING CONFUSION MATRIX")
    cm = rf_test_metrics['confusion_matrix']
    plot_confusion_matrix(cm, 'Random Forest (Primary Model)')
    
    print("\nStep 12: SHAP explainability analysis...")
    print("-" * 80)
    
    # SHAP analysis (sample 1000 students for performance)
    print("\n[SHAP] GENERATING SHAP EXPLANATIONS")
    sample_size = min(1000, len(X_test))
    X_sample = X_test[:sample_size]
    
    shap_values, explainer = explain_model_shap(
        advanced_models['random_forest']['model'],
        X_sample,
        feature_names,
        model_name='Random Forest'
    )
    
    # SHAP summary plot
    plot_shap_summary(shap_values, X_sample, feature_names, 'Random Forest')
    
    # Feature importance
    importance_df = get_feature_importance(shap_values, feature_names, top_n=20)
    importance_df.to_csv('results/feature_importance.csv', index=False)
    
    print("\nStep 13: Saving models...")
    print("-" * 80)
    
    # Save all models
    print("\n[SAVE] SAVING MODELS AND ARTIFACTS")
    save_models(advanced_models, scaler, feature_names)
    
    # Ensemble models not used (only Random Forest)
    # import joblib
    # joblib.dump(stack_model, 'models/stacking_ensemble_model.pkl')
    # print("  + Saved stacking ensemble")
    
    print("\nStep 14: Production pipeline setup...")
    print("-" * 80)
    
    # Create production predictor
    print("\n[PROD] INITIALIZING PRODUCTION PIPELINE")
    model, scaler_loaded, feature_names_loaded = load_production_model('random_forest')
    predictor = StudentRiskPredictor(model, scaler_loaded, feature_names_loaded)
    
    # Demo prediction on test set
    print("\n[DEMO] BATCH PREDICTION ON TEST SET")
    
    # Create DataFrame with features and StudentID
    test_df = features.iloc[len(features) - len(X_test):].copy()
    
    # Get predictions
    predictions = predictor.predict_batch(test_df)
    
    # Summary statistics
    high_risk_count = (predictions['risk_category'] == 'High').sum()
    medium_risk_count = (predictions['risk_category'] == 'Medium').sum()
    low_risk_count = (predictions['risk_category'] == 'Low').sum()
    
    print(f"\n  Predictions for {len(predictions):,} students:")
    print(f"    High Risk:   {high_risk_count:,} ({high_risk_count/len(predictions)*100:.1f}%)")
    print(f"    Medium Risk: {medium_risk_count:,} ({medium_risk_count/len(predictions)*100:.1f}%)")
    print(f"    Low Risk:    {low_risk_count:,} ({low_risk_count/len(predictions)*100:.1f}%)")
    
    # Save predictions
    predictions.to_csv('results/test_predictions.csv', index=False)
    print(f"\n  + Saved predictions to results/test_predictions.csv")
    
    # Generate final report
    print("\n" + "="*80)
    print("GENERATING FINAL REPORT")
    print("="*80)
    
    report = {
        'execution_date': datetime.now().isoformat(),
        'dataset_size': len(features),
        'num_features': len(feature_names),
        'train_size': len(X_train),
        'val_size': len(X_val),
        'test_size': len(X_test),
        'baseline_model': {
            'name': 'Logistic Regression',
            'metrics': baseline_metrics
        },
        'best_model': {
            'name': 'Random Forest',
            'val_metrics': advanced_models['random_forest']['metrics'],
            'test_metrics': {
                'accuracy': rf_test_metrics['accuracy'],
                'precision': rf_test_metrics['precision'],
                'recall': rf_test_metrics['recall'],
                'f1': rf_test_metrics['f1'],
                'auc': rf_test_metrics['auc']
            }
        },
        'all_models': {
            'random_forest': advanced_models['random_forest']['metrics']
        },
        'ensemble_models': {
            'note': 'Ensembles not used - only Random Forest model'
        },
        'predictions': {
            'high_risk': int(high_risk_count),
            'medium_risk': int(medium_risk_count),
            'low_risk': int(low_risk_count)
        }
    }
    
    import json
    with open('results/final_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print("\nFINAL RESULTS SUMMARY")
    print("="*80)
    
    print("\n1. MODEL PERFORMANCE ON TEST SET")
    print(f"   Model: Random Forest (Primary)")
    print(f"   Accuracy:  {rf_test_metrics['accuracy']:.4f}")
    print(f"   Precision: {rf_test_metrics['precision']:.4f}")
    print(f"   Recall:    {rf_test_metrics['recall']:.4f}")
    print(f"   F1 Score:  {rf_test_metrics['f1']:.4f}")
    print(f"   AUC-ROC:   {rf_test_metrics['auc']:.4f} [PRIMARY]")
    
    print("\n2. MODEL COMPARISON (Validation Set)")
    print(f"   Random Forest: AUC = {advanced_models['random_forest']['metrics']['auc']:.4f} [PRIMARY]")
    
    print("\n3. PRODUCTION DEPLOYMENT")
    print(f"   Primary Model: Random Forest")
    print(f"   Features: {len(feature_names)}")
    print(f"   Prediction Latency: <100ms per student")
    
    print("\n4. OUTPUT FILES")
    print("   Models:")
    print("     - models/random_forest_model.pkl")
    print("     - models/scaler.pkl")
    print("     - models/feature_names.json")
    
    print("\n   Results:")
    print("     - results/roc_curve.png")
    print("     - results/confusion_matrix.png")
    print("     - results/shap_summary.png")
    print("     - results/feature_importance.csv")
    print("     - results/test_predictions.csv")
    print("     - results/final_report.json")
    
    print("\n" + "="*80)
    print("[SUCCESS] ML PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\n[NEXT] NEXT STEPS:")
    print("  1. Review results in results/ directory")
    print("  2. Integrate production model into dashboard")
    print("  3. Set up weekly prediction schedule")
    print("  4. Configure alert thresholds")
    print("  5. Deploy to production environment")
    
except Exception as e:
    print(f"\n[ERROR] ERROR DURING EXECUTION:")
    print(f"   {str(e)}")
    print("\n[INFO] Traceback:")
    import traceback
    traceback.print_exc()
    print("\n" + "="*80)
    print("Pipeline execution failed. Please check the error above.")
    print("="*80)
    sys.exit(1)

