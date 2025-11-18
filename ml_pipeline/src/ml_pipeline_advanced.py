"""
HSU EARLY WARNING SYSTEM - ADVANCED ML PIPELINE (Steps 10-14)
==============================================================
Ensemble Methods, Evaluation, Explainability, and Production Pipeline

This script continues from ml_pipeline_complete.py with:
- Step 10: Ensemble Methods
- Step 11: Model Evaluation
- Step 12: SHAP Explainability
- Step 13: Model Persistence
- Step 14: Production Pipeline
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Core ML Libraries
# Ensembles not used - only Random Forest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve
)

# Advanced ML Models
# Only Random Forest is used
from sklearn.ensemble import RandomForestClassifier

# Explainability
import shap

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Utilities
import joblib
import os
from datetime import datetime
import json


# ============================================================================
# STEP 10: ENSEMBLE METHODS
# ============================================================================
print("\n" + "="*80)
print("STEP 10: ENSEMBLE METHODS")
print("="*80)

def create_stacking_ensemble(models, X_train, y_train, X_val, y_val):
    """
    Stacking ensemble disabled - only Random Forest is used
    
    Returns:
        tuple: (None, None) - Ensembles not used with single model
    """
    print("\nStacking Ensemble: Not used (only Random Forest model)")
    print("  Returning Random Forest model directly")
    
    # Return None since we're not using ensembles
    return None, None


def create_voting_ensemble(models, X_val, y_val):
    """
    Voting ensemble disabled - only Random Forest is used
    
    Returns:
        tuple: (None, None) - Ensembles not used with single model
    """
    print("\nVoting Ensemble: Not used (only Random Forest model)")
    print("  Returning Random Forest model directly")
    
    # Return None since we're not using ensembles
    return None, None


# ============================================================================
# STEP 11: MODEL EVALUATION
# ============================================================================
print("\n" + "="*80)
print("STEP 11: MODEL EVALUATION")
print("="*80)

def evaluate_model_comprehensive(model, X_test, y_test, model_name='Model', threshold=0.35):
    """
    Comprehensive model evaluation on test set
    
    Metrics:
    - ROC-AUC
    - Confusion Matrix
    - Classification Report
    - Precision-Recall Curve
    
    Args:
        threshold: Decision threshold for binary classification (default: 0.50)
                   Lower threshold improves recall (catches more at-risk students)
    
    Returns:
        dict: Comprehensive metrics
    """
    print(f"\n{'='*80}")
    print(f"EVALUATING {model_name.upper()} ON TEST SET")
    print(f"{'='*80}")
    print(f"Decision Threshold: {threshold:.2f}")
    
    # Predictions with custom threshold (lower threshold = better recall)
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    
    # Basic metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    
    print(f"\n1. BASIC METRICS")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  AUC-ROC:   {auc:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"\n2. CONFUSION MATRIX")
    print(f"  True Negatives:  {tn:,}")
    print(f"  False Positives: {fp:,}")
    print(f"  False Negatives: {fn:,}")
    print(f"  True Positives:  {tp:,}")
    
    # Calculate additional metrics
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    print(f"\n3. ADDITIONAL METRICS")
    print(f"  Specificity (True Negative Rate): {specificity:.4f}")
    print(f"  Negative Predictive Value:        {npv:.4f}")
    print(f"  False Positive Rate:              {fpr:.4f}")
    print(f"  False Negative Rate:              {fnr:.4f}")
    
    # Classification report
    print(f"\n4. CLASSIFICATION REPORT")
    print(classification_report(y_test, y_pred, target_names=['Retained', 'At-Risk']))
    
    # Collect metrics
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'confusion_matrix': cm.tolist(),
        'specificity': specificity,
        'npv': npv,
        'fpr': fpr,
        'fnr': fnr,
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_positives': int(tp)
    }
    
    return metrics


def plot_roc_curve(models, X_test, y_test, save_path='results/roc_curve.png'):
    """
    Plot ROC curve for Random Forest model
    """
    print("\n5. GENERATING ROC CURVE COMPARISON")
    
    plt.figure(figsize=(10, 8))
    
    # Only plot Random Forest
    if 'random_forest' in models:
        model = models['random_forest']['model']
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        
        plt.plot(fpr, tpr, label=f'Random Forest (AUC = {auc:.4f})', linewidth=2)
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve - Random Forest Model', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  + ROC curve saved to: {save_path}")


def plot_confusion_matrix(cm, model_name, save_path='results/confusion_matrix.png'):
    """
    Plot confusion matrix heatmap
    """
    plt.figure(figsize=(8, 6))
    
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=['Retained', 'At-Risk'],
        yticklabels=['Retained', 'At-Risk'],
        cbar_kws={'label': 'Count'}
    )
    
    plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  + Confusion matrix saved to: {save_path}")


# ============================================================================
# STEP 12: SHAP EXPLAINABILITY
# ============================================================================
print("\n" + "="*80)
print("STEP 12: SHAP EXPLAINABILITY")
print("="*80)

def explain_model_shap(model, X_sample, feature_names, model_name='Random Forest'):
    """
    Generate SHAP explanations for model predictions
    
    Returns:
        shap_values, explainer
    """
    print(f"\nGenerating SHAP explanations for {model_name}...")
    print(f"  Sample size: {len(X_sample)} students")
    
    # Create SHAP explainer
    if model_name in ['Random Forest']:
        explainer = shap.TreeExplainer(model)
    else:
        # Use KernelExplainer for other models (slower)
        explainer = shap.KernelExplainer(
            model.predict_proba, 
            shap.sample(X_sample, 100)
        )
    
    # Calculate SHAP values
    print("  Calculating SHAP values...")
    shap_values = explainer.shap_values(X_sample)
    
    # Handle different SHAP value formats
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Use positive class
    
    print("  + SHAP values calculated")
    
    return shap_values, explainer


def plot_shap_summary(shap_values, X_sample, feature_names, model_name='Random Forest',
                      save_path='results/shap_summary.png'):
    """
    Plot SHAP summary plot (feature importance)
    """
    print(f"\n  Generating SHAP summary plot...")
    
    plt.figure(figsize=(12, 10))
    
    # Convert to DataFrame for plotting
    X_df = pd.DataFrame(X_sample, columns=feature_names)
    
    shap.summary_plot(
        shap_values, 
        X_df,
        max_display=20,
        show=False
    )
    
    plt.title(f'SHAP Feature Importance - {model_name}', 
              fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  + SHAP summary saved to: {save_path}")


def get_feature_importance(shap_values, feature_names, top_n=20):
    """
    Get top N most important features from SHAP values
    
    Returns:
        DataFrame: Feature importance rankings
    """
    # Calculate mean absolute SHAP values
    # Handle different SHAP value shapes
    if len(shap_values.shape) == 3:
        # For multi-class or multi-output, take mean across classes
        mean_abs_shap = np.abs(shap_values).mean(axis=(0, 1))
    elif len(shap_values.shape) == 2:
        # For binary classification, take mean across samples
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        # If still 2D, take mean again
        if len(mean_abs_shap.shape) > 1:
            mean_abs_shap = mean_abs_shap.mean(axis=0)
    else:
        # For 1D array
        mean_abs_shap = np.abs(shap_values)
    
    # Ensure we have the right number of features
    if len(mean_abs_shap) != len(feature_names):
        # If shape mismatch, try to flatten
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        if len(mean_abs_shap.shape) > 1:
            mean_abs_shap = mean_abs_shap.flatten()
        # Take first len(feature_names) values
        mean_abs_shap = mean_abs_shap[:len(feature_names)]
    
    # Ensure it's 1D
    mean_abs_shap = mean_abs_shap.flatten() if len(mean_abs_shap.shape) > 1 else mean_abs_shap
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names[:len(mean_abs_shap)],
        'importance': mean_abs_shap
    }).sort_values('importance', ascending=False)
    
    print(f"\n  TOP {top_n} MOST IMPORTANT FEATURES:")
    print("  " + "-" * 60)
    for idx, row in importance_df.head(top_n).iterrows():
        print(f"  {idx+1:2d}. {row['feature']:40s} {row['importance']:.4f}")
    
    return importance_df


# ============================================================================
# STEP 13: MODEL PERSISTENCE
# ============================================================================
print("\n" + "="*80)
print("STEP 13: MODEL PERSISTENCE")
print("="*80)

def save_models(models, scaler, feature_names, model_dir='models'):
    """
    Save Random Forest model, scaler, and metadata
    
    Saves:
    - Random Forest model
    - Scaler
    - Feature names
    - Model metadata
    """
    print(f"\nSaving models to: {model_dir}/")
    
    os.makedirs(model_dir, exist_ok=True)
    
    # Save only Random Forest model
    if 'random_forest' in models:
        model_path = f"{model_dir}/random_forest_model.pkl"
        joblib.dump(models['random_forest']['model'], model_path)
        print(f"  + Saved random_forest to {model_path}")
    
    # Save scaler
    scaler_path = f"{model_dir}/scaler.pkl"
    joblib.dump(scaler, scaler_path)
    print(f"  + Saved scaler to {scaler_path}")
    
    # Save feature names
    features_path = f"{model_dir}/feature_names.json"
    with open(features_path, 'w') as f:
        json.dump(feature_names, f, indent=2)
    print(f"  + Saved feature names to {features_path}")
    
    # Save metadata
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'num_features': len(feature_names),
        'models': list(models.keys()),
        'version': '1.0'
    }
    
    metadata_path = f"{model_dir}/metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  + Saved metadata to {metadata_path}")


def load_production_model(model_name='random_forest', model_dir='models'):
    """
    Load production model and dependencies
    
    Returns:
        tuple: (model, scaler, feature_names)
    """
    print(f"\nLoading {model_name} model from: {model_dir}/")
    
    # Load model
    model_path = f"{model_dir}/{model_name}_model.pkl"
    model = joblib.load(model_path)
    print(f"  + Loaded model from {model_path}")
    
    # Load scaler
    scaler_path = f"{model_dir}/scaler.pkl"
    scaler = joblib.load(scaler_path)
    print(f"  + Loaded scaler from {scaler_path}")
    
    # Load feature names
    features_path = f"{model_dir}/feature_names.json"
    with open(features_path, 'r') as f:
        feature_names = json.load(f)
    print(f"  + Loaded {len(feature_names)} feature names")
    
    return model, scaler, feature_names


# ============================================================================
# STEP 14: PRODUCTION PIPELINE
# ============================================================================
print("\n" + "="*80)
print("STEP 14: PRODUCTION PIPELINE")
print("="*80)

class StudentRiskPredictor:
    """
    Production-ready predictor for student dropout risk
    
    Usage:
        predictor = StudentRiskPredictor(model, scaler, feature_names)
        risk_score = predictor.predict_single(student_features)
        risk_scores = predictor.predict_batch(student_features_df)
    """
    
    def __init__(self, model, scaler, feature_names):
        self.model = model
        self.scaler = scaler
        self.feature_names = feature_names
        
        # Risk thresholds (OPTIMIZED FOR RECALL - Two-tier system)
        # Optimized threshold: 0.35 for 77.42% recall (Target: 75-85%)
        # High Risk: >= 0.35 (Immediate intervention) - catches 77% of at-risk
        # Medium Risk: 0.30-0.35 (Monitoring)
        # Low Risk: < 0.30 (No action)
        self.HIGH_RISK = 0.35  # Optimized for 77.42% recall (97.06% precision, 0.69% FPR)
        self.MEDIUM_RISK = 0.30  # Monitoring threshold
        self.LOW_RISK = 0.25  # No action threshold
    
    def predict_single(self, student_features):
        """
        Predict risk for a single student
        
        Args:
            student_features: Dict or Series with feature values
        
        Returns:
            dict: Prediction results
        """
        # Convert to DataFrame
        if isinstance(student_features, dict):
            features_df = pd.DataFrame([student_features])
        else:
            features_df = pd.DataFrame([student_features.to_dict()])
        
        # Ensure correct feature order
        features_df = features_df[self.feature_names]
        
        # Scale features
        features_scaled = self.scaler.transform(features_df)
        
        # Predict
        risk_score = self.model.predict_proba(features_scaled)[0, 1]
        risk_category = self._categorize_risk(risk_score)
        
        return {
            'risk_score': float(risk_score),
            'risk_category': risk_category,
            'requires_intervention': risk_score >= self.MEDIUM_RISK,
            'immediate_action': risk_score >= self.HIGH_RISK
        }
    
    def predict_batch(self, students_df):
        """
        Predict risk for multiple students
        
        Args:
            students_df: DataFrame with student features
        
        Returns:
            DataFrame: Predictions for all students
        """
        # Ensure correct feature order
        features_df = students_df[self.feature_names]
        
        # Scale features
        features_scaled = self.scaler.transform(features_df)
        
        # Predict
        risk_scores = self.model.predict_proba(features_scaled)[:, 1]
        risk_categories = [self._categorize_risk(score) for score in risk_scores]
        
        # Create results DataFrame
        results = pd.DataFrame({
            'StudentID': students_df['StudentID'] if 'StudentID' in students_df.columns else range(len(students_df)),
            'risk_score': risk_scores,
            'risk_category': risk_categories,
            'requires_intervention': risk_scores >= self.MEDIUM_RISK,
            'immediate_action': risk_scores >= self.HIGH_RISK
        })
        
        return results
    
    def _categorize_risk(self, score):
        """
        Categorize risk score using optimized thresholds for recall
        
        Two-tier system (Optimized for 77% recall):
        - High Risk (>=0.35): Immediate intervention required
        - Medium Risk (0.30-0.35): Monitor closely
        - Low Risk (<0.30): Normal monitoring
        """
        if score >= self.HIGH_RISK:
            return 'High'
        elif score >= self.MEDIUM_RISK:
            return 'Medium'
        else:
            return 'Low'
    
    def get_intervention_recommendations(self, risk_score, student_features):
        """
        Get intervention recommendations based on risk score
        
        Returns:
            list: Recommended interventions
        """
        recommendations = []
        
        if risk_score >= self.HIGH_RISK:
            recommendations.append("URGENT: Assign case manager")
            recommendations.append("Schedule immediate advisor meeting")
            
        if student_features.get('Rule1_GPA_Below_2', 0) == 1:
            recommendations.append("Mandatory tutoring program")
            
        if student_features.get('Rule2_Attendance_Below_80', 0) == 1:
            recommendations.append("Contact student about attendance")
            
        if student_features.get('Rule5_High_Balance', 0) == 1:
            recommendations.append("Financial aid consultation")
            
        if student_features.get('Rule9_High_Severity', 0) == 1:
            recommendations.append("Mental health support services")
            
        if not recommendations:
            recommendations.append("Continue monitoring")
        
        return recommendations


def demo_production_pipeline():
    """
    Demonstrate production pipeline usage
    """
    print("\n" + "="*80)
    print("PRODUCTION PIPELINE DEMONSTRATION")
    print("="*80)
    
    print("\nThis pipeline can:")
    print("  1. Predict risk for single students (real-time)")
    print("  2. Predict risk for batches (weekly updates)")
    print("  3. Generate intervention recommendations")
    print("  4. Support dashboard integration")
    print("  5. Enable alert generation")
    
    print("\nExample Usage:")
    print("""
    # Load production model
    model, scaler, feature_names = load_production_model('random_forest')
    
    # Create predictor
    predictor = StudentRiskPredictor(model, scaler, feature_names)
    
    # Single prediction
    result = predictor.predict_single(student_features)
    print(f"Risk Score: {result['risk_score']:.2%}")
    print(f"Category: {result['risk_category']}")
    
    # Batch prediction
    results_df = predictor.predict_batch(students_df)
    high_risk = results_df[results_df['immediate_action'] == True]
    print(f"High-risk students: {len(high_risk)}")
    """)


if __name__ == "__main__":
    print("\n" + "="*80)
    print("HSU EARLY WARNING SYSTEM - ADVANCED ML PIPELINE")
    print("="*80)
    print("\nThis script implements Steps 10-14:")
    print("  Step 10: Ensemble Methods")
    print("  Step 11: Model Evaluation")
    print("  Step 12: SHAP Explainability")
    print("  Step 13: Model Persistence")
    print("  Step 14: Production Pipeline")
    print("\nTo run this script, first execute ml_pipeline_complete.py")
    print("="*80)
    
    demo_production_pipeline()

