"""
HSU EARLY WARNING SYSTEM - PRODUCTION PREDICTION SCRIPT
========================================================
Weekly batch prediction script for production use

This script:
1. Loads production model (Random Forest)
2. Processes current student data
3. Generates risk predictions
4. Creates intervention alerts
5. Exports results for dashboard

Usage:
    python production_predict.py
    
Schedule: Run weekly (Friday night for Monday action)
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from ml_pipeline_complete import load_data, engineer_features
from ml_pipeline_advanced import load_production_model, StudentRiskPredictor

# Configuration (OPTIMIZED FOR RECALL)
# Optimized threshold: 0.35 for 77.42% recall (Target: 75-85%)
HIGH_RISK_THRESHOLD = 0.35  # Optimized for 77.42% recall (97.06% precision, 0.69% FPR)
MEDIUM_RISK_THRESHOLD = 0.30  # Monitoring threshold
OUTPUT_DIR = 'production_output'

def main():
    """
    Main production prediction workflow
    """
    print("="*80)
    print("HSU EARLY WARNING SYSTEM - WEEKLY PREDICTION")
    print("="*80)
    print(f"Execution Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    try:
        # Step 1: Load current data
        print("STEP 1: Loading Current Student Data")
        print("-" * 80)
        datasets = load_data()
        print(f"+ Loaded data for {len(datasets['students']):,} students\n")
        
        # Step 2: Engineer features
        print("STEP 2: Engineering Features")
        print("-" * 80)
        features = engineer_features(datasets)
        print(f"+ Engineered {len(features.columns)-1} features\n")
        
        # Step 3: Load production model
        print("STEP 3: Loading Production Model")
        print("-" * 80)
        model, scaler, feature_names = load_production_model('random_forest')
        predictor = StudentRiskPredictor(model, scaler, feature_names)
        print("+ Loaded Random Forest production model\n")
        
        # Step 4: Generate predictions
        print("STEP 4: Generating Risk Predictions")
        print("-" * 80)
        predictions = predictor.predict_batch(features)
        print(f"+ Generated predictions for {len(predictions):,} students\n")
        
        # Step 5: Categorize and prioritize
        print("STEP 5: Risk Categorization")
        print("-" * 80)
        
        high_risk = predictions[predictions['risk_category'] == 'High']
        medium_risk = predictions[predictions['risk_category'] == 'Medium']
        low_risk = predictions[predictions['risk_category'] == 'Low']
        
        print(f"  High Risk:   {len(high_risk):,} students ({len(high_risk)/len(predictions)*100:.1f}%)")
        print(f"  Medium Risk: {len(medium_risk):,} students ({len(medium_risk)/len(predictions)*100:.1f}%)")
        print(f"  Low Risk:    {len(low_risk):,} students ({len(low_risk)/len(predictions)*100:.1f}%)\n")
        
        # Step 6: Generate alerts
        print("STEP 6: Generating Intervention Alerts")
        print("-" * 80)
        
        # High risk students (immediate action)
        high_risk_students = datasets['students'][
            datasets['students']['StudentID'].isin(high_risk['StudentID'])
        ]
        
        alerts = []
        for idx, student in high_risk_students.iterrows():
            student_id = student['StudentID']
            risk_info = predictions[predictions['StudentID'] == student_id].iloc[0]
            
            # Get student features for intervention recommendations
            student_features = features[features['StudentID'] == student_id].iloc[0]
            recommendations = predictor.get_intervention_recommendations(
                risk_info['risk_score'],
                student_features.to_dict()
            )
            
            alert = {
                'StudentID': student_id,
                'BannerID': student['BannerID'],
                'FirstName': student['FirstName'],
                'LastName': student['LastName'],
                'Email': student['Email'],
                'RiskScore': f"{risk_info['risk_score']:.3f}",
                'RiskCategory': risk_info['risk_category'],
                'Recommendations': '; '.join(recommendations),
                'AlertDate': datetime.now().strftime('%Y-%m-%d'),
                'AlertType': 'HIGH_PRIORITY'
            }
            alerts.append(alert)
        
        alerts_df = pd.DataFrame(alerts)
        print(f"  + Generated {len(alerts_df)} HIGH PRIORITY alerts\n")
        
        # Step 7: Export results
        print("STEP 7: Exporting Results")
        print("-" * 80)
        
        # All predictions
        predictions_file = f"{OUTPUT_DIR}/weekly_predictions_{datetime.now().strftime('%Y%m%d')}.csv"
        predictions.to_csv(predictions_file, index=False)
        print(f"  + Saved all predictions: {predictions_file}")
        
        # High-risk alerts
        alerts_file = f"{OUTPUT_DIR}/high_risk_alerts_{datetime.now().strftime('%Y%m%d')}.csv"
        alerts_df.to_csv(alerts_file, index=False)
        print(f"  + Saved high-risk alerts: {alerts_file}")
        
        # Summary report
        summary = {
            'prediction_date': datetime.now().isoformat(),
            'total_students': len(predictions),
            'high_risk_count': len(high_risk),
            'medium_risk_count': len(medium_risk),
            'low_risk_count': len(low_risk),
            'alerts_generated': len(alerts_df),
            'model_used': 'Random Forest',
            'thresholds': {
                'high_risk': HIGH_RISK_THRESHOLD,
                'medium_risk': MEDIUM_RISK_THRESHOLD
            }
        }
        
        import json
        summary_file = f"{OUTPUT_DIR}/prediction_summary_{datetime.now().strftime('%Y%m%d')}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"  + Saved summary report: {summary_file}\n")
        
        # Step 8: Display sample alerts
        print("STEP 8: Sample High-Risk Alerts")
        print("-" * 80)
        
        if len(alerts_df) > 0:
            print("\n  Top 5 High-Risk Students:\n")
            for i, row in alerts_df.head(5).iterrows():
                print(f"  {i+1}. {row['FirstName']} {row['LastName']} (ID: {row['StudentID']})")
                print(f"     Risk Score: {row['RiskScore']}")
                print(f"     Recommendations: {row['Recommendations']}")
                print()
        else:
            print("  No high-risk students identified this week! 🎉\n")
        
        # Success
        print("="*80)
        print("✅ WEEKLY PREDICTION COMPLETED SUCCESSFULLY")
        print("="*80)
        
        print(f"\n📧 NEXT STEPS:")
        print(f"  1. Review high-risk alerts: {alerts_file}")
        print(f"  2. Send alerts to advisors")
        print(f"  3. Update dashboard with predictions")
        print(f"  4. Schedule follow-up for next week")
        
        return predictions, alerts_df, summary
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None


if __name__ == "__main__":
    predictions, alerts, summary = main()
    
    if predictions is not None:
        print(f"\n🎯 Prediction complete!")
        print(f"   Total predictions: {len(predictions):,}")
        print(f"   Alerts generated: {len(alerts):,}")
        print(f"\n   Files saved to: {OUTPUT_DIR}/")

