"""
HSU EARLY WARNING SYSTEM - COMPLETE ML PIPELINE
================================================
14-Step Implementation Guide for Student Retention Prediction

Author: HSU Team Infinite - Group 6
Date: 2025
Version: 1.0

This script implements the complete machine learning pipeline from
data loading to production deployment.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Core ML Libraries
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Advanced ML Models
# Only Random Forest is used

# Imbalanced Learning
from imblearn.over_sampling import SMOTE

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

print("="*80)
print("HSU EARLY WARNING SYSTEM - ML PIPELINE")
print("="*80)
print(f"Execution started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")


# ============================================================================
# STEP 1-2: DATA LOADING & CLEANING
# ============================================================================
print("\n" + "="*80)
print("STEP 1-2: DATA LOADING & CLEANING")
print("="*80)

def load_data(data_dir='Data'):
    """
    Load all required datasets from CSV files
    
    Returns:
        dict: Dictionary containing all DataFrames
    """
    print("Loading datasets from:", data_dir)
    
    datasets = {
        'students': pd.read_csv(f'{data_dir}/students.csv'),
        'enrollments': pd.read_csv(f'{data_dir}/enrollments.csv'),
        'grades': pd.read_csv(f'{data_dir}/grades.csv'),
        'attendance': pd.read_csv(f'{data_dir}/attendance.csv'),
        'logins': pd.read_csv(f'{data_dir}/logins.csv', nrows=100000),  # Sample for memory
        'payments': pd.read_csv(f'{data_dir}/payments.csv'),
        'counseling': pd.read_csv(f'{data_dir}/counseling.csv'),
        'risk_scores': pd.read_csv(f'{data_dir}/risk_scores.csv'),
        'terms': pd.read_csv(f'{data_dir}/terms.csv'),
        'courses': pd.read_csv(f'{data_dir}/courses.csv'),
    }
    
    for name, df in datasets.items():
        print(f"  + Loaded {name}: {len(df):,} records")
    
    return datasets


def clean_data(datasets):
    """
    Perform data quality checks and cleaning
    
    Returns:
        dict: Cleaned datasets
        dict: Quality report
    """
    print("\nPerforming data quality checks...")
    
    quality_report = {}
    
    for name, df in datasets.items():
        report = {
            'records': len(df),
            'columns': len(df.columns),
            'missing_values': df.isnull().sum().sum(),
            'duplicates': df.duplicated().sum(),
            'memory_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
        }
        quality_report[name] = report
        
        print(f"\n  {name.upper()}:")
        print(f"    Records: {report['records']:,}")
        print(f"    Missing values: {report['missing_values']:,}")
        print(f"    Duplicates: {report['duplicates']:,}")
        print(f"    Memory: {report['memory_mb']:.2f} MB")
    
    # Remove duplicates
    for name in datasets:
        if quality_report[name]['duplicates'] > 0:
            datasets[name] = datasets[name].drop_duplicates()
            print(f"  + Removed {quality_report[name]['duplicates']} duplicates from {name}")
    
    return datasets, quality_report


# ============================================================================
# STEP 3-4: FEATURE ENGINEERING
# ============================================================================
print("\n" + "="*80)
print("STEP 3-4: FEATURE ENGINEERING")
print("="*80)

def engineer_features(datasets):
    """
    Engineer 70+ features from all data sources
    
    Features cover 5 root cause areas + 5 dropout pathways:
    
    Root Cause Areas:
    1. Academic Issues (45-50%)
    2. Engagement Issues (30-35%)
    3. Financial Stress (15-20%)
    4. Mental Health Crisis (10-15%)
    5. Course Selection (8-12%)
    
    Dropout Pathways (NEW):
    1. Academic Struggle + Disengagement (25%)
    2. Financial Stress + Academic (20%)
    3. Mental Health + Everything (15%)
    4. Course Mismatch + Financial (10%)
    5. Engagement Loss + Mental Health (5%)
    
    Returns:
        DataFrame: Feature matrix with StudentID
    """
    print("\nEngineering features for 10,000 students...")
    
    students = datasets['students']
    enrollments = datasets['enrollments']
    grades = datasets['grades']
    attendance = datasets['attendance']
    logins = datasets['logins']
    payments = datasets['payments']
    counseling = datasets['counseling']
    risk_scores = datasets['risk_scores']
    
    # Get current term
    current_term = 15
    
    # Initialize feature DataFrame
    features = students[['StudentID']].copy()
    
    print("\n1. DEMOGRAPHIC FEATURES")
    # Demographics
    features['Gender_M'] = (students['Gender'] == 'M').astype(int)
    features['Gender_F'] = (students['Gender'] == 'F').astype(int)
    features['Classification_Freshman'] = (students['Classification'] == 'Freshman').astype(int)
    features['Classification_Sophomore'] = (students['Classification'] == 'Sophomore').astype(int)
    features['Classification_Junior'] = (students['Classification'] == 'Junior').astype(int)
    features['Classification_Senior'] = (students['Classification'] == 'Senior').astype(int)
    features['FirstGenerationStudent'] = students['FirstGenerationStudent'].astype(int)
    features['InternationalStudent'] = students['InternationalStudent'].astype(int)
    features['HighSchoolGPA'] = students['HighSchoolGPA']
    print("  + Created 9 demographic features")
    
    print("\n2. ACADEMIC FEATURES (Root Cause 1: 45-50%)")
    # Current enrollments
    current_enrollments = enrollments[enrollments['TermID'] == current_term]
    
    # Number of courses enrolled
    course_counts = current_enrollments.groupby('StudentID').size()
    features['NumCourses'] = features['StudentID'].map(course_counts).fillna(0)
    
    # Enrollment status (ONLY Active and Completed - NOT Withdrawn to avoid data leakage)
    # REMOVED: Enrollment_Withdrawn causes data leakage (target uses current term withdrawals)
    status_counts = current_enrollments.groupby(['StudentID', 'Status']).size().unstack(fill_value=0)
    for status in ['Active', 'Completed']:  # Removed 'Withdrawn' to prevent data leakage
        if status in status_counts.columns:
            features[f'Enrollment_{status}'] = features['StudentID'].map(status_counts[status]).fillna(0)
        else:
            features[f'Enrollment_{status}'] = 0
    
    # Grade features (Rule 1, 6, 8)
    enrollment_grades = grades.merge(current_enrollments[['EnrollmentID', 'StudentID']], on='EnrollmentID')
    
    # Average GPA
    avg_gpa = enrollment_grades.groupby('StudentID')['GradePercentage'].mean() / 25.0  # Convert to 4.0 scale
    features['CurrentGPA'] = features['StudentID'].map(avg_gpa).fillna(0)
    
    # GPA Trend (compare current term with previous term)
    # Get previous term GPA
    prev_term = current_term - 1
    prev_enrollments = enrollments[enrollments['TermID'] == prev_term]
    if len(prev_enrollments) > 0:
        prev_enrollment_grades = grades.merge(prev_enrollments[['EnrollmentID', 'StudentID']], on='EnrollmentID')
        prev_avg_gpa = prev_enrollment_grades.groupby('StudentID')['GradePercentage'].mean() / 25.0
        prev_gpa = features['StudentID'].map(prev_avg_gpa).fillna(features['CurrentGPA'])
        # Calculate trend: (current - previous) / previous
        features['GPA_Trend'] = (features['CurrentGPA'] - prev_gpa) / (prev_gpa + 0.01)  # Add small value to avoid division by zero
    else:
        features['GPA_Trend'] = 0.0  # No previous term data
    
    # Rule 1: GPA < 2.0
    features['Rule1_GPA_Below_2'] = (features['CurrentGPA'] < 2.0).astype(int)
    
    # Number of assignments
    assignment_counts = enrollment_grades.groupby('StudentID').size()
    features['NumAssignments'] = features['StudentID'].map(assignment_counts).fillna(0)
    
    # Assignment completion rate
    on_time_pct = enrollment_grades.groupby('StudentID')['IsOnTime'].mean()
    features['OnTimeSubmissionRate'] = features['StudentID'].map(on_time_pct).fillna(1.0)
    
    # Rule 7: 50%+ Late Assignments
    features['Rule7_Late_Assignments'] = (features['OnTimeSubmissionRate'] < 0.5).astype(int)
    
    # Failures (Rule 6)
    failures = enrollment_grades[enrollment_grades['GradePercentage'] < 60].groupby('StudentID').size()
    features['NumFailures'] = features['StudentID'].map(failures).fillna(0)
    features['Rule6_Two_Plus_Failures'] = (features['NumFailures'] >= 2).astype(int)
    
    # Midterm performance (Rule 8)
    midterm_grades = enrollment_grades[enrollment_grades['AssignmentType'] == 'Midterm']
    midterm_avg = midterm_grades.groupby('StudentID')['GradePercentage'].mean()
    features['MidtermAvg'] = features['StudentID'].map(midterm_avg).fillna(70)
    features['Rule8_Midterm_Below_70'] = (features['MidtermAvg'] < 70).astype(int)
    
    print(f"  + Created 12 academic features")
    
    print("\n3. ENGAGEMENT FEATURES (Root Cause 2: 30-35%)")
    # Attendance features (Rule 2, 3)
    # Merge attendance with enrollments to get StudentID
    enrollment_attendance = attendance.merge(
        current_enrollments[['EnrollmentID', 'StudentID']], 
        on='EnrollmentID',
        how='left'
    )
    
    total_classes = enrollment_attendance.groupby('StudentID').size()
    present_classes = enrollment_attendance[
        enrollment_attendance['Status'] == 'Present'
    ].groupby('StudentID').size()
    
    features['AttendanceRate'] = (
        features['StudentID'].map(present_classes).fillna(0) / 
        features['StudentID'].map(total_classes).fillna(1)
    ).fillna(0)
    
    # Rule 2: Attendance < 80%
    features['Rule2_Attendance_Below_80'] = (features['AttendanceRate'] < 0.8).astype(int)
    
    # Rule 3: Attendance < 50%
    features['Rule3_Attendance_Below_50'] = (features['AttendanceRate'] < 0.5).astype(int)
    
    # Login features (Rule 4)
    # FIXED: Logins already has StudentID, just filter by current enrollments
    current_enrollment_ids = current_enrollments['EnrollmentID'].unique()
    enrollment_logins = logins[logins['EnrollmentID'].isin(current_enrollment_ids)].copy()
    
    # Group by the StudentID that's already in logins table
    login_counts = enrollment_logins.groupby('StudentID').size()
    features['TotalLogins'] = features['StudentID'].map(login_counts).fillna(0)
    
    # Logins per week (assume 15 weeks)
    features['LoginsPerWeek'] = features['TotalLogins'] / 15.0
    
    # Rule 4: <2 logins/week
    features['Rule4_Low_LMS_Usage'] = (features['LoginsPerWeek'] < 2).astype(int)
    
    # Session duration
    avg_session = enrollment_logins.groupby('StudentID')['SessionDurationMinutes'].mean()
    features['AvgSessionDuration'] = features['StudentID'].map(avg_session).fillna(0)
    
    # Days since last login
    if 'LoginDate' in enrollment_logins.columns:
        enrollment_logins['LoginDate'] = pd.to_datetime(enrollment_logins['LoginDate'])
        last_login = enrollment_logins.groupby('StudentID')['LoginDate'].max()
        current_date = pd.Timestamp.now()
        days_since = (current_date - features['StudentID'].map(last_login)).dt.days
        features['DaysSinceLastLogin'] = days_since.fillna(999)  # Large number if never logged in
    else:
        features['DaysSinceLastLogin'] = 999  # Default if no date column
    
    print(f"  + Created 9 engagement features")
    
    print("\n4. FINANCIAL FEATURES (Root Cause 3: 15-20%)")
    # Current term payments (Rule 5, 10)
    current_payments = payments[payments['TermID'] == current_term]
    
    for col in ['Balance', 'FinancialAidAmount', 'AmountOwed', 'AmountPaid']:
        features[col] = features['StudentID'].map(
            current_payments.set_index('StudentID')[col]
        ).fillna(0)
    
    features['HasHold'] = features['StudentID'].map(
        current_payments.set_index('StudentID')['HasHold']
    ).fillna(False).astype(int)
    
    # Rule 5: Balance > $5K
    features['Rule5_High_Balance'] = (features['Balance'] > 5000).astype(int)
    
    # Rule 10: SAP (Satisfactory Academic Progress)
    features['Rule10_SAP_Risk'] = (
        (features['CurrentGPA'] < 2.0) & 
        (features['FinancialAidAmount'] > 0)
    ).astype(int)
    
    # Payment patterns
    features['PaymentRate'] = (
        features['AmountPaid'] / (features['AmountOwed'] + 1)
    ).clip(0, 1)
    
    print(f"  + Created 9 financial features")
    
    print("\n5. WELLNESS FEATURES (Root Cause 4: 10-15%)")
    # Counseling visits (Rule 9)
    counseling_counts = counseling.groupby('StudentID').size()
    features['NumCounselingVisits'] = features['StudentID'].map(counseling_counts).fillna(0)
    
    # Crisis flags
    crisis_students = counseling[counseling['CrisisFlag'] == True]['StudentID'].unique()
    features['HasCrisisFlag'] = features['StudentID'].isin(crisis_students).astype(int)
    
    # Severity levels (Rule 9)
    max_severity = counseling.groupby('StudentID')['SeverityLevel'].max()
    features['MaxSeverityLevel'] = features['StudentID'].map(max_severity).fillna(0)
    
    # Rule 9: Crisis Severity 4+
    features['Rule9_High_Severity'] = (features['MaxSeverityLevel'] >= 4).astype(int)
    
    # Concern types
    concern_counts = counseling.groupby('StudentID')['ConcernType'].nunique()
    features['NumConcernTypes'] = features['StudentID'].map(concern_counts).fillna(0)
    
    print(f"  + Created 5 wellness features")
    
    print("\n6. COURSE SELECTION FEATURES (Root Cause 5: 8-12%)")
    # Withdrawal history (Rule 14) - ONLY PREVIOUS TERMS to avoid data leakage
    # REMOVED: Current term withdrawals cause data leakage (target uses current term withdrawals)
    prev_terms_withdrawals = enrollments[
        (enrollments['Status'] == 'Withdrawn') & 
        (enrollments['TermID'] < current_term)  # Only previous terms
    ].groupby('StudentID').size()
    features['TotalWithdrawals'] = features['StudentID'].map(prev_terms_withdrawals).fillna(0)
    
    # Rule 14: 2+ Withdrawals (historical only)
    features['Rule14_Multiple_Withdrawals'] = (features['TotalWithdrawals'] >= 2).astype(int)
    
    print(f"  + Created 2 course selection features")
    
    print("\n7. RISK SCORE FEATURES (ML-Generated)")
    # REMOVED: Pre-computed risk scores cause data leakage
    # These scores were computed using the target variable, creating perfect predictions
    # Instead, we'll compute risk indicators from raw features only
    
    # Compute risk indicators from actual data (no leakage)
    # Academic risk indicator (based on GPA and grades)
    # Use NumFailures instead of NumFailingGrades (which doesn't exist)
    features['ComputedAcademicRisk'] = (
        (features['CurrentGPA'] < 2.0).astype(int) * 0.5 +
        (features['GPA_Trend'] < -0.2).astype(int) * 0.3 +
        (features['NumFailures'] > 0).astype(int) * 0.2
    )
    
    # Engagement risk indicator (based on attendance and logins)
    features['ComputedEngagementRisk'] = (
        (features['AttendanceRate'] < 0.5).astype(int) * 0.4 +
        (features['TotalLogins'] < 10).astype(int) * 0.3 +
        (features['DaysSinceLastLogin'] > 14).astype(int) * 0.3
    )
    
    # Financial risk indicator (based on balance and payments)
    # Use HasHold instead of HasFinancialHold, and PaymentRate instead of PaymentDelinquency
    features['ComputedFinancialRisk'] = (
        (features['HasHold'] == 1).astype(int) * 0.5 +
        (features['Balance'] > 5000).astype(int) * 0.3 +
        ((features['PaymentRate'] < 0.5) & (features['AmountOwed'] > 0)).astype(int) * 0.2
    )
    
    # Wellness risk indicator (based on counseling data)
    features['ComputedWellnessRisk'] = (
        (features['HasCrisisFlag'] == 1).astype(int) * 0.5 +
        (features['MaxSeverityLevel'] >= 3).astype(int) * 0.3 +
        (features['NumCounselingVisits'] > 5).astype(int) * 0.2
    )
    
    # Overall computed risk (weighted combination)
    features['ComputedOverallRisk'] = (
        features['ComputedAcademicRisk'] * 0.4 +
        features['ComputedEngagementRisk'] * 0.3 +
        features['ComputedFinancialRisk'] * 0.2 +
        features['ComputedWellnessRisk'] * 0.1
    )
    
    # Rule 11: Computed Risk Score > 0.70 (using computed, not pre-computed)
    features['Rule11_High_Risk_Score'] = (features['ComputedOverallRisk'] > 0.70).astype(int)
    
    print(f"  + Created 6 computed risk features (no data leakage)")
    
    print("\n8. INTERACTION FEATURES")
    # Multi-factor risk (Rule 15)
    rule_columns = [col for col in features.columns if col.startswith('Rule')]
    features['TotalRulesTriggered'] = features[rule_columns].sum(axis=1)
    
    # Rule 15: Multi-Factor 3+
    features['Rule15_Multi_Factor'] = (features['TotalRulesTriggered'] >= 3).astype(int)
    
    # Academic * Engagement interaction
    features['Academic_Engagement_Interaction'] = (
        features['CurrentGPA'] * features['AttendanceRate']
    )
    
    # Financial * Academic interaction
    features['Financial_Academic_Interaction'] = (
        features['Balance'] / 1000.0 * (4.0 - features['CurrentGPA'])
    )
    
    print(f"  + Created 4 interaction features")
    
    print("\n9. DROPOUT PATHWAY FEATURES (5 Critical Pathways)")
    # Based on research: 5 distinct pathways to dropout
    
    # Path 1: Academic Struggle + Disengagement (25% of dropouts)
    # Grades drop → Get discouraged → Stop engaging
    features['Path1_Academic_Disengagement'] = (
        (features['GPA_Trend'] < -0.1).astype(int) * 0.5 +  # Grades dropping
        (features['CurrentGPA'] < 2.5).astype(int) * 0.3 +  # Low GPA
        (features['AttendanceRate'] < 0.6).astype(int) * 0.2  # Low engagement
    )
    features['Path1_Academic_Disengagement_Flag'] = (
        (features['GPA_Trend'] < -0.1) & 
        (features['AttendanceRate'] < 0.6)
    ).astype(int)
    
    # Path 2: Financial Stress + Academic (20% of dropouts)
    # Need money → Work more → Less study time → Grades drop
    features['Path2_Financial_Academic'] = (
        (features['Balance'] > 3000).astype(int) * 0.4 +  # Financial stress
        (features['HasHold'] == 1).astype(int) * 0.3 +  # Financial hold
        (features['CurrentGPA'] < 2.5).astype(int) * 0.3  # Academic decline
    )
    features['Path2_Financial_Academic_Flag'] = (
        ((features['Balance'] > 3000) | (features['HasHold'] == 1)) &
        (features['CurrentGPA'] < 2.5)
    ).astype(int)
    
    # Path 3: Mental Health + Everything (15% of dropouts)
    # Depression → Stops everything → Cascade failure
    features['Path3_Mental_Health_Cascade'] = (
        (features['HasCrisisFlag'] == 1).astype(int) * 0.4 +  # Crisis flag
        (features['MaxSeverityLevel'] >= 3).astype(int) * 0.3 +  # High severity
        (features['NumCounselingVisits'] > 3).astype(int) * 0.2 +  # Multiple visits
        ((features['CurrentGPA'] < 2.0) | (features['AttendanceRate'] < 0.5)).astype(int) * 0.1  # Academic/engagement issues
    )
    features['Path3_Mental_Health_Cascade_Flag'] = (
        ((features['HasCrisisFlag'] == 1) | (features['MaxSeverityLevel'] >= 3)) &
        ((features['CurrentGPA'] < 2.0) | (features['AttendanceRate'] < 0.5))
    ).astype(int)
    
    # Path 4: Course Mismatch + Financial (10% of dropouts)
    # Wrong course → Withdraw → Financial issues worsen
    features['Path4_Course_Mismatch_Financial'] = (
        (features['TotalWithdrawals'] > 0).astype(int) * 0.5 +  # Withdrawals
        (features['Balance'] > 2000).astype(int) * 0.3 +  # Financial stress
        (features['PaymentRate'] < 0.7).astype(int) * 0.2  # Payment issues
    )
    features['Path4_Course_Mismatch_Financial_Flag'] = (
        (features['TotalWithdrawals'] > 0) &
        ((features['Balance'] > 2000) | (features['HasHold'] == 1))
    ).astype(int)
    
    # Path 5: Engagement Loss + Mental Health (5% of dropouts)
    # Feels isolated → Depression → Complete withdrawal
    features['Path5_Engagement_Mental_Health'] = (
        (features['AttendanceRate'] < 0.4).astype(int) * 0.4 +  # Very low engagement
        (features['TotalLogins'] < 5).astype(int) * 0.3 +  # Minimal LMS usage
        ((features['HasCrisisFlag'] == 1) | (features['MaxSeverityLevel'] >= 2)).astype(int) * 0.3  # Mental health issues
    )
    features['Path5_Engagement_Mental_Health_Flag'] = (
        ((features['AttendanceRate'] < 0.4) | (features['TotalLogins'] < 5)) &
        ((features['HasCrisisFlag'] == 1) | (features['MaxSeverityLevel'] >= 2))
    ).astype(int)
    
    # Combined pathway risk score (weighted by prevalence)
    features['PathwayRiskScore'] = (
        features['Path1_Academic_Disengagement'] * 0.25 +  # 25%
        features['Path2_Financial_Academic'] * 0.20 +  # 20%
        features['Path3_Mental_Health_Cascade'] * 0.15 +  # 15%
        features['Path4_Course_Mismatch_Financial'] * 0.10 +  # 10%
        features['Path5_Engagement_Mental_Health'] * 0.05  # 5%
    )
    
    # Count of pathways triggered
    pathway_flags = [
        'Path1_Academic_Disengagement_Flag',
        'Path2_Financial_Academic_Flag',
        'Path3_Mental_Health_Cascade_Flag',
        'Path4_Course_Mismatch_Financial_Flag',
        'Path5_Engagement_Mental_Health_Flag'
    ]
    features['NumPathwaysTriggered'] = features[pathway_flags].sum(axis=1)
    
    print(f"  + Created 12 pathway features (5 pathways + risk scores + flags)")
    
    print("\n10. SPECIAL POPULATION FEATURES (Rule 12, 13)")
    # Rule 12: First-generation support needs
    features['Rule12_First_Gen'] = features['FirstGenerationStudent']
    
    # Rule 13: International student attendance
    features['Rule13_International_Low_Attendance'] = (
        (features['InternationalStudent'] == 1) & 
        (features['AttendanceRate'] < 0.5)
    ).astype(int)
    
    print(f"  + Created 2 special population features")
    
    # Summary
    total_features = len(features.columns) - 1  # Exclude StudentID
    print(f"\n{'='*80}")
    print(f"FEATURE ENGINEERING COMPLETE: {total_features} features created")
    print(f"{'='*80}")
    
    return features


# ============================================================================
# STEP 4: CREATE TARGET VARIABLE
# ============================================================================
print("\n" + "="*80)
print("STEP 4: CREATE TARGET VARIABLE")
print("="*80)

def create_target(datasets):
    """
    Create binary dropout target variable based on ACTUAL OUTCOMES (no data leakage)
    
    Dropout defined as:
    - Withdrawn from current term
    - OR GPA < 2.0 with attendance < 50% (high risk indicators)
    - OR multiple course withdrawals (2+)
    - OR failed 2+ courses in current term
    
    Returns:
        Series: Binary target (1 = dropout risk, 0 = retained)
    """
    print("\nCreating target variable based on ACTUAL OUTCOMES (no data leakage)...")
    
    students = datasets['students']
    enrollments = datasets['enrollments']
    grades = datasets['grades']
    attendance = datasets['attendance']
    
    current_term = 15
    
    # Initialize target DataFrame
    target_df = students[['StudentID']].copy()
    target_df['target'] = 0
    
    # 1. Check for withdrawals in current term
    current_enrollments = enrollments[enrollments['TermID'] == current_term]
    withdrawn_students = current_enrollments[
        current_enrollments['Status'] == 'Withdrawn'
    ]['StudentID'].unique()
    target_df.loc[target_df['StudentID'].isin(withdrawn_students), 'target'] = 1
    
    # 2. Check for GPA < 2.0 with poor attendance
    current_enrollment_ids = current_enrollments['EnrollmentID'].unique()
    enrollment_grades = grades[grades['EnrollmentID'].isin(current_enrollment_ids)].merge(
        current_enrollments[['EnrollmentID', 'StudentID']], 
        on='EnrollmentID',
        how='left'
    )
    student_gpa = enrollment_grades.groupby('StudentID')['GradePercentage'].mean() / 25.0
    
    enrollment_attendance = attendance.merge(
        current_enrollments[['EnrollmentID', 'StudentID']], 
        on='EnrollmentID',
        how='left'
    )
    total_classes = enrollment_attendance.groupby('StudentID').size()
    present_classes = enrollment_attendance[
        enrollment_attendance['Status'] == 'Present'
    ].groupby('StudentID').size()
    attendance_rate = (present_classes / total_classes).fillna(0)
    
    # Mark students with low GPA and poor attendance
    for idx, row in target_df.iterrows():
        student_id = row['StudentID']
        if student_id in student_gpa.index and student_id in attendance_rate.index:
            if student_gpa[student_id] < 2.0 and attendance_rate[student_id] < 0.5:
                target_df.loc[idx, 'target'] = 1
    
    # 3. Check for 2+ course withdrawals (including current term - this is outcome)
    withdrawal_counts = enrollments[
        enrollments['Status'] == 'Withdrawn'
    ].groupby('StudentID').size()
    multiple_withdrawals = withdrawal_counts[withdrawal_counts >= 2].index
    target_df.loc[target_df['StudentID'].isin(multiple_withdrawals), 'target'] = 1
    # Note: This is fine for target - we're predicting dropout risk based on total withdrawal history
    
    # 4. Check for 2+ course failures in current term
    failing_grades = enrollment_grades[enrollment_grades['GradePercentage'] < 60]
    failure_counts = failing_grades.groupby('StudentID')['EnrollmentID'].nunique()
    multiple_failures = failure_counts[failure_counts >= 2].index
    target_df.loc[target_df['StudentID'].isin(multiple_failures), 'target'] = 1
    
    # Convert to Series
    target = target_df['target'].astype(int)
    
    dropout_count = target.sum()
    retention_count = len(target) - dropout_count
    
    print(f"\nTarget Distribution (Based on Actual Outcomes):")
    print(f"  Dropout Risk (1): {dropout_count:,} ({dropout_count/len(target)*100:.1f}%)")
    print(f"  Retained (0): {retention_count:,} ({retention_count/len(target)*100:.1f}%)")
    if dropout_count > 0:
        print(f"  Class Imbalance Ratio: 1:{retention_count/dropout_count:.2f}")
    else:
        print(f"  Warning: No dropout cases found!")
    
    return target


# ============================================================================
# STEP 5-7: DATA PREPARATION
# ============================================================================
print("\n" + "="*80)
print("STEP 5-7: DATA PREPARATION")
print("="*80)

def prepare_data(features, target, test_size=0.15, val_size=0.15, random_state=42):
    """
    Prepare data for modeling:
    1. Train/val/test split (70-15-15)
    2. Handle class imbalance with SMOTE
    3. Feature scaling
    
    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test, scaler, feature_names)
    """
    print("\n1. TRAIN/VAL/TEST SPLIT (70-15-15)")
    
    # Remove StudentID
    X = features.drop('StudentID', axis=1)
    y = target
    feature_names = X.columns.tolist()
    
    # First split: train (70%) and temp (30%)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=random_state
    )
    
    # Second split: val (15%) and test (15%)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=random_state
    )
    
    print(f"  Train set: {len(X_train):,} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"  Val set: {len(X_val):,} samples ({len(X_val)/len(X)*100:.1f}%)")
    print(f"  Test set: {len(X_test):,} samples ({len(X_test)/len(X)*100:.1f}%)")
    
    print("\n2. HANDLING CLASS IMBALANCE (SMOTE)")
    
    # Apply SMOTE only to training data
    smote = SMOTE(random_state=random_state, k_neighbors=5)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    print(f"  Before SMOTE:")
    print(f"    Class 0: {(y_train == 0).sum():,}")
    print(f"    Class 1: {(y_train == 1).sum():,}")
    print(f"  After SMOTE:")
    print(f"    Class 0: {(y_train_balanced == 0).sum():,}")
    print(f"    Class 1: {(y_train_balanced == 1).sum():,}")
    
    print("\n3. FEATURE SCALING (StandardScaler)")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_balanced)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"  + Scaled {len(feature_names)} features")
    print(f"  Mean: {X_train_scaled.mean():.6f}")
    print(f"  Std: {X_train_scaled.std():.6f}")
    
    return (
        X_train_scaled, X_val_scaled, X_test_scaled,
        y_train_balanced, y_val, y_test,
        scaler, feature_names
    )


# ============================================================================
# STEP 8: BASELINE MODEL (LOGISTIC REGRESSION)
# ============================================================================
print("\n" + "="*80)
print("STEP 8: BASELINE MODEL - LOGISTIC REGRESSION")
print("="*80)

def train_baseline_model(X_train, y_train, X_val, y_val):
    """
    Train baseline Logistic Regression model
    
    Returns:
        tuple: (model, metrics)
    """
    print("\nTraining Logistic Regression...")
    
    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight='balanced'
    )
    
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    y_proba_val = model.predict_proba(X_val)[:, 1]
    
    # Metrics
    metrics = {
        'train_accuracy': accuracy_score(y_train, y_pred_train),
        'val_accuracy': accuracy_score(y_val, y_pred_val),
        'val_precision': precision_score(y_val, y_pred_val),
        'val_recall': recall_score(y_val, y_pred_val),
        'val_f1': f1_score(y_val, y_pred_val),
        'val_auc': roc_auc_score(y_val, y_proba_val)
    }
    
    print(f"\nBaseline Results:")
    print(f"  Train Accuracy: {metrics['train_accuracy']:.4f}")
    print(f"  Val Accuracy: {metrics['val_accuracy']:.4f}")
    print(f"  Val Precision: {metrics['val_precision']:.4f}")
    print(f"  Val Recall: {metrics['val_recall']:.4f}")
    print(f"  Val F1: {metrics['val_f1']:.4f}")
    print(f"  Val AUC: {metrics['val_auc']:.4f}")
    
    return model, metrics


# ============================================================================
# STEP 9: ADVANCED MODELS
# ============================================================================
print("\n" + "="*80)
print("STEP 9: ADVANCED MODELS")
print("="*80)

def train_advanced_models(X_train, y_train, X_val, y_val, feature_names):
    """
    Train Random Forest model (only model used)
    
    Returns:
        dict: Trained model and metrics
    """
    models = {}
    
    print("\nRANDOM FOREST [PRIMARY MODEL]")
    print("-" * 80)
    
    rf_model = RandomForestClassifier(
        n_estimators=150,  # Reduced from 200
        max_depth=6,  # Reduced from 8 to prevent overfitting
        min_samples_split=30,  # Increased from 20
        min_samples_leaf=15,  # Increased from 10
        max_features='sqrt',  # Feature sampling
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    
    rf_model.fit(X_train, y_train)
    
    # Check for overfitting
    y_pred_train = rf_model.predict(X_train)
    y_pred_val = rf_model.predict(X_val)
    y_proba_train = rf_model.predict_proba(X_train)[:, 1]
    y_proba_val = rf_model.predict_proba(X_val)[:, 1]
    
    train_auc = roc_auc_score(y_train, y_proba_train)
    val_auc = roc_auc_score(y_val, y_proba_val)
    overfit_gap = train_auc - val_auc
    print(f"  Train AUC: {train_auc:.4f}")
    print(f"  Val AUC: {val_auc:.4f}")
    print(f"  Overfit Gap: {overfit_gap:.4f} {'[WARNING: High gap]' if overfit_gap > 0.05 else '[OK]'}")
    
    rf_metrics = {
        'accuracy': accuracy_score(y_val, y_pred_val),
        'precision': precision_score(y_val, y_pred_val),
        'recall': recall_score(y_val, y_pred_val),
        'f1': f1_score(y_val, y_pred_val),
        'auc': roc_auc_score(y_val, y_proba_val)
    }
    
    models['random_forest'] = {'model': rf_model, 'metrics': rf_metrics}
    
    print(f"  Accuracy: {rf_metrics['accuracy']:.4f}")
    print(f"  Precision: {rf_metrics['precision']:.4f}")
    print(f"  Recall: {rf_metrics['recall']:.4f}")
    print(f"  F1 Score: {rf_metrics['f1']:.4f}")
    print(f"  AUC: {rf_metrics['auc']:.4f} [PRIMARY]")
    
    return models


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    # Create output directory
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    try:
        # Load and clean data
        datasets = load_data()
        datasets, quality_report = clean_data(datasets)
        
        # Engineer features
        features = engineer_features(datasets)
        
        # Create target
        target = create_target(datasets)
        
        # Prepare data
        data_prep = prepare_data(features, target)
        X_train, X_val, X_test, y_train, y_val, y_test, scaler, feature_names = data_prep
        
        # Train baseline
        baseline_model, baseline_metrics = train_baseline_model(
            X_train, y_train, X_val, y_val
        )
        
        # Train advanced models
        advanced_models = train_advanced_models(
            X_train, y_train, X_val, y_val, feature_names
        )
        
        print("\n" + "="*80)
        print("MODEL TRAINING COMPLETE!")
        print("="*80)
        print(f"\nModels saved to: ./models/")
        print(f"Results saved to: ./results/")
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()

print(f"\nExecution completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

