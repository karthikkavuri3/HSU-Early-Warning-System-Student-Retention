"""
Create Web-Optimized Sample Dataset
====================================
Reduces the large dataset to a manageable size for web display
while maintaining referential integrity.

Sample Size: 150 students (from 10,000)
Output: Data_Web/ directory
"""

import pandas as pd
import os
import shutil
from pathlib import Path

print("="*80)
print("CREATING WEB-OPTIMIZED SAMPLE DATASET")
print("="*80)

# Configuration
SAMPLE_SIZE = 150  # Number of students to include
OUTPUT_DIR = 'Data_Web'
SOURCE_DIR = 'Data'

# Create output directory
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR)
print(f"\n✓ Created output directory: {OUTPUT_DIR}/")

# Step 1: Load and sample students
print("\n[1/12] Sampling students...")
students = pd.read_csv(f'{SOURCE_DIR}/students.csv')
print(f"  Original: {len(students):,} students")

# Sample students with stratified sampling by risk level
sample_students = students.groupby('RiskClassification', group_keys=False).apply(
    lambda x: x.sample(n=min(len(x), SAMPLE_SIZE // 4), random_state=42)
).reset_index(drop=True)

# Ensure we have exactly SAMPLE_SIZE students
if len(sample_students) < SAMPLE_SIZE:
    remaining = SAMPLE_SIZE - len(sample_students)
    additional = students[~students['StudentID'].isin(sample_students['StudentID'])].sample(
        n=remaining, random_state=42
    )
    sample_students = pd.concat([sample_students, additional]).reset_index(drop=True)
elif len(sample_students) > SAMPLE_SIZE:
    sample_students = sample_students.sample(n=SAMPLE_SIZE, random_state=42).reset_index(drop=True)

sample_student_ids = sample_students['StudentID'].tolist()
sample_students.to_csv(f'{OUTPUT_DIR}/students.csv', index=False)
print(f"  Sampled: {len(sample_students):,} students")
print(f"  Risk distribution: {sample_students['RiskClassification'].value_counts().to_dict()}")

# Step 2: Filter enrollments
print("\n[2/12] Filtering enrollments...")
enrollments = pd.read_csv(f'{SOURCE_DIR}/enrollments.csv')
print(f"  Original: {len(enrollments):,} enrollments")
sample_enrollments = enrollments[enrollments['StudentID'].isin(sample_student_ids)]
sample_course_ids = sample_enrollments['CourseID'].unique()
sample_enrollments.to_csv(f'{OUTPUT_DIR}/enrollments.csv', index=False)
print(f"  Filtered: {len(sample_enrollments):,} enrollments")

# Step 3: Filter grades (uses EnrollmentID)
print("\n[3/12] Filtering grades...")
grades = pd.read_csv(f'{SOURCE_DIR}/grades.csv')
print(f"  Original: {len(grades):,} grades")
sample_enrollment_ids = sample_enrollments['EnrollmentID'].unique()
sample_grades = grades[grades['EnrollmentID'].isin(sample_enrollment_ids)]
sample_grades.to_csv(f'{OUTPUT_DIR}/grades.csv', index=False)
print(f"  Filtered: {len(sample_grades):,} grades")

# Step 4: Filter attendance (uses EnrollmentID)
print("\n[4/12] Filtering attendance...")
attendance = pd.read_csv(f'{SOURCE_DIR}/attendance.csv')
print(f"  Original: {len(attendance):,} attendance records")
sample_attendance = attendance[attendance['EnrollmentID'].isin(sample_enrollment_ids)]
sample_attendance.to_csv(f'{OUTPUT_DIR}/attendance.csv', index=False)
print(f"  Filtered: {len(sample_attendance):,} attendance records")

# Step 5: Filter logins
print("\n[5/12] Filtering logins...")
logins = pd.read_csv(f'{SOURCE_DIR}/logins.csv')
print(f"  Original: {len(logins):,} login records")
sample_logins = logins[logins['StudentID'].isin(sample_student_ids)]
sample_logins.to_csv(f'{OUTPUT_DIR}/logins.csv', index=False)
print(f"  Filtered: {len(sample_logins):,} login records")

# Step 6: Filter payments
print("\n[6/12] Filtering payments...")
payments = pd.read_csv(f'{SOURCE_DIR}/payments.csv')
print(f"  Original: {len(payments):,} payment records")
sample_payments = payments[payments['StudentID'].isin(sample_student_ids)]
sample_payments.to_csv(f'{OUTPUT_DIR}/payments.csv', index=False)
print(f"  Filtered: {len(sample_payments):,} payment records")

# Step 7: Filter counseling
print("\n[7/12] Filtering counseling...")
counseling = pd.read_csv(f'{SOURCE_DIR}/counseling.csv')
print(f"  Original: {len(counseling):,} counseling visits")
sample_counseling = counseling[counseling['StudentID'].isin(sample_student_ids)]
sample_counseling.to_csv(f'{OUTPUT_DIR}/counseling.csv', index=False)
print(f"  Filtered: {len(sample_counseling):,} counseling visits")

# Step 8: Filter risk scores
print("\n[8/12] Filtering risk scores...")
risk_scores = pd.read_csv(f'{SOURCE_DIR}/risk_scores.csv')
print(f"  Original: {len(risk_scores):,} risk score records")
sample_risk_scores = risk_scores[risk_scores['StudentID'].isin(sample_student_ids)]
sample_risk_scores.to_csv(f'{OUTPUT_DIR}/risk_scores.csv', index=False)
print(f"  Filtered: {len(sample_risk_scores):,} risk score records")

# Step 9: Filter courses (only those enrolled by sampled students)
print("\n[9/12] Filtering courses...")
courses = pd.read_csv(f'{SOURCE_DIR}/courses.csv')
print(f"  Original: {len(courses):,} courses")
sample_courses = courses[courses['CourseID'].isin(sample_course_ids)]
sample_dept_ids = sample_courses['DepartmentID'].unique()
sample_courses.to_csv(f'{OUTPUT_DIR}/courses.csv', index=False)
print(f"  Filtered: {len(sample_courses):,} courses")

# Step 10: Filter departments (only those with courses)
print("\n[10/12] Filtering departments...")
departments = pd.read_csv(f'{SOURCE_DIR}/departments.csv')
print(f"  Original: {len(departments):,} departments")
sample_departments = departments[departments['DepartmentID'].isin(sample_dept_ids)]
sample_departments.to_csv(f'{OUTPUT_DIR}/departments.csv', index=False)
print(f"  Filtered: {len(sample_departments):,} departments")

# Step 11: Filter faculty (only those teaching sampled courses)
print("\n[11/12] Filtering faculty...")
faculty = pd.read_csv(f'{SOURCE_DIR}/faculty.csv')
print(f"  Original: {len(faculty):,} faculty members")
sample_faculty = faculty[faculty['DepartmentID'].isin(sample_dept_ids)]
sample_faculty.to_csv(f'{OUTPUT_DIR}/faculty.csv', index=False)
print(f"  Filtered: {len(sample_faculty):,} faculty members")

# Step 12: Copy terms (keep all terms)
print("\n[12/12] Copying terms...")
terms = pd.read_csv(f'{SOURCE_DIR}/terms.csv')
terms.to_csv(f'{OUTPUT_DIR}/terms.csv', index=False)
print(f"  Copied: {len(terms):,} terms")

# Summary Report
print("\n" + "="*80)
print("DATA SAMPLING COMPLETE!")
print("="*80)

summary = {
    'students': len(sample_students),
    'enrollments': len(sample_enrollments),
    'grades': len(sample_grades),
    'attendance': len(sample_attendance),
    'logins': len(sample_logins),
    'payments': len(sample_payments),
    'counseling': len(sample_counseling),
    'risk_scores': len(sample_risk_scores),
    'courses': len(sample_courses),
    'departments': len(sample_departments),
    'faculty': len(sample_faculty),
    'terms': len(terms)
}

print("\nSAMPLE DATASET SUMMARY:")
print("-" * 80)
for table, count in summary.items():
    print(f"  {table:20s}: {count:>8,} records")

print(f"\nTotal Records: {sum(summary.values()):,}")
print(f"Output Directory: {OUTPUT_DIR}/")

# Calculate reduction percentage
original_students = 10000
reduction = ((original_students - SAMPLE_SIZE) / original_students) * 100
print(f"\nData Reduction: {reduction:.1f}% ({original_students:,} → {SAMPLE_SIZE:,} students)")

# Create a README for the web data
readme_content = f"""# Web-Optimized Sample Dataset

## Overview
This is a sampled subset of the full HSU Early Warning System dataset,
optimized for web display and testing.

## Dataset Size
- Students: {len(sample_students):,} (sampled from 10,000)
- Total Records: {sum(summary.values()):,}

## Risk Distribution
{sample_students['RiskClassification'].value_counts().to_string()}

## Files Included
1. students.csv - Student demographic and academic data
2. enrollments.csv - Course registration records
3. grades.csv - Academic performance data
4. attendance.csv - Class attendance records
5. logins.csv - LMS access logs
6. payments.csv - Financial transaction records
7. counseling.csv - Support services data
8. risk_scores.csv - Risk assessment data
9. courses.csv - Course catalog
10. departments.csv - Academic departments
11. faculty.csv - Faculty information
12. terms.csv - Academic terms

## Usage
Use this dataset for:
- Web dashboard development
- Frontend testing
- Demo purposes
- Quick prototyping

For full ML model training, use the complete dataset in Data/ directory.

## Generation Date
{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

with open(f'{OUTPUT_DIR}/README.txt', 'w') as f:
    f.write(readme_content)

print("\n✓ Created README.txt in Data_Web/")
print("\n" + "="*80)
print("You can now use Data_Web/ for your website!")
print("="*80)
