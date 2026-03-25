import pandas as pd
import numpy as np

print("="*60)
print("STUDENT PERFORMANCE DATASET ANALYSIS")
print("="*60)

df = pd.read_csv('Student_Performance.csv')

print(f"\n1. DATASET SIZE: {len(df)} students")

# Check for missing values
print(f"\n2. MISSING VALUES:")
print(df.isnull().sum())

# Grade distribution
print(f"\n3. GRADE DISTRIBUTION:")
print(df['final_grade'].value_counts().sort_index())

# Pass/Fail balance
grade_map = {
    'a': 1, 'b': 1, 'c': 0, 'd': 0, 'e': 0, 'f': 0,
    'A': 1, 'B': 1, 'C': 0, 'D': 0, 'E': 0, 'F': 0
}
df['pass_fail'] = df['final_grade'].str.strip().map(grade_map)
df = df.dropna(subset=['pass_fail'])

pass_count = df['pass_fail'].sum()
fail_count = len(df) - pass_count
pass_rate = df['pass_fail'].mean() * 100

print(f"\n4. PASS/FAIL BALANCE:")
print(f"   Pass: {pass_count} ({pass_rate:.1f}%)")
print(f"   Fail: {fail_count} ({100-pass_rate:.1f}%)")

if pass_rate < 30 or pass_rate > 70:
    print(f"   ⚠️ WARNING: Imbalanced! (Should be 40-60%)")
else:
    print(f"   ✓ GOOD: Balanced dataset")

# Score statistics
print(f"\n5. SCORE STATISTICS:")
print(f"   Math Score - Mean: {df['math_score'].mean():.1f}, Std: {df['math_score'].std():.1f}")
print(f"   Science Score - Mean: {df['science_score'].mean():.1f}, Std: {df['science_score'].std():.1f}")
print(f"   English Score - Mean: {df['english_score'].mean():.1f}, Std: {df['english_score'].std():.1f}")
print(f"   Overall Score - Mean: {df['overall_score'].mean():.1f}, Std: {df['overall_score'].std():.1f}")

# Study hours and attendance
print(f"\n6. STUDY & ATTENDANCE:")
print(f"   Study Hours - Mean: {df['study_hours'].mean():.1f}, Std: {df['study_hours'].std():.1f}")
print(f"   Attendance % - Mean: {df['attendance_percentage'].mean():.1f}, Std: {df['attendance_percentage'].std():.1f}")

# Correlation analysis
print(f"\n7. CORRELATION WITH PASS/FAIL:")
correlations = df[['study_hours', 'attendance_percentage', 'math_score', 
                    'science_score', 'english_score', 'overall_score', 'pass_fail']].corr()['pass_fail'].sort_values(ascending=False)
print(correlations)

if correlations['overall_score'] < 0.3:
    print(f"\n   ⚠️ WARNING: Weak correlation between scores and pass/fail!")
    print(f"   This means scores don't predict grades well.")
else:
    print(f"\n   ✓ GOOD: Scores correlate with pass/fail")

# Check for duplicates
duplicates = df.duplicated().sum()
print(f"\n8. DUPLICATE ROWS: {duplicates}")
if duplicates > 100:
    print(f"   ⚠️ WARNING: Many duplicates detected!")

# Check data variance
print(f"\n9. DATA VARIANCE CHECK:")
if df['math_score'].std() < 5:
    print(f"   ⚠️ WARNING: Math scores have very low variance (all similar)")
if df['study_hours'].std() < 1:
    print(f"   ⚠️ WARNING: Study hours have very low variance (all similar)")
if df['attendance_percentage'].std() < 5:
    print(f"   ⚠️ WARNING: Attendance has very low variance (all similar)")

# Sample data
print(f"\n10. SAMPLE DATA (First 5 Pass and First 5 Fail):")
print("\nPassing Students:")
print(df[df['pass_fail']==1][['study_hours', 'attendance_percentage', 'math_score', 
                                'science_score', 'english_score', 'overall_score', 'final_grade']].head())
print("\nFailing Students:")
print(df[df['pass_fail']==0][['study_hours', 'attendance_percentage', 'math_score', 
                                'science_score', 'english_score', 'overall_score', 'final_grade']].head())

print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)