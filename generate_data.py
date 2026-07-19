"""
generate_data.py
Generates a synthetic student performance dataset (regression target)
for the Student Performance Predictor lab project.

Run once to (re)create data/students_sample.csv
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N = 300

hours_studied_per_week = np.round(np.random.uniform(0, 10, N), 1)
attendance = np.round(np.random.uniform(40, 100, N), 1)
gpa = np.round(np.random.uniform(1.5, 4.0, N), 2)
assignment_score = np.round(np.random.uniform(30, 100, N), 1)

final_grade = (
    0.30 * (hours_studied_per_week / 10) * 100
    + 0.30 * attendance
    + 0.20 * (gpa / 4.0) * 100
    + 0.20 * assignment_score
)
final_grade += np.random.normal(0, 4, N)
final_grade = np.clip(final_grade, 0, 100).round(1)

df = pd.DataFrame({
    "hours_studied_per_week": hours_studied_per_week,
    "attendance": attendance,
    "gpa": gpa,
    "assignment_score": assignment_score,
    "final_grade": final_grade
})

df.to_csv("data/students_sample.csv", index=False)
print("Saved data/students_sample.csv with", len(df), "rows")
print(df.describe())
