# Student Performance Predictor

An AI-powered Streamlit dashboard that predicts a student's **final grade (%)**
from their study habits, attendance, GPA, and assignment score — and explains
*why* it made that prediction.

Built for the AI Lab Project (UI + AI Integration).

---

## 1. Problem

Educators often want an early signal of how a student is likely to perform,
using a few easily collected numbers: weekly study hours, attendance,
previous GPA, and assignment scores. This app trains a regression model on
sample student data and lets a user instantly predict an individual
student's expected final grade by entering their details.

## 2. AI Used

- **Random Forest Regressor** (primary model) — an ensemble of decision
  trees that handles non-linear relationships and gives feature importance
  for free.
- **Linear Regression** (comparison baseline) — trained automatically
  alongside the Random Forest so the two approaches can be compared
  side by side on the **Compare Models** page.

## 3. How It Works (Important — No File Upload)

This app does **not** ask the user to upload a CSV. Instead:

1. The app loads and trains on the built-in sample dataset
   (`data/students_sample.csv`) automatically in the background the moment
   it starts.
2. The user directly types/enters a single student's details into four
   input fields on the Dashboard:
   - Hours Studied per Week
   - Attendance (%)
   - Previous GPA (out of 4.0)
   - Assignment Score (%)
3. Clicking **Predict Performance** feeds those values into the trained
   model and instantly shows the result — a predicted grade %, a
   performance category, and a plain-language explanation.

There is no dataset upload step for the end user; the training dataset is
fixed and bundled with the app, and the only user input is the one
student's details used for a live prediction.

## 4. Features

- **Dashboard** — main screen: student input form, prediction gauge,
  explanation checklist, evaluation metrics, insight charts, and a table
  of recent predictions
- **Prediction History** — full log of every prediction made this session
- **Compare Models** — Random Forest vs Linear Regression metrics and chart
- **Data Insights** — scatter plots (Attendance vs Grade, Hours vs Grade)
  with trend lines, grade distribution chart, and feature importance
- **Model Info** — model type, features used, and evaluation metrics
- **About** — project summary

## 5. Setup & Run Instructions

### Requirements
- Python 3.9+ 
### Create a virtual environment (recommended)
Open the integrated terminal: Terminal → New Terminal, then run:

python -m venv venv

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run the app
```bash
streamlit run app.py
```

Then open the local URL Streamlit prints (usually `http://localhost:8501`).

### Using the app
1. Open the app — it trains automatically on launch, no action needed.
2. On the **Dashboard**, enter a student's Hours Studied, Attendance, GPA,
   and Assignment Score.
3. Click **Predict Performance** to see the predicted grade, category, and
   explanation.
4. Explore **Compare Models**, **Data Insights**, and **Model Info** from
   the sidebar for deeper analysis.

### (Optional) Regenerate the sample dataset
```bash
python generate_data.py
```
This overwrites `data/students_sample.csv` with a fresh synthetic dataset
used to train the model. This is a developer/instructor step, not something
the end user does through the app UI.

## 6. Dataset

`data/students_sample.csv` — 300 synthetic student records used to train
the model, with columns:

| Column | Description |
|---|---|
| hours_studied_per_week | Weekly study hours (0-10) |
| attendance | Attendance percentage (40-100) |
| gpa | Previous GPA out of 4.0 (1.5-4.0) |
| assignment_score | Assignment score percentage (30-100) |
| final_grade | Target: final grade percentage (0-100) |

This dataset is used only to train the model at startup — it is **not**
uploaded or edited by the end user through the app.

## 7. Project Structure

```
StudentPerformancePredictor/
├── app.py              # Streamlit dashboard UI (sidebar pages, input form, gauge)
├── model.py             # load_data, preprocess_data, run_model_or_algorithm,
│                         # predict_single, grade_category, generate_explanation
├── visuals.py            # Plotly charts: gauge, scatter+trend, distribution,
│                         # model comparison, feature importance
├── generate_data.py       # Script used to (re)create the sample dataset
├── requirements.txt
├── README.md
├── data/
│   └── students_sample.csv
└── screenshots/           # Adding Our Project UI screenshots 
```

## 8. Evaluation Approach

The app trains both a Random Forest Regressor and a Linear Regression model
on the same train/test split and displays R², MAE, MSE, and RMSE for the
primary model, with a side-by-side comparison chart against the baseline on
the **Compare Models** page — satisfying the "compare at least two
approaches" requirement.

## 9. Limitations & Future Improvements

- Dataset is synthetic; real-world student data would improve validity
- No cross-validation (single train/test split)
- No option for the user to bring their own dataset — model always trains
  on the bundled sample data
- Could add SHAP values for more precise per-prediction feature attribution
- Could add authentication/persistence so prediction history survives
  across sessions instead of resetting on refresh

