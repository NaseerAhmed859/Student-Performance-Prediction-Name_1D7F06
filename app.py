"""
app.py
Streamlit dashboard UI for the Student Performance Predictor.
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd

from model import (
    load_data,
    preprocess_data,
    run_model_or_algorithm,
    predict_single,
    grade_category,
    generate_explanation,
    FEATURE_COLUMNS,
)
from visuals import (
    build_gauge,
    scatter_with_trend,
    grade_distribution_bar,
    plot_metric_comparison,
    build_feature_importance_bar,
)

st.set_page_config(page_title="Student Performance Predictor", layout="wide", page_icon="🎓")

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .stApp { background-color: #f4f6fb; }
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #eaeaea;
    }
    .metric-row { display:flex; justify-content: space-between; padding: 10px 0; border-bottom: 1px solid #f0f0f0; }
    .badge-excellent { background:#dcfce7; color:#16a34a; padding:3px 10px; border-radius:8px; font-size:13px; font-weight:600; }
    .badge-good { background:#dbeafe; color:#2563eb; padding:3px 10px; border-radius:8px; font-size:13px; font-weight:600; }
    .badge-average { background:#fef3c7; color:#b45309; padding:3px 10px; border-radius:8px; font-size:13px; font-weight:600; }
    .badge-poor { background:#fee2e2; color:#dc2626; padding:3px 10px; border-radius:8px; font-size:13px; font-weight:600; }
    h1, h2, h3 { color:#111827; }
</style>
""", unsafe_allow_html=True)

BADGE_CLASS = {
    "Excellent": "badge-excellent",
    "Good": "badge-good",
    "Average": "badge-average",
    "Poor": "badge-poor",
}

# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []


@st.cache_resource(show_spinner=False)
def get_trained_models():
    """Load data + train both Random Forest and Linear Regression once."""
    raw_df = load_data("data/students_sample.csv")
    X, y = preprocess_data(raw_df)
    rf_result = run_model_or_algorithm(X, y, {"model_type": "random_forest", "n_estimators": 200})
    lr_result = run_model_or_algorithm(X, y, {"model_type": "linear_regression"})
    return raw_df, rf_result, lr_result


raw_df, rf_result, lr_result = get_trained_models()
primary = rf_result  # Random Forest is the primary model shown on Dashboard

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🎓 Student Performance\nPredictor")
    page = st.radio(
        "Navigate",
        ["🏠 Dashboard", "📈 Prediction History", "🔁 Compare Models", "🥧 Data Insights", "ℹ️ Model Info", "📄 About"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption(
        "This app predicts a student's final grade based on study habits, "
        "attendance, GPA and assignment scores using Machine Learning."
    )

# ---------------------------------------------------------------------------
# DASHBOARD PAGE
# ---------------------------------------------------------------------------
if page == "🏠 Dashboard":
    st.title("Student Performance Predictor")
    st.caption("Predict student performance and explore insights")

    col_left, col_right = st.columns([1, 1])

    with col_left:
        with st.container(border=True):
            st.subheader("📝 Enter Student Details")
            hours = st.number_input("Hours Studied per Week", 0.0, 70.0, 5.0, 0.5)
            attendance = st.number_input("Attendance (%)", 0.0, 100.0, 92.0, 1.0)
            gpa = st.number_input("Previous GPA (out of 4.0)", 0.0, 4.0, 3.6, 0.1)
            assignment = st.number_input("Assignment Score (%)", 0.0, 100.0, 90.0, 1.0)

            predict_clicked = st.button("🚀 Predict Performance", type="primary", use_container_width=True)
            reset_clicked = st.button("↻ Reset", use_container_width=True)

            if reset_clicked:
                st.rerun()

    with col_right:
        with st.container(border=True):
            st.subheader("🎯 Predicted Performance")
            if predict_clicked:
                student_input = {
                    "hours_studied_per_week": hours,
                    "attendance": attendance,
                    "gpa": gpa,
                    "assignment_score": assignment,
                }
                pred_score = predict_single(primary["model"], student_input)
                category = grade_category(pred_score)

                st.plotly_chart(build_gauge(pred_score, category), use_container_width=True)
                st.markdown(
                    f"<h3 style='text-align:center;'>{category}</h3>",
                    unsafe_allow_html=True,
                )

                messages = {
                    "Excellent": "Great job! The student is expected to perform excellent. Keep up the good work!",
                    "Good": "Solid performance expected. A bit more consistency could push this even higher.",
                    "Average": "This student is on track but may benefit from extra support or study time.",
                    "Poor": "This student may be at risk — consider additional support or intervention.",
                }
                st.info(f"🏆 {messages[category]}")

                st.session_state.history.insert(0, {
                    "Hours Studied": hours,
                    "Attendance (%)": attendance,
                    "GPA": gpa,
                    "Assignment Score (%)": assignment,
                    "Predicted Grade (%)": round(pred_score, 1),
                    "Performance": category,
                })
                st.session_state.history = st.session_state.history[:20]
                st.session_state["last_prediction"] = (student_input, pred_score)
            else:
                st.plotly_chart(build_gauge(0, "Average"), use_container_width=True)
                st.caption("Enter details on the left and click **Predict Performance** to see a result.")

    # Explanation + Evaluation Metrics row
    col_a, col_b = st.columns([1, 1])
    with col_a:
        with st.container(border=True):
            st.subheader("💡 Explanation")
            if "last_prediction" in st.session_state:
                student_input, pred_score = st.session_state["last_prediction"]
                points = generate_explanation(student_input, pred_score)
                for is_positive, text in points:
                    icon = "✅" if is_positive else "⚠️"
                    st.write(f"{icon} {text}")
                category = grade_category(pred_score)
                st.markdown(f"**Overall Prediction:** <span class='{BADGE_CLASS[category]}'>{category} Performance</span>", unsafe_allow_html=True)
            else:
                st.caption("Run a prediction to see an explanation here.")

    with col_b:
        with st.container(border=True):
            st.subheader("📊 Evaluation Metrics")
            m = primary["metrics"]
            st.markdown(f"<div class='metric-row'><span>R² Score</span><b>{m['r2']:.2f}</b></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric-row'><span>Mean Absolute Error (MAE)</span><b>{m['mae']:.2f}</b></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric-row'><span>Mean Squared Error (MSE)</span><b>{m['mse']:.2f}</b></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric-row'><span>Root Mean Squared Error (RMSE)</span><b>{m['rmse']:.2f}</b></div>", unsafe_allow_html=True)

    # Data Insights & Visualizations
    with st.container(border=True):
        st.subheader("🥧 Data Insights & Visualizations")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.plotly_chart(
                scatter_with_trend(raw_df, "attendance", "final_grade", "Attendance vs Final Grade",
                                    "Attendance (%)", "Final Grade (%)", color="#3b82f6"),
                use_container_width=True,
            )
        with c2:
            st.plotly_chart(
                scatter_with_trend(raw_df, "hours_studied_per_week", "final_grade", "Hours Studied vs Final Grade",
                                    "Hours Studied per Week", "Final Grade (%)", color="#22c55e"),
                use_container_width=True,
            )
        with c3:
            st.plotly_chart(grade_distribution_bar(raw_df), use_container_width=True)

    # Recent Predictions table
    with st.container(border=True):
        st.subheader("🕒 Recent Predictions")
        if st.session_state.history:
            hist_df = pd.DataFrame(st.session_state.history[:5])
            st.dataframe(hist_df, use_container_width=True, hide_index=True)
            if len(st.session_state.history) > 5:
                st.caption("See **Prediction History** in the sidebar for the full list.")
        else:
            st.caption("No predictions yet — run one above to see it appear here.")

# ---------------------------------------------------------------------------
# PREDICTION HISTORY PAGE
# ---------------------------------------------------------------------------
elif page == "📈 Prediction History":
    st.title("Prediction History")
    if st.session_state.history:
        st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True, hide_index=True)
        if st.button("🗑️ Clear History"):
            st.session_state.history = []
            st.rerun()
    else:
        st.info("No predictions yet. Go to the Dashboard to run one.")

# ---------------------------------------------------------------------------
# COMPARE MODELS PAGE
# ---------------------------------------------------------------------------
elif page == "🔁 Compare Models":
    st.title("Compare Models")
    st.caption("Random Forest vs Linear Regression, trained on the same data split.")

    st.plotly_chart(plot_metric_comparison(rf_result["metrics"], lr_result["metrics"],
                                            "Random Forest", "Linear Regression"),
                     use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.markdown("**Random Forest**")
            for k, v in rf_result["metrics"].items():
                st.write(f"{k.upper()}: **{v:.3f}**")
    with col2:
        with st.container(border=True):
            st.markdown("**Linear Regression**")
            for k, v in lr_result["metrics"].items():
                st.write(f"{k.upper()}: **{v:.3f}**")

# ---------------------------------------------------------------------------
# DATA INSIGHTS PAGE
# ---------------------------------------------------------------------------
elif page == "🥧 Data Insights":
    st.title("Data Insights")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            scatter_with_trend(raw_df, "attendance", "final_grade", "Attendance vs Final Grade",
                                "Attendance (%)", "Final Grade (%)", color="#3b82f6"),
            use_container_width=True,
        )
    with c2:
        st.plotly_chart(
            scatter_with_trend(raw_df, "hours_studied_per_week", "final_grade", "Hours Studied vs Final Grade",
                                "Hours Studied per Week", "Final Grade (%)", color="#22c55e"),
            use_container_width=True,
        )
    st.plotly_chart(grade_distribution_bar(raw_df), use_container_width=True)
    st.plotly_chart(build_feature_importance_bar(primary["feature_importances"]), use_container_width=True)
    with st.expander("Preview raw dataset"):
        st.dataframe(raw_df, use_container_width=True)

# ---------------------------------------------------------------------------
# MODEL INFO PAGE
# ---------------------------------------------------------------------------
elif page == "ℹ️ Model Info":
    st.title("Model Info")
    st.write("**Primary model:** Random Forest Regressor (200 trees)")
    st.write(f"**Features used:** {', '.join(FEATURE_COLUMNS)}")
    st.write("**Target:** final_grade (0-100)")
    st.write(generate_explanation(feature_importances=primary["feature_importances"]))
    st.plotly_chart(build_feature_importance_bar(primary["feature_importances"]), use_container_width=True)
    st.subheader("Metrics")
    st.json({k: round(v, 3) for k, v in primary["metrics"].items()})

# ---------------------------------------------------------------------------
# ABOUT PAGE
# ---------------------------------------------------------------------------
else:
    st.title("About")
    st.write(
        "This app predicts a student's final performance based on study habits, "
        "attendance, GPA and assignment scores using Machine Learning."
    )
    st.write("Built with **Python**, **Streamlit**, **scikit-learn**, and **Plotly**.")
    st.caption("AI-Lab Project")
