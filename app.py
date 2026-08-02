"""
NeuroSync AI - Intelligent Lifestyle, Burnout & Productivity Analytics Platform
--------------------------------------------------------------------------------
Main Streamlit application entry point.

Run:
    streamlit run app.py
"""
from dotenv import load_dotenv
load_dotenv()

import os
import time
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from utils.styling import inject_theme, SPLASH_HTML, glass_card_open, glass_card_close, risk_badge, sidebar_nav
from predict import (
    load_artifacts, list_available_models, load_model, predict, RAW_INPUT_COLUMNS
)

st.set_page_config(
    page_title="NeuroSync AI",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_theme()

DATA_PATH = "dataset/neurosync_dataset.csv"
COMPARISON_PATH = "reports/model_comparison_table.csv"


# ---------------------------------------------------------------------------
# Splash screen (shown once per session)
# ---------------------------------------------------------------------------
if "splash_done" not in st.session_state:
    st.session_state.splash_done = False

if not st.session_state.splash_done:
    placeholder = st.empty()
    with placeholder.container():
        st.markdown(SPLASH_HTML, unsafe_allow_html=True)
    time.sleep(1.6)
    st.session_state.splash_done = True
    placeholder.empty()
    st.rerun()


# ---------------------------------------------------------------------------
# Cached data / artifact loaders
# ---------------------------------------------------------------------------
@st.cache_data
def get_dataset():
    if not os.path.exists(DATA_PATH):
        return None
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def get_artifacts():
    try:
        return load_artifacts()
    except FileNotFoundError:
        return None, None, None


@st.cache_resource
def get_model(path, kind):
    return load_model(path, kind)


df = get_dataset()
scaler, target_encoder, selected_features = get_artifacts()
available_models = list_available_models() if scaler is not None else {}


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
with st.sidebar:
    page = sidebar_nav()

    if available_models:
        st.sidebar.caption(f"{len(available_models)} model(s) loaded")
    else:
        st.sidebar.caption("No trained models found yet")


# ---------------------------------------------------------------------------
# PAGE: Home
# ---------------------------------------------------------------------------
if page == "Home":
    st.markdown('<div class="hero-title">NeuroSync AI</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-subtitle">Intelligent Lifestyle, Burnout & Productivity '
        'Analytics Platform</div>', unsafe_allow_html=True,
    )

    cols = st.columns(4)
    kpis = [
        ("Rows in Dataset", f"{len(df):,}" if df is not None else "—"),
        ("Features Tracked", "21" if df is not None else "—"),
        ("Models Trained", str(len(available_models))),
        ("Target Classes", "3 (Low / Medium / High)"),
    ]
    for c, (label, value) in zip(cols, kpis):
        with c:
            glass_card_open()
            st.markdown(f'<div class="kpi-value">{value}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="kpi-label">{label}</div>', unsafe_allow_html=True)
            glass_card_close()

    st.markdown('<div class="section-title">What NeuroSync AI Does</div>', unsafe_allow_html=True)
    glass_card_open()
    st.write(
        "NeuroSync AI analyzes lifestyle and behavioral data — sleep, work hours, "
        "screen time, exercise, stress, and mood — to predict burnout risk, wellness "
        "score, and productivity using a suite of Machine Learning and Deep Learning "
        "models. Use the sidebar to explore the dataset, compare model performance, "
        "or run a live prediction for yourself or a batch of users."
    )
    glass_card_close()


# ---------------------------------------------------------------------------
# PAGE: Dashboard Overview
# ---------------------------------------------------------------------------
elif page == "Dashboard Overview":
    st.markdown('<div class="section-title">Dashboard Overview</div>', unsafe_allow_html=True)

    if df is None:
        st.warning("Dataset not found at dataset/neurosync_dataset.csv. Run generate_dataset.py first.")
    else:
        cols = st.columns(4)
        metrics = [
            ("Avg Stress Score", f"{df['Stress_Score'].mean():.1f}"),
            ("Avg Sleep Hours", f"{df['Sleep_Hours'].mean():.1f} hrs"),
            ("Avg Wellness Score", f"{df['Wellness_Score'].mean():.1f}"),
            ("Avg Productivity", f"{df['Productivity_Score'].mean():.1f}"),
        ]
        for c, (label, value) in zip(cols, metrics):
            c.metric(label, value)

        col1, col2 = st.columns(2)
        with col1:
            glass_card_open()
            fig = px.pie(
                df, names="Burnout_Risk", title="Burnout Risk Distribution",
                color="Burnout_Risk",
                color_discrete_map={"Low": "#C4B0F5", "Medium": "#7C3AED", "High": "#D0286B"},
                hole=0.5,
            )
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#1F1147")
            st.plotly_chart(fig, use_container_width=True)
            glass_card_close()

        with col2:
            glass_card_open()
            fig = px.box(
                df, x="Burnout_Risk", y="Stress_Score", color="Burnout_Risk",
                title="Stress Score by Burnout Risk",
                color_discrete_map={"Low": "#C4B0F5", "Medium": "#7C3AED", "High": "#D0286B"},
            )
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#1F1147")
            st.plotly_chart(fig, use_container_width=True)
            glass_card_close()

        glass_card_open()
        occ_avg = df.groupby("Occupation")[["Stress_Score", "Wellness_Score", "Productivity_Score"]].mean().reset_index()
        fig = px.bar(
            occ_avg, x="Occupation", y=["Stress_Score", "Wellness_Score", "Productivity_Score"],
            barmode="group", title="Average Scores by Occupation",
        )
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#1F1147")
        st.plotly_chart(fig, use_container_width=True)
        glass_card_close()


# ---------------------------------------------------------------------------
# PAGE: Dataset Viewer
# ---------------------------------------------------------------------------
elif page == "Dataset Viewer":
    st.markdown('<div class="section-title">Dataset Viewer</div>', unsafe_allow_html=True)

    if df is None:
        st.warning("Dataset not found. Run dataset/generate_dataset.py first.")
    else:
        col1, col2, col3 = st.columns(3)
        occ_filter = col1.multiselect("Occupation", sorted(df["Occupation"].unique()))
        risk_filter = col2.multiselect("Burnout Risk", sorted(df["Burnout_Risk"].unique()))
        gender_filter = col3.multiselect("Gender", sorted(df["Gender"].unique()))

        filtered = df.copy()
        if occ_filter:
            filtered = filtered[filtered["Occupation"].isin(occ_filter)]
        if risk_filter:
            filtered = filtered[filtered["Burnout_Risk"].isin(risk_filter)]
        if gender_filter:
            filtered = filtered[filtered["Gender"].isin(gender_filter)]

        st.caption(f"Showing {len(filtered):,} of {len(df):,} rows")
        st.dataframe(filtered.head(1000), use_container_width=True, height=480)

        glass_card_open()
        st.write("Summary Statistics")
        st.dataframe(filtered.describe().round(2), use_container_width=True)
        glass_card_close()


# ---------------------------------------------------------------------------
# PAGE: Exploratory Data Analysis
# ---------------------------------------------------------------------------
elif page == "Exploratory Data Analysis":
    st.markdown('<div class="section-title">Exploratory Data Analysis</div>', unsafe_allow_html=True)

    if df is None:
        st.warning("Dataset not found. Run dataset/generate_dataset.py first.")
    else:
        numeric_df = df.select_dtypes(include=np.number)

        col1, col2 = st.columns(2)
        with col1:
            feature = st.selectbox("Feature for distribution", numeric_df.columns, index=list(numeric_df.columns).index("Stress_Score"))
            glass_card_open()
            fig = px.histogram(df, x=feature, color="Burnout_Risk", barmode="overlay", nbins=40,
                                color_discrete_map={"Low": "#C4B0F5", "Medium": "#7C3AED", "High": "#D0286B"})
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#1F1147")
            st.plotly_chart(fig, use_container_width=True)
            glass_card_close()

        with col2:
            glass_card_open()
            fig = px.box(df, x="Burnout_Risk", y=feature, color="Burnout_Risk",
                         color_discrete_map={"Low": "#C4B0F5", "Medium": "#7C3AED", "High": "#D0286B"})
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#1F1147")
            st.plotly_chart(fig, use_container_width=True)
            glass_card_close()

        glass_card_open()
        st.write("Correlation Heatmap")
        corr = numeric_df.corr()
        fig = px.imshow(corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#1F1147")
        st.plotly_chart(fig, use_container_width=True)
        glass_card_close()


# ---------------------------------------------------------------------------
# PAGE: Model Comparison
# ---------------------------------------------------------------------------
elif page == "Model Comparison":
    st.markdown('<div class="section-title">Model Comparison</div>', unsafe_allow_html=True)

    if os.path.exists(COMPARISON_PATH):
        comparison = pd.read_csv(COMPARISON_PATH)
        st.dataframe(comparison, use_container_width=True)

        glass_card_open()
        fig = px.bar(comparison.sort_values("Accuracy"), x="Accuracy", y="Model", orientation="h",
                     title="Model Accuracy Comparison", color="Accuracy", color_continuous_scale="Purples")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#1F1147")
        st.plotly_chart(fig, use_container_width=True)
        glass_card_close()

        best = comparison.sort_values("Accuracy", ascending=False).iloc[0]
        st.success(f"Best performing model: **{best['Model']}** (Accuracy: {best['Accuracy']:.4f})")
    elif available_models:
        st.info(
            "reports/model_comparison_table.csv not found yet (train.py may not have "
            "completed all models, e.g. Deep Learning models need TensorFlow). "
            "Showing a live test-set evaluation for the models currently saved instead "
            "(first load may take a minute; results are cached after that)."
        )
        if not os.path.exists("dataset/processed/X_test.csv"):
            st.warning("dataset/processed/X_test.csv not found. Run preprocessing.py first.")
        else:
            @st.cache_data(show_spinner="Evaluating saved models on the test set...")
            def compute_live_comparison(model_items):
                from evaluation import evaluate_model as _eval_model

                X_test = pd.read_csv("dataset/processed/X_test.csv")
                y_test = pd.read_csv("dataset/processed/y_test.csv").squeeze("columns")

                rows = []
                for name, path, kind in model_items:
                    model = load_model(path, kind)
                    if kind == "keras":
                        proba = model.predict(X_test.to_numpy(dtype="float32"), verbose=0)
                        y_pred = np.argmax(proba, axis=1)
                    else:
                        y_pred = model.predict(X_test)
                        proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
                    rows.append(_eval_model(name, y_test, y_pred, proba))

                return pd.DataFrame(rows).sort_values("Accuracy", ascending=False).reset_index(drop=True)

            model_items = tuple((name, path, kind) for name, (path, kind) in available_models.items())
            live_comparison = compute_live_comparison(model_items)
            st.dataframe(live_comparison, use_container_width=True)

            glass_card_open()
            fig = px.bar(live_comparison.sort_values("Accuracy"), x="Accuracy", y="Model", orientation="h",
                         title="Model Accuracy Comparison (live)", color="Accuracy", color_continuous_scale="Purples")
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#1F1147")
            st.plotly_chart(fig, use_container_width=True)
            glass_card_close()
    else:
        st.warning("No trained models found. Run train.py first.")


# ---------------------------------------------------------------------------
# PAGE: Prediction (single input)
# ---------------------------------------------------------------------------
elif page == "Prediction":
    st.markdown('<div class="section-title">Burnout Risk Prediction</div>', unsafe_allow_html=True)

    if not available_models:
        st.warning("No trained models found. Run train.py first.")
    else:
        model_name = st.selectbox("Choose a model", list(available_models.keys()))

        with st.form("prediction_form"):
            c1, c2, c3 = st.columns(3)
            age = c1.number_input("Age", 18, 65, 28)
            gender = c2.selectbox("Gender", ["Male", "Female", "Other"])
            occupation = c3.selectbox(
                "Occupation",
                ["IT/Software", "Healthcare", "Education", "Business/Finance", "Student", "Freelancer", "Other"],
            )

            c1, c2, c3 = st.columns(3)
            work_hours = c1.slider("Work Hours / day", 2.0, 16.0, 9.0)
            sleep_hours = c2.slider("Sleep Hours / night", 3.0, 10.0, 6.5)
            screen_time = c3.slider("Screen Time / day", 1.0, 16.0, 7.0)

            c1, c2, c3 = st.columns(3)
            exercise_frequency = c1.slider("Exercise Frequency / week", 0, 7, 2)
            daily_steps = c2.number_input("Daily Steps", 500, 20000, 5000, step=500)
            water_intake = c3.slider("Water Intake (L)", 0.3, 5.0, 1.8)

            c1, c2, c3 = st.columns(3)
            caffeine_intake = c1.slider("Caffeine Intake (cups)", 0, 10, 3)
            bmi = c2.slider("BMI", 15.0, 45.0, 24.0)
            heart_rate = c3.slider("Resting Heart Rate", 50, 130, 78)

            c1, c2 = st.columns(2)
            bp_systolic = c1.slider("BP Systolic", 90, 180, 120)
            bp_diastolic = c2.slider("BP Diastolic", 55, 115, 78)

            c1, c2, c3 = st.columns(3)
            stress_score = c1.slider("Stress Score", 0, 100, 55)
            mood_score = c2.slider("Mood Score", 0, 100, 55)
            social_hours = c3.slider("Social Interaction Hours", 0.0, 10.0, 3.0)

            c1, c2, c3 = st.columns(3)
            meditation_minutes = c1.slider("Meditation Minutes", 0, 90, 10)
            weekend_rest = c2.slider("Weekend Rest Hours", 2.0, 16.0, 9.0)
            productivity_score = c3.slider("Productivity Score", 0, 100, 60)

            wellness_score = st.slider("Wellness Score", 0, 100, 60)

            submitted = st.form_submit_button("Predict Burnout Risk")

        if submitted:
            raw = pd.DataFrame([{
                "Age": age, "Gender": gender, "Occupation": occupation,
                "Work_Hours": work_hours, "Sleep_Hours": sleep_hours, "Screen_Time": screen_time,
                "Exercise_Frequency": exercise_frequency, "Daily_Steps": daily_steps,
                "Water_Intake": water_intake, "Caffeine_Intake": caffeine_intake,
                "BMI": bmi, "Heart_Rate": heart_rate, "BP_Systolic": bp_systolic,
                "BP_Diastolic": bp_diastolic, "Stress_Score": stress_score, "Mood_Score": mood_score,
                "Social_Interaction_Hours": social_hours, "Meditation_Minutes": meditation_minutes,
                "Weekend_Rest_Hours": weekend_rest, "Productivity_Score": productivity_score,
                "Wellness_Score": wellness_score,
            }])

            path, kind = available_models[model_name]
            model = get_model(path, kind)
            result = predict(raw, model, kind, scaler, target_encoder, selected_features)
            label = result["Burnout_Risk_Prediction"].iloc[0]

            glass_card_open()
            st.markdown(f"### Prediction: {risk_badge(label)}", unsafe_allow_html=True)
            prob_cols = [c for c in result.columns if c.startswith("Probability_")]
            if prob_cols:
                probs = result[prob_cols].iloc[0]
                fig = go.Figure(go.Bar(
                    x=[c.replace("Probability_", "") for c in prob_cols],
                    y=probs.values, marker_color=["#C4B0F5", "#7C3AED", "#D0286B"],
                ))
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#1F1147", title="Class Probabilities", yaxis_title="Probability",
                )
                st.plotly_chart(fig, use_container_width=True)
            glass_card_close()

            if label == "High":
                st.info(
                    "Your burnout risk is high. Contributing factors likely include low "
                    "sleep, high work hours/screen time, or elevated stress. Consider "
                    "improving sleep, reducing screen time, and adding regular exercise."
                )


# ---------------------------------------------------------------------------
# PAGE: Batch CSV Prediction
# ---------------------------------------------------------------------------
elif page == "Batch CSV Prediction":
    st.markdown('<div class="section-title">Batch CSV Prediction</div>', unsafe_allow_html=True)

    if not available_models:
        st.warning("No trained models found. Run train.py first.")
    else:
        model_name = st.selectbox("Choose a model", list(available_models.keys()), key="batch_model")
        st.caption("CSV must contain columns: " + ", ".join(RAW_INPUT_COLUMNS))

        uploaded = st.file_uploader("Upload CSV file", type=["csv"])
        if uploaded is not None:
            batch_df = pd.read_csv(uploaded)
            st.dataframe(batch_df.head(10), use_container_width=True)

            if st.button("Run Batch Prediction"):
                path, kind = available_models[model_name]
                model = get_model(path, kind)
                try:
                    result = predict(batch_df, model, kind, scaler, target_encoder, selected_features)
                    combined = pd.concat([batch_df.reset_index(drop=True), result], axis=1)
                    st.dataframe(combined, use_container_width=True)

                    csv_bytes = combined.to_csv(index=False).encode("utf-8")
                    st.download_button("Download Predictions CSV", csv_bytes, "neurosync_predictions.csv", "text/csv")
                except ValueError as e:
                    st.error(str(e))


# ---------------------------------------------------------------------------

# PAGE: AI Wellness Assistant
# ---------------------------------------------------------------------------
elif page == "AI Wellness Assistant":
    st.markdown('<div class="section-title">AI Wellness Assistant</div>', unsafe_allow_html=True)
    glass_card_open()

    from ai_assistant import (
        ask,
        gemini_available,
        openai_available,
        huggingface_available,
    )

    provider_options = []

    if gemini_available():
        provider_options.append("Gemini")

    if openai_available():
        provider_options.append("OpenAI")

    if huggingface_available():
        provider_options.append("Hugging Face")

    if not provider_options:
        st.warning(
            "No API key found. Add your API key(s) in the .env file."
        )
    else:
        provider = st.selectbox("AI Provider", provider_options)

        question = st.text_area(
            "Ask about your wellness, or describe your recent lifestyle",
            placeholder="e.g. I've been sleeping 5 hours and working 11 hour days, what should I change?",
        )

        if st.button("Get AI Guidance"):
            if not question.strip():
                st.error("Please enter a question or description.")
            else:
                try:
                    with st.spinner(f"Asking {provider}..."):
                        answer = ask(provider, question)
                    st.markdown(answer)
                except Exception as e:
                    st.error(f"Error calling {provider} API: {e}")

    glass_card_close()

# ---------------------------------------------------------------------------
# PAGE: About Project
# ---------------------------------------------------------------------------

elif page == "About Project":
    st.markdown('<div class="section-title">About NeuroSync AI</div>', unsafe_allow_html=True)
    glass_card_open()

    st.markdown("""
    
### 🧠 Project Overview

**NeuroSync AI** is an AI-powered Burnout Risk Prediction and Wellness Analytics platform developed to identify individuals who may be experiencing stress, fatigue, or burnout based on their daily lifestyle and work habits.

In today's fast-paced world, students and professionals often face long working hours, excessive screen time, lack of sleep, and high stress levels. These factors gradually reduce productivity, negatively impact mental health, and increase the risk of burnout. Unfortunately, many people recognize these warning signs only when the situation becomes serious.

NeuroSync AI addresses this challenge by using Machine Learning and Artificial Intelligence to analyze lifestyle data, estimate burnout risk, and provide personalized wellness guidance before the condition worsens.

---

### 🎯 Problem Statement

Many individuals experience burnout because they:

- Work for long hours without proper breaks.
- Spend excessive time on digital devices.
- Get insufficient sleep.
- Have poor work-life balance.
- Experience continuous stress and mental exhaustion.

Traditional assessments are mostly manual and reactive. NeuroSync AI provides a proactive AI-driven solution for early burnout detection.

---

### 💡 Proposed Solution

NeuroSync AI analyzes user lifestyle information and predicts burnout risk using trained Machine Learning models. Based on the prediction, the platform provides meaningful insights, visual analytics, and AI-generated wellness recommendations to help users improve their daily routine.

---

### ⚙️ Key Features

- 🏠 Modern Interactive Dashboard
- 📊 Burnout Risk Prediction
- 📁 Batch CSV Prediction
- 📈 Exploratory Data Analysis (EDA)
- 🤖 AI Wellness Assistant
- 📉 Model Performance Comparison
- 📋 Dataset Viewer
- 📊 Visual Charts & Analytics
- ⚡ Real-time Prediction Results

---

### 🧠 Machine Learning Workflow

The complete project workflow includes:

1. Data Collection
2. Data Cleaning & Preprocessing
3. Feature Engineering
4. Exploratory Data Analysis (EDA)
5. Machine Learning Model Training
6. Deep Learning Model Training
7. Model Evaluation
8. Burnout Risk Prediction
9. AI-based Wellness Recommendations

---

### 💻 Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- TensorFlow / Keras
- Plotly
- Hugging Face API
- Machine Learning
- Deep Learning

---

### 🌍 Expected Impact

NeuroSync AI helps students, employees, freelancers, and organizations monitor burnout risk at an early stage. By providing predictive insights and personalized wellness suggestions, the platform encourages healthier work habits, improved productivity, better mental well-being, and informed decision-making.

**"Predict Early. Prevent Burnout. Improve Well-being."**
""")

    glass_card_close()