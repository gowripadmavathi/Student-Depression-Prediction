import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Student Depression Predictor",
    page_icon="🧠",
    layout="centered"
)

# ── Load model & scaler ───────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_model():
    model  = pickle.load(open(os.path.join(BASE, "model.pkl"),  "rb"))
    scaler = pickle.load(open(os.path.join(BASE, "scaler.pkl"), "rb"))
    return model, scaler

model, scaler = load_model()

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f8f4ff; }
    .stButton > button {
        background: linear-gradient(135deg, #6929C4, #9B59B6);
        color: white; font-size: 18px; font-weight: bold;
        border-radius: 12px; padding: 12px 40px;
        border: none; width: 100%; margin-top: 10px;
    }
    .stButton > button:hover { opacity: 0.9; }
    .result-box {
        padding: 25px; border-radius: 15px;
        text-align: center; margin-top: 20px;
        font-size: 22px; font-weight: bold;
    }
    .depressed   { background: #FFEBEE; border: 2px solid #E53935; color: #B71C1C; }
    .no-depressed{ background: #E8F5E9; border: 2px solid #43A047; color: #1B5E20; }
    .prob-bar    { height: 22px; border-radius: 10px; margin-top: 8px; }
    .section-title { color: #6929C4; font-weight: 700; font-size: 17px; margin-top: 15px; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────
st.markdown("## 🧠 Student Depression Predictor")
st.markdown("Fill in the student details below and click **Predict** to check depression risk.")
st.markdown("---")

# ── Input Form ────────────────────────────────────────────────
st.markdown('<p class="section-title">👤 Personal Information</p>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
with col2:
    age = st.slider("Age", 15, 35, 20)

st.markdown('<p class="section-title">📚 Academic & Lifestyle</p>', unsafe_allow_html=True)
col3, col4 = st.columns(2)
with col3:
    academic_pressure = st.slider("Academic Pressure", 1.0, 5.0, 3.0, step=0.5,
                                   help="1 = Very Low, 5 = Very High")
    study_satisfaction = st.slider("Study Satisfaction", 1.0, 5.0, 3.0, step=0.5,
                                    help="1 = Very Unsatisfied, 5 = Very Satisfied")
    study_hours = st.slider("Daily Study Hours", 1, 16, 6)
with col4:
    sleep = st.selectbox("Sleep Duration", [
        "Less than 5 hours", "5-6 hours", "7-8 hours", "More than 8 hours"
    ], index=2)
    diet = st.selectbox("Dietary Habits", ["Unhealthy", "Moderate", "Healthy"], index=1)
    financial_stress = st.slider("Financial Stress", 1, 5, 2,
                                  help="1 = Very Low, 5 = Very High")

st.markdown('<p class="section-title">🏥 Mental Health Background</p>', unsafe_allow_html=True)
col5, col6 = st.columns(2)
with col5:
    suicidal = st.selectbox("Ever had Suicidal Thoughts?", ["No", "Yes"])
with col6:
    family_history = st.selectbox("Family History of Mental Illness?", ["No", "Yes"])

st.markdown("---")

# ── Predict Button ────────────────────────────────────────────
if st.button("🔍 Predict Depression Risk"):

    # Encode inputs
    gender_enc   = 1 if gender == "Female" else 0
    suicidal_enc = 1 if suicidal == "Yes" else 0
    family_enc   = 1 if family_history == "Yes" else 0
    sleep_enc    = {"Less than 5 hours": 1, "5-6 hours": 2, "7-8 hours": 3, "More than 8 hours": 4}[sleep]
    diet_enc     = {"Unhealthy": 1, "Moderate": 2, "Healthy": 3}[diet]

    input_data = pd.DataFrame([{
        "Gender": gender_enc,
        "Age": age,
        "Academic Pressure": academic_pressure,
        "Study Satisfaction": study_satisfaction,
        "Sleep Duration": sleep_enc,
        "Dietary Habits": diet_enc,
        "Have you ever had suicidal thoughts ?": suicidal_enc,
        "Study Hours": study_hours,
        "Financial Stress": financial_stress,
        "Family History of Mental Illness": family_enc
    }])

    input_scaled = scaler.transform(input_data)
    prediction   = model.predict(input_scaled)[0]
    probability  = model.predict_proba(input_scaled)[0][1]
    prob_pct     = round(probability * 100, 1)

    # ── Result Display ────────────────────────────────────────
    if prediction == 1:
        st.markdown(f"""
        <div class="result-box depressed">
            ⚠️ Depression Risk Detected<br>
            <span style="font-size:16px;">Probability: {prob_pct}%</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-box no-depressed">
            ✅ No Depression Risk Detected<br>
            <span style="font-size:16px;">Probability: {prob_pct}%</span>
        </div>
        """, unsafe_allow_html=True)

    # ── Risk Factors ──────────────────────────────────────────
    st.markdown("### 📊 Risk Factor Analysis")

    factors = {
        "Academic Pressure": (academic_pressure - 1) / 4,
        "Financial Stress":  (financial_stress - 1) / 4,
        "Poor Sleep":        (4 - sleep_enc) / 3,
        "Unhealthy Diet":    (3 - diet_enc) / 2,
        "Suicidal Thoughts": suicidal_enc,
        "Family History":    family_enc,
    }

    for factor, score in factors.items():
        pct  = int(score * 100)
        color = "#E53935" if pct > 60 else "#FF9800" if pct > 30 else "#43A047"
        st.markdown(f"""
        <div style="margin-bottom:10px;">
            <div style="display:flex; justify-content:space-between;">
                <span style="font-size:14px;">{factor}</span>
                <span style="font-size:14px; font-weight:bold; color:{color};">{pct}%</span>
            </div>
            <div style="background:#e0e0e0; border-radius:10px; height:12px;">
                <div style="width:{pct}%; background:{color}; height:12px; border-radius:10px;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Recommendations ───────────────────────────────────────
    st.markdown("### 💡 Recommendations")
    tips = []
    if sleep_enc <= 2:
        tips.append("😴 **Improve sleep** — aim for 7–8 hours daily")
    if academic_pressure >= 4:
        tips.append("📚 **Manage academic load** — break tasks into smaller goals")
    if financial_stress >= 4:
        tips.append("💰 **Seek financial guidance** — talk to college counsellor")
    if diet_enc == 1:
        tips.append("🥗 **Improve diet** — eat balanced meals regularly")
    if suicidal_enc == 1:
        tips.append("🆘 **Please seek professional help immediately** — contact a counsellor")
    if not tips:
        tips.append("🌟 Keep maintaining your healthy lifestyle!")

    for tip in tips:
        st.markdown(f"- {tip}")

# ── Footer ────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#999; font-size:13px;'>
    Built by <b>Narala Gowri Padmavathi</b> · 
    Logistic Regression Model · Accuracy: 100% · Dataset: 502 Students<br>
    ⚠️ This tool is for academic purposes only — not a clinical diagnosis tool.
</div>
""", unsafe_allow_html=True)
