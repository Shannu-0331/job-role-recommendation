import streamlit as st
import pandas as pd
import re
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Job Recommendation System",
    page_icon="💼",
    layout="wide"
)

# -------------------------------------------------
# TITLE
# -------------------------------------------------
st.title("💼 Job Role & Skill Recommendation System")
st.caption("Predict a suitable job role and get recommended skills based on your input.")

st.divider()

# -------------------------------------------------
# LOAD & PREPROCESS DATA
# -------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("jobs_skills_dataset.csv")

    df["skills"] = df["skills"].astype(str).str.lower()
    df["job_role"] = df["job_role"].astype(str).str.lower()

    # Remove low-frequency roles
    job_counts = df["job_role"].value_counts()
    valid_roles = job_counts[job_counts >= 10].index
    df = df[df["job_role"].isin(valid_roles)]

    def clean_skills(text):
        text = re.sub(r"[|/;]", ",", text)
        text = re.sub(r"[^a-z, ]", "", text)
        text = re.sub(r"\s+", " ", text)
        return [s.strip() for s in text.split(",") if s.strip()]

    df["skills_temp"] = df["skills"].apply(clean_skills)

    all_skills = [s for row in df["skills_temp"] for s in row]
    skill_freq = Counter(all_skills)
    common_skills = {s for s, c in skill_freq.items() if c >= 2}

    df["skills_temp"] = df["skills_temp"].apply(
        lambda skills: [s for s in skills if s in common_skills]
    )

    df = df[df["skills_temp"].apply(len) >= 2]
    df["skills"] = df["skills_temp"].apply(lambda x: ", ".join(x))

    return df[["job_role", "skills"]]


df = load_data()

# -------------------------------------------------
# TRAIN MODEL
# -------------------------------------------------
@st.cache_resource
def train_model(df):
    tfidf = TfidfVectorizer(stop_words="english", max_features=1000)
    X = tfidf.fit_transform(df["skills"])

    le = LabelEncoder()
    y = le.fit_transform(df["job_role"])

    model = LinearSVC()
    model.fit(X, y)

    return tfidf, le, model


tfidf, label_encoder, svm_model = train_model(df)

# -------------------------------------------------
# ROLE → SKILL MAP
# -------------------------------------------------
def build_role_skill_map(df):
    role_skill_map = {}
    for role in df["job_role"].unique():
        skills = []
        for text in df[df["job_role"] == role]["skills"]:
            skills.extend(text.split(", "))
        role_skill_map[role] = Counter(skills)
    return role_skill_map


role_skill_map = build_role_skill_map(df)

# -------------------------------------------------
# PREDICTION LOGIC
# -------------------------------------------------
def predict_and_recommend(user_input, top_n=6):
    user_vector = tfidf.transform([user_input.lower()])
    pred_label = svm_model.predict(user_vector)
    role = label_encoder.inverse_transform(pred_label)[0]

    user_skills = set([s.strip() for s in user_input.lower().split(",")])

    recommendations = []
    for skill, _ in role_skill_map[role].most_common():
        if skill not in user_skills:
            recommendations.append(skill)
        if len(recommendations) == top_n:
            break

    return role, recommendations

# -------------------------------------------------
# SIDEBAR INPUT (CLEAN LOOK)
# -------------------------------------------------
st.sidebar.header("🔹 User Input")

user_input = st.sidebar.text_area(
    "Enter your skills (comma separated)",
    placeholder="python, sql, pandas, machine learning"
)

predict_btn = st.sidebar.button("Predict Job Role")

# -------------------------------------------------
# MAIN OUTPUT AREA
# -------------------------------------------------
if predict_btn:
    if user_input.strip() == "":
        st.warning("Please enter at least one skill.")
    else:
        role, skills = predict_and_recommend(user_input)

        st.subheader("🎯 Predicted Job Role")
        st.success(role.upper())

        st.subheader("📌 Recommended Skills")
        if skills:
            for skill in skills:
                st.markdown(f"- **{skill}**")
        else:
            st.info("You already have most of the required skills!")

else:
    st.info("⬅️ Enter your skills from the sidebar to get started.")

# -------------------------------------------------
# FOOTER
# -------------------------------------------------
st.divider()
st.caption("Built using Machine Learning (Linear SVM) and Streamlit")
