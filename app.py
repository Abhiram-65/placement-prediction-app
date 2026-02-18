import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load model and preprocessing objects
model = joblib.load("models/decision_tree_model.pkl")
scaler = joblib.load("models/scaler.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")

# Load dataset
df = pd.read_csv("data/students_dataset.csv")

# Title
st.title("Placement Probability & Gap Analysis Engine")

st.info("This system predicts placement category based on student profile and provides improvement suggestions. Predictions are probabilistic, not guaranteed.")

st.markdown("---")

# Input Section
st.header("Student Input")

cgpa = st.slider("CGPA", 5.0, 10.0, 7.5)
backlogs = st.selectbox("Backlogs (0 = No, 1 = Yes)", [0, 1])
internships = st.slider("Internship Count", 0, 3, 1)
coding = st.slider("Coding Rating", 800, 2200, 1200)
aptitude = st.slider("Aptitude Score", 30, 100, 60)

# Create input dataframe
student_data = pd.DataFrame([{
    "CGPA": cgpa,
    "Backlogs": backlogs,
    "Internship_Count": internships,
    "Coding_Rating": coding,
    "Aptitude_Score": aptitude
}])

# Scale input
student_scaled = scaler.transform(student_data)

st.markdown("---")

# Prediction Button
if st.button("Predict Placement"):

    # Prediction
    prediction_encoded = model.predict(student_scaled)
    prediction = label_encoder.inverse_transform(prediction_encoded)

    # Probabilities
    probabilities = model.predict_proba(student_scaled)
    prob_df = pd.DataFrame(probabilities, columns=label_encoder.classes_)

    # Result
    st.subheader("Prediction Result")
    st.success(f"Predicted Placement Category: {prediction[0]}")

    # Probability Chart
    st.subheader("Probability Distribution")

    fig, ax = plt.subplots()
    ax.bar(prob_df.columns, prob_df.iloc[0])
    plt.xticks(rotation=45)
    st.pyplot(fig)

    st.markdown("---")

    # Gap Analysis
    st.subheader("Gap Analysis (Target: Tier_1)")

    target_class = "Tier_1"
    target_df = df[df["Target_Class"] == target_class]

    target_mean = target_df.drop("Target_Class", axis=1).mean()

    student_vector = student_data.iloc[0]
    gap = target_mean - student_vector
    gap_positive = gap[gap > 0]

    if len(gap_positive) > 0:
        for feature, value in gap_positive.items():
            if feature == "CGPA":
                value = min(value, 1.0)  # realistic cap
            st.write(f"Increase {feature} by approximately {round(value, 2)}")
    else:
        st.success("You already match or exceed Tier_1 profile!")
