import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from PIL import Image

# Load dataset
diabetes_df = pd.read_csv('diabetes.csv')
X = diabetes_df.drop('Outcome', axis=1)
y = diabetes_df['Outcome']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=1)

# Model selection
model_choice = st.sidebar.selectbox("🔍 Choose Model", ["SVM (Linear)", "Logistic Regression"])
if model_choice == "SVM (Linear)":
    model = svm.SVC(kernel='linear', probability=True)
else:
    model = LogisticRegression()

model.fit(X_train, y_train)
train_acc = accuracy_score(model.predict(X_train), y_train)
test_acc = accuracy_score(model.predict(X_test), y_test)

# App layout
st.set_page_config(page_title="Diabetes Predictor", layout="wide")
col1, col2 = st.columns([1, 3])

with col1:
    try:
        img = Image.open("img.jpeg")
        st.image(img, caption="Diabetes Awareness", use_container_width=True)
    except FileNotFoundError:
        st.warning("Image 'img.jpeg' not found.")

with col2:
    st.title("🩺 Diabetes Risk Prediction")
    st.markdown("Use the sidebar to enter your health metrics and predict your diabetes risk.")

# Sidebar inputs
with st.sidebar:
    st.header("📋 Input Features")
    preg = st.slider('Pregnancies', 0, 17, 3)
    glucose = st.slider('Glucose', 0, 199, 117)
    bp = st.slider('Blood Pressure', 0, 122, 72)
    skinthickness = st.slider('Skin Thickness', 0, 99, 23)
    insulin = st.slider('Insulin', 0, 846, 30)
    bmi = st.slider('BMI', 0.0, 67.1, 32.0)
    dpf = st.slider('Diabetes Pedigree Function', 0.078, 2.42, 0.3725, 0.001)
    age = st.slider('Age', 21, 81, 29)

# Prediction
input_data = [preg, glucose, bp, skinthickness, insulin, bmi, dpf, age]
input_df = pd.DataFrame([input_data], columns=X.columns)
input_scaled = scaler.transform(input_df)
prediction = model.predict(input_scaled)
probability = model.predict_proba(input_scaled)[0][1]

st.subheader("🧠 Prediction Result")
if prediction[0] == 1:
    st.warning(f"⚠️ This person has diabetes.\nConfidence: {probability:.2%}")
else:
    st.success(f"✅ This person does not have diabetes.\nConfidence: {1 - probability:.2%}")

# Downloadable report
report = f"Prediction: {'Diabetic' if prediction[0] == 1 else 'Non-Diabetic'}\nConfidence: {probability:.2%}"
st.download_button("📥 Download Report", report, file_name="diabetes_report.txt")

# Dataset insights
with st.expander("📊 Dataset Summary"):
    st.write(diabetes_df.describe())

with st.expander("📈 Outcome Distribution"):
    st.bar_chart(diabetes_df["Outcome"].value_counts())

with st.expander("📌 Model Performance"):
    st.metric("Training Accuracy", f"{train_acc:.2%}")
    st.metric("Testing Accuracy", f"{test_acc:.2%}")