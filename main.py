import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import streamlit as st

# Load the dataset
df = pd.read_csv("diabetes.csv")


# Data Preprocessing
X = df.drop('Outcome', axis=1)
y = df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Model Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy * 100:.2f}%')

# Save the trained model
joblib.dump(model, 'diabetes_model.pkl')

# Streamlit Interface
st.title("Diabetes Prediction System")

# Input fields
pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20)
glucose = st.number_input("Glucose Level", min_value=0)
blood_pressure = st.number_input("Blood Pressure", min_value=0)
skin_thickness = st.number_input("Skin Thickness", min_value=0)
insulin = st.number_input("Insulin", min_value=0)
BMI = st.number_input("BMI", min_value=0.0)
diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0)
age = st.number_input("Age", min_value=0)

if st.button("Predict"):
    model = joblib.load('diabetes_model.pkl')
    prediction = model.predict([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, BMI, diabetes_pedigree, age]])
    result = "Diabetic" if prediction[0] == 1 else "Non-Diabetic"
    st.write(f"Prediction: {result}")
