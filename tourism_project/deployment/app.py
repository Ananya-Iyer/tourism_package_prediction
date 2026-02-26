import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the trained model
model_path = hf_hub_download(repo_id="ananyaarvindiyer/tourism-package-prediction-ui", filename="best_tourism_package_prediction_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI
st.title("Tourism Package Purchase Prediction")
st.write("""
This application predicts whether a customer will purchase the newly introduced Wellness Tourism Package before contacting them.
""")

# User inputs

# Categorical
contact_source = st.selectbox("Contact Source", ["Self Enquiry", "Company Invited"])
occupation = st.selectbox("Occupation", ["Salaried", "Free Lancer", "Small Business", "Large Business"])
gender = st.selectbox("Gender", ["Male", "Female"])
product_proposed = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
marital_status = st.selectbox("Marital Status", ["Married", "Divorced", "Single"])
passport = st.selectbox("Passport", ["Yes", "No"])
own_car = st.selectbox("Own Car", ["Yes", "No"])
job_title = st.selectbox("Job Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
city_tier = st.selectbox("City Tier", ["1", "2", "3"])
property_star = st.selectbox("Preferred Property Star", ["3", "4", "5", "7"])


# Numerical
cust_age = st.number_input("Age", min_value=1, max_value=100, value=25, step=1)
pitch_duration = st.number_input("Sales Pitch Duration", min_value=1, max_value=200, value=5, step=1)
people_count = st.number_input("Visitors Count", min_value=1, max_value=50, value=2, step=1)
follow_up_count = st.number_input("Follow up Count", min_value=1, max_value=20, value=2, step=1)
number_of_trips = st.number_input("Number of Trips", min_value=1, max_value=100, value=1, step=1)
number_of_childern = st.number_input("Number of Children Visiting", min_value=0, max_value=10, value=1, step=1)
monthly_income = st.number_input("Monthly Income", min_value=1000, max_value=1000000, value=5000)
pitch_feedback = st.number_input("Pitch Satisfaction Score", min_value=1, max_value=10, value=1, step=1)


# Assemble input into DataFrame
input_data = pd.DataFrame([{
    'Age': cust_age,
    'TypeofContact': contact_source,
    'CityTier': city_tier,
    'DurationOfPitch': pitch_duration,
    'Occupation': occupation,
    'Gender': gender,
    'NumberOfPersonVisiting': people_count,
    'NumberOfFollowups': follow_up_count,
    'ProductPitched': product_proposed,
    'PreferredPropertyStar': property_star,
    'MaritalStatus': marital_status,
    'NumberOfTrips': number_of_trips,
    'Passport': 1 if passport == "Yes" else 0,
    'PitchSatisfactionScore': pitch_feedback,
    'OwnCar': 1 if own_car == "Yes" else 0,
    'NumberOfChildrenVisiting': number_of_childern,
    'Designation': job_title,
    'MonthlyIncome': monthly_income
}])

# #------------------------For debugging----------------

# expected = list(model.feature_names_in_)
# actual = list(input_data.columns)

# st.write("Expected columns:", expected)
# st.write("Actual columns:", actual)

# st.write("Missing columns:", set(expected) - set(actual))
# st.write("Extra columns:", set(actual) - set(expected))
# #--------------------------------------------------------


# Predict button
if st.button("Predict Purchase"):
    prediction = model.predict(input_data)[0]
    st.subheader("Prediction Result:")
    if prediction == 1:
        st.success("Customer is likely to PURCHASE the package ✅.")
    else:
        st.error("Customer is NOT likely to purchase ❌.")
