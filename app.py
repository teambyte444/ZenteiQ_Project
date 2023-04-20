import streamlit as st
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split


data = pd.read_csv('C://Users.//ADMIN//Downloads//Project_data.csv.csv')

# Remove irrelevant columns
data.drop(['Curricular units 1st sem (credited)', 'Curricular units 1st sem (enrolled)', 
           'Curricular units 1st sem (evaluations)', 'Curricular units 1st sem (approved)', 
           'Curricular units 1st sem (grade)', 'Curricular units 1st sem (without evaluations)', 
           'Curricular units 2nd sem (credited)', 'Curricular units 2nd sem (enrolled)', 
           'Curricular units 2nd sem (evaluations)', 'Curricular units 2nd sem (approved)', 
           'Curricular units 2nd sem (grade)', 'Curricular units 2nd sem (without evaluations)'], axis=1, inplace=True)

# Encode categorical variables
data = pd.get_dummies(data, columns=['Marital status', 'Application mode', 'Course', 'Daytime/evening attendance', 
                                     'Previous qualification', 'Nacionality', 'Mother\'s qualification', 
                                     'Father\'s qualification', 'Mother\'s occupation', 'Father\'s occupation'])

# Fill missing values with the mean
data.fillna(data.mean(numeric_only=True), inplace=True)

# Separate target variable from features
X = data.drop('Target', axis=1)
y = data['Target']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.preprocessing import LabelEncoder
# Convert string labels to numeric labels
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

# Train XGBoost model
model = xgb.XGBClassifier()
model.fit(X_train, y_train)

# Define Streamlit web application
st.title('Student Dropout Prediction')

# Define input fields
marital_status = st.selectbox('Marital Status', ['Married', 'Single', 'Divorced'])
application_mode = st.selectbox('Application Mode', ['Online', 'Offline'])
application_order = st.number_input('Application Order', min_value=0, max_value=10)
course = st.selectbox('Course', ['Engineering', 'Science', 'Arts'])
attendance = st.selectbox('Daytime/Evening Attendance', ['Daytime', 'Evening'])
prev_qual = st.selectbox('Previous Qualification', ['High School', 'Bachelor\'s Degree', 'Master\'s Degree'])
prev_qual_grade = st.number_input('Previous Qualification Grade', min_value=0, max_value=20)
nationality = st.selectbox('Nationality', ['Portuguese', 'Non-Portuguese'])
mother_qual = st.selectbox('Mother\'s Qualification', ['High School', 'Bachelor\'s Degree', 'Master\'s Degree'])
father_qual = st.selectbox('Father\'s Qualification', ['High School', 'Bachelor\'s Degree', 'Master\'s Degree'])
mother_occ = st.selectbox('Mother\'s Occupation', ['Unemployed', 'Employed'])
father_occ = st.selectbox('Father\'s Occupation', ['Unemployed', 'Employed'])
admission_grade = st.number_input('Admission Grade', min_value=0, max_value=20)
displaced = st.selectbox('Displaced', ['Yes', 'No'])
special_needs = st.selectbox('Educational Special Needs', ['Yes', 'No'])
debtor = st.selectbox('Debtor', ['Yes', 'No'])
fees_up_to_date = st.selectbox('Tuition Fees Up to Date', ['Yes', 'No'])
gender = st.selectbox('Gender', ['Male', 'Female'])
scholarship_holder = st.selectbox('Scholarship Holder', ['Yes', 'No'])
age = st.number_input('Age at Enrollment', min_value=17, max_value=60)
international = st.selectbox('International', ['Yes', 'No'])
cu_1_credited = st.number_input('Curricular Units 1st Sem (Credited)', min_value=0, max_value=60)
cu_1_enrolled = st.number_input('Curricular Units 1st Sem (Enrolled)', min_value=0, max_value=60)
cu_1_evaluations = st.number_input('Curricular Units 1st Sem (Evaluations)', min_value=0, max_value=60)
cu_1_approved = st.number_input('Curricular Units 1st Sem (Approved)', min_value=0, max_value=60)
cu_1_grade = st.number_input('Curricular Units 1st Sem (Grade)', min_value=0, max_value=20)
cu_1_without_eval = st.number_input('Curricular Units 1st Sem (Without Evaluations)', min_value=0, max_value=60)
cu_2_credited = st.number_input('Curricular Units 2nd Sem (Credited)', min_value=0, max_value=60)
cu_2_enrolled = st.number_input('Curricular Units 2nd Sem (Enrolled)', min_value=0, max_value=60)
cu_2_evaluations = st.number_input('Curricular Units 2nd Sem (Evaluations)', min_value=0, max_value=60)
cu_2_approved = st.number_input('Curricular Units 2nd Sem (Approved)', min_value=0, max_value=60)
cu_2_grade = st.number_input('Curricular Units 2nd Sem (Grade)', min_value=0, max_value=20)
cu_2_without_eval = st.number_input('Curricular Units 2nd Sem (Without Evaluations)', min_value=0, max_value=60)
unemployment_rate = st.number_input('Unemployment Rate', min_value=0.0, max_value=100.0, step=0.1)
inflation_rate = st.number_input('Inflation Rate', min_value=0.0, max_value=100.0, step=0.1)
gdp = st.number_input('GDP', min_value=0.0)

import random

def my_prediction_function(marital_status, application_mode, application_order, course, attendance, prev_qual, prev_qual_grade, nationality, mother_qual, father_qual, mother_occ, father_occ, admission_grade, displaced, special_needs, debtor, fees_up_to_date, gender, scholarship_holder, age, international, cu_1_credited, cu_1_enrolled, cu_1_evaluations, cu_1_approved, cu_1_grade, cu_1_without_eval, cu_2_credited, cu_2_enrolled, cu_2_evaluations, cu_2_approved, cu_2_grade, cu_2_without_eval, unemployment_rate, inflation_rate, gdp):
    prediction = random.choice(['Accepted', 'Rejected'])
    return prediction

# Create a button to generate the prediction
if st.button('Generate Prediction'):
    # Call your prediction function here and store the result in a variable (e.g. "prediction")
    prediction = my_prediction_function(marital_status, application_mode, application_order, course, attendance, prev_qual, prev_qual_grade, nationality, mother_qual, father_qual, mother_occ, father_occ, admission_grade, displaced, special_needs, debtor, fees_up_to_date, gender, scholarship_holder, age, international, cu_1_credited, cu_1_enrolled, cu_1_evaluations, cu_1_approved, cu_1_grade, cu_1_without_eval, cu_2_credited, cu_2_enrolled, cu_2_evaluations, cu_2_approved, cu_2_grade, cu_2_without_eval, unemployment_rate, inflation_rate, gdp)

    # Display the prediction to the user
    st.write('Prediction:', prediction)
