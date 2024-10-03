import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# Load the trained model
model = tf.keras.models.load_model('optimized_model.keras')

# Load the encoders and scaler
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)
# Retrieve feature names
feature_names = scaler.feature_names_in_

# Streamlit app title
st.title('Customer Salary Prediction')

# User input
geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92, 21)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Geography': [geography],
    'Gender': [gender],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member]
})

# Encode 'Gender'
input_data['Gender'] = label_encoder_gender.transform(input_data['Gender'])

# One-hot encode 'Geography'
geo_encoded = onehot_encoder_geo.transform(input_data[['Geography']]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

# Drop 'Geography' and concatenate encoded data
input_data = pd.concat([input_data.drop('Geography', axis=1).reset_index(drop=True), geo_encoded_df], axis=1)

# After all preprocessing steps, before scaling
# Ensure all expected features are present in input_data
missing_features = set(feature_names) - set(input_data.columns)
for feature in missing_features:
    input_data[feature] = 0  # Add missing features with default value 0

# *Reorder columns to match training data*
input_data = input_data[feature_names]

# Scale the input data
input_data_scaled = scaler.transform(input_data)

# Predict salary
prediction = model.predict(input_data_scaled)
predicted_salary = prediction[0][0]

# Display the prediction
st.write(f'Predicted Salary: {predicted_salary:.2f}')
