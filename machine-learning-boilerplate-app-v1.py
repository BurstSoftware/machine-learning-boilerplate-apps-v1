import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# --- Data Generation ---
@st.cache_data
def generate_sample_data(num_samples=100):
    np.random.seed(42)  # for reproducibility
    data = {
        'feature1': np.random.rand(num_samples),
        'feature2': np.random.rand(num_samples),
        'feature3': np.random.rand(num_samples),
        'target': np.random.choice(['A', 'B', 'C'], num_samples)  # Classification labels
    }
    df = pd.DataFrame(data)
    return df

# Generate or load sample data
df = generate_sample_data()

# --- Data Preprocessing ---
X = df[['feature1', 'feature2', 'feature3']]
y = df['target']

# --- Model Training ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# --- Streamlit App ---
st.title('Basic Machine Learning App')
st.info('This app demonstrates a simple machine learning classification task.')

# Sidebar for input features
with st.sidebar:
    st.header('Input Features')
    feature1 = st.slider('Feature 1', 0.0, 1.0, 0.5)
    feature2 = st.slider('Feature 2', 0.0, 1.0, 0.5)
    feature3 = st.slider('Feature 3', 0.0, 1.0, 0.5)

# --- Prediction Section ---
st.header('Prediction')

# Prepare input data for prediction
input_data = pd.DataFrame({
    'feature1': [feature1],
    'feature2': [feature2],
    'feature3': [feature3]
})

# Make prediction
prediction = clf.predict(input_data)[0]
prediction_proba = clf.predict_proba(input_data)

# Display Prediction Result
st.write(f'**Predicted Class:** {prediction}')

# Display Class Probabilities
df_prediction_proba = pd.DataFrame(prediction_proba, columns=clf.classes_)
st.write('**Class Probabilities:**')
st.dataframe(df_prediction_proba)

# --- Data Display Section ---
st.header('Data Display')

if st.checkbox('Show Raw Data'):
    st.subheader('Raw Data')
    st.dataframe(df)

if st.checkbox('Show Data Statistics'):
    st.subheader('Data Statistics')
    st.dataframe(df.describe())

# --- Model Explanation ---
st.header('Model Explanation')
st.write('A Random Forest Classifier was used to build this model.  It predicts which class a new data point belongs to based on the values of feature1, feature2, and feature3.')
