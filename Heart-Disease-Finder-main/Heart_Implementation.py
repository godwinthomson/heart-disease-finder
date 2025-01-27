import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

# Set Streamlit title and header
st.title("Heart Disease Prediction using Logistic Regression")
st.header("Predicting the presence of heart disease")

# Load the dataset
url = r"C:\Users\admin\Documents\Projects\Heart Disease Prediction Project\Heart_Disease_DataSet.csv"
columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
df = pd.read_csv(url, header=None, names=columns)

# Data Preprocessing
label_encoder = LabelEncoder()
categorical_columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']

# Apply label encoding to categorical columns
for col in categorical_columns:
    df[col] = label_encoder.fit_transform(df[col])

# Ensure all columns are numeric
df = df.apply(pd.to_numeric, errors='coerce')

# Handle missing values by using SimpleImputer
imputer = SimpleImputer(strategy='mean')
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Split the dataset into features (X) and target (y)
X = df_imputed.drop(columns='target')
y = df_imputed['target']

# Ensure target column is binary (0 or 1), handling any unexpected values
if not y.isin([0, 1]).all():
    y = y.apply(lambda x: 1 if x > 0 else 0)  # Convert values greater than 0 to 1

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train the Logistic Regression model
logreg_classifier = LogisticRegression(max_iter=1000, random_state=42)
logreg_classifier.fit(X_scaled, y)

# Evaluate the model
y_pred = logreg_classifier.predict(X_scaled)
accuracy = accuracy_score(y, y_pred)
classification_rep = classification_report(y, y_pred)

# Display the model's accuracy and classification report (optional, can be removed if not needed)
#st.write(f"Model Accuracy: {accuracy * 100:.2f}%")
#st.write("Classification Report:")
#st.text(classification_rep)

# Input fields for the user to enter their data for prediction
st.header("Input Your Data to Predict Heart Disease")

# Create input fields for each feature
def user_input():
    input_data = {}
    for feature in X.columns:
        if feature == 'sex':
            input_data[feature] = st.selectbox(feature, [0, 1])  # Male: 1, Female: 0
        elif feature == 'cp':
            input_data[feature] = st.selectbox(feature, [0, 1, 2, 3])  # Chest pain type
        elif feature == 'fbs':
            input_data[feature] = st.selectbox(feature, [0, 1])  # Fasting blood sugar
        elif feature == 'restecg':
            input_data[feature] = st.selectbox(feature, [0, 1, 2])  # Resting electrocardiographic results
        elif feature == 'exang':
            input_data[feature] = st.selectbox(feature, [0, 1])  # Exercise induced angina
        elif feature == 'slope':
            input_data[feature] = st.selectbox(feature, [0, 1, 2])  # Slope of peak exercise ST segment
        elif feature == 'ca':
            input_data[feature] = st.selectbox(feature, [0, 1, 2, 3])  # Number of major vessels colored by fluoroscopy
        elif feature == 'thal':
            input_data[feature] = st.selectbox(feature, [1, 2, 3])  # Thalassemia
        else:
            input_data[feature] = st.slider(feature, min_value=int(df[feature].min()), max_value=int(df[feature].max()), value=int(df[feature].mean()))
    return pd.DataFrame(input_data, index=[0])

# Get user input
user_data = user_input()

# Button to trigger prediction
predict_button = st.button("Predict Heart Disease")

if predict_button:
    # Ensure the user input is numeric and apply standardization
    user_data = user_data.apply(pd.to_numeric, errors='coerce')
    user_data_scaled = scaler.transform(user_data)

    # Predict the presence of heart disease
    user_prediction = logreg_classifier.predict(user_data_scaled)

    # Convert the prediction (0 or 1) into a human-readable result
    if user_prediction[0] == 1:
        prediction = "Heart disease detected."
    else:
        prediction = "No heart disease detected."

    # Display the prediction result
    st.write(f"Prediction: {prediction}")
