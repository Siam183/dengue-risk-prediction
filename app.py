import streamlit as st
import pandas as pd
import requests
import numpy as np
import joblib  # Use joblib instead of pickle
from datetime import datetime

# Check Scikit-learn version
import sklearn
st.write(f"Scikit-learn version: {sklearn.__version__}")

# Load trained model
def load_model():
    model_path = "dengue_prediction_model.pkl"
    try:
        model = joblib.load(model_path)  # Use joblib for better compatibility
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def load_data():
    file_path = "dengue_outbreak_dataset_2008_today.csv"
    return pd.read_csv(file_path)

def fetch_weather_data(location, date):
    api_key = "d9e3711fdeb6fb638849a60f0ad71d52"  # Your API key
    url = f"https://api.openweathermap.org/data/2.5/forecast?q={location},BD&appid={api_key}&units=metric"
    
    response = requests.get(url)
    
    # Debugging: Print the response for error checking
    print(f"API Response Status: {response.status_code}")
    print(f"API Response Text: {response.text}")
    
    if response.status_code == 200:
        data = response.json()
        for entry in data["list"]:
            forecast_date = datetime.utcfromtimestamp(entry["dt"]).date()
            if forecast_date == date:
                return {
                    "Temp_Min (°C)": entry["main"]["temp_min"],
                    "Temp_Max (°C)": entry["main"]["temp_max"],
                    "Humidity (%)": entry["main"]["humidity"],
                    "Rainfall (mm)": entry.get("rain", {}).get("3h", 0),
                }
    else:
        return {"error": f"Error fetching data: {response.status_code} - {response.text}"}

def predict_dengue_risk(model, features):
    feature_values = np.array([[features["Temp_Min (°C)"], features["Temp_Max (°C)"], features["Humidity (%)"], features["Rainfall (mm)"]]])
    
    if hasattr(model, "predict"):
        return model.predict(feature_values)[0]
    else:
        return "Error: Model is not compatible."

def main():
    st.title("Real-Time Dengue Outbreak Risk Prediction in Bangladesh")
    
    model = load_model()
    if model is None:
        st.error("Failed to load model. Check if 'dengue_prediction_model.pkl' exists.")
        return

    df = load_data()
    
    st.subheader("Dataset Preview")
    st.write(df.head())
    
    st.subheader("Enter Location and Date for Prediction")
    location = st.text_input("Enter city name in Bangladesh", "Dhaka")
    date_choice = st.date_input("Select Date for Prediction", datetime.today().date())
    
    if st.button("Predict Dengue Risk"):
        weather_data = fetch_weather_data(location, date_choice)
        
        if "error" in weather_data:
            st.write(weather_data["error"])
        elif weather_data:
            risk = predict_dengue_risk(model, weather_data)
            st.subheader("Predicted Dengue Outbreak Risk")
            st.write(f"The estimated dengue outbreak risk for {location} on {date_choice} is {risk:.2f}%.")
        else:
            st.write("Failed to fetch weather data. Please check the location or try again later.")

if __name__ == "__main__":
    main()
