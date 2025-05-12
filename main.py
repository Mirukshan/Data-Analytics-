import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

# Function to load or train a new model
def load_trained_model():
    if os.path.exists('model_rf_optimized.joblib'):
        st.write("Loading Random Forest model from 'model_rf_optimized.joblib'...")
        model = joblib.load('model_rf_optimized.joblib')
    else:
        st.write("No model file found. Training a new model...")
        model = None
    return model

# Function to train and evaluate the model
def train_new_model(df):
    FEATURES = ["PM10", "SO2", "NO2", "CO", "O3", "TEMP", "PRES", "RAIN", "WSPM"]
    TARGET = "PM2.5"
    X = df[FEATURES].fillna(df[FEATURES].mean())  # Fill missing values with mean
    y = df[TARGET].fillna(df[TARGET].mean())     # Fill missing values with mean

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    accuracy = r2 * 100

    st.write(f"Model trained successfully!")
    st.write(f"Mean Squared Error (MSE): {mse:.2f}")
    st.write(f"R-squared (R²): {r2:.3f}")
    st.write(f"Model Accuracy: {accuracy:.2f}%")

    joblib.dump(model, 'model_rf_optimized.joblib')
    return model

# Function to handle data merging
def merge_datasets():
    datasets = {
        "Data_Aotizhongxin": ("data/Data_Aotizhongxin.csv", "Forest"),
        "Data_Dingling": ("data/Data_Dingling.csv", "Factory"),
        "Data_Gucheng": ("data/Data_Gucheng.csv", "Urban"),
        "Data_Huairou": ("data/Data_Huairou.csv", "Rural"),
        "Data_Wanshouxigong": ("data/Data_Wanshouxigong.csv", "Urban"),
    }

    merged_df = pd.DataFrame()
    for data_name, (file_path, category) in datasets.items():
        df = pd.read_csv(file_path)
        df['Category'] = category
        merged_df = pd.concat([merged_df, df], ignore_index=True)

    output_file = "merged_data.csv"
    merged_df.to_csv(output_file, index=False)
    return merged_df

# Sidebar for navigation
def sidebar_navigation():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select a page:", 
                            ["Model Training & Evaluation", "Data Visualizations", "Real-Time AQI Prediction"])
    return page

# Model Training & Evaluation page
def page_modeling_prediction():
    st.title("Model Training & Evaluation")

    # Merge datasets or load merged data
    merged_df = merge_datasets()
    st.markdown("#### Dataset Overview")
    st.dataframe(merged_df.head())

    # Load or train the model
    model = load_trained_model()
    if model is None:
        model = train_new_model(merged_df)

    # Evaluate the trained model
    st.markdown("### Model Evaluation")
    FEATURES = ["PM10", "SO2", "NO2", "CO", "O3", "TEMP", "PRES", "RAIN", "WSPM"]
    TARGET = "PM2.5"
    X = merged_df[FEATURES].fillna(merged_df[FEATURES].mean())
    y = merged_df[TARGET].fillna(merged_df[TARGET].mean())
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    accuracy = r2 * 100

    st.write(f"Mean Squared Error (MSE): {mse:.2f}")
    st.write(f"R-squared (R²): {r2:.3f}")
    st.write(f"Model Accuracy: {accuracy:.2f}%")

def page_visualizations():
    st.title("Data Visualizations")

    # Merge datasets or load merged data
    merged_df = merge_datasets()

    # Data cleaning and preprocessing
    merged_df_cleaned = merged_df.dropna().drop_duplicates()

    # Display basic data info
    st.subheader("Data Overview")
    st.write(f"Shape of the dataset: {merged_df_cleaned.shape}")
    st.write("Data Types and Missing Values:")
    st.write(merged_df_cleaned.info())
    missing_values = merged_df_cleaned.isnull().sum()
    st.write("Missing Values in Each Column:")
    st.write(missing_values)

    # Basic statistical summary
    st.write("Statistical Summary:")
    st.write(merged_df_cleaned.describe())

    # Univariate analysis: Distribution of each numerical column
    st.subheader("Univariate Analysis: Distribution of Numerical Columns")
    numerical_columns = merged_df_cleaned.select_dtypes(include=['float64', 'int64']).columns
    for col in numerical_columns:
        st.subheader(f'Distribution of {col}')
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.histplot(merged_df_cleaned[col], kde=True, bins=30, ax=ax)
        st.pyplot(fig)

    # Correlation heatmap
    st.subheader("Correlation Heatmap")
    numerical_features = merged_df_cleaned.select_dtypes(include=['number'])
    correlation_matrix = numerical_features.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, ax=ax)
    plt.title('Correlation Heatmap')
    st.pyplot(fig)

    # Multivariate analysis: Boxplot
    st.subheader("Multivariate Analysis: Boxplot")
    numerical_columns = merged_df_cleaned.select_dtypes(include=['number']).columns
    numerical_columns = [col for col in numerical_columns if col != 'Category']  # Exclude 'Category'
    for col in numerical_columns:
        st.subheader(f'Distribution of {col} by Category')
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.boxplot(data=merged_df_cleaned, x='Category', y=col, ax=ax)
        st.pyplot(fig)

    # Pairplot for relationships between features
    st.subheader("Pairplot of Numerical Features by Category")
    sns.pairplot(merged_df_cleaned[['CO', 'NO2', 'O3', 'PM10', 'PM2.5', 'SO2', 'Category']], hue='Category')
    st.pyplot(plt.gcf())

# Real-Time AQI Prediction page
def real_time_aqi_prediction(model):
    st.title("Real-Time AQI Prediction")

    feature_cols = ["PM10", "SO2", "NO2", "CO", "O3", "TEMP", "PRES", "RAIN", "WSPM"]
    input_data = {}

    st.markdown("### Enter Feature Values")
    for feature in feature_cols:
        input_data[feature] = st.number_input(f"{feature}:", value=0.0)

    if st.button("Predict PM2.5 Concentration"):
        user_data = np.array([list(input_data.values())])
        user_prediction = model.predict(user_data)
        pm25 = user_prediction[0]

        # Categorize air quality
        if pm25 <= 50:
            quality, quality_class = "Good", "good"
        elif pm25 <= 100:
            quality, quality_class = "Moderate", "moderate"
        elif pm25 <= 150:
            quality, quality_class = "Unhealthy for sensitive groups", "unhealthy-sensitive"
        elif pm25 <= 200:
            quality, quality_class = "Unhealthy", "unhealthy"
        elif pm25 <= 300:
            quality, quality_class = "Very Unhealthy", "very-unhealthy"
        else:
            quality, quality_class = "Hazardous", "hazardous"

        st.write(f"**Predicted PM2.5 Concentration:** {pm25:.2f}")
        st.markdown(f"<div class='{quality_class}'>Air Quality: {quality}</div>", unsafe_allow_html=True)

# Main function with sidebar for page navigation
def main():
    page = sidebar_navigation()

    model = load_trained_model()

    if page == "Model Training & Evaluation":
        page_modeling_prediction()
    elif page == "Data Visualizations":
        page_visualizations()
    elif page == "Real-Time AQI Prediction":
        if model is None:
            st.write("Model not found. Please train or upload a model first.")
        else:
            real_time_aqi_prediction(model)

if __name__ == "__main__":
    main()
