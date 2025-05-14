import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import numpy as np
import os # Import os for file path handling
import gdown # Import gdown for downloading from Google Drive

# Import necessary components for the pipeline (needed for loading the model)
# These must be available in the environment where the model is loaded
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

import io # Required for capturing df.info() output

# --- Configuration ---
# Replace with the raw URL of your merged_data.csv file on GitHub

# Replace with the public shareable ID of your trained model file on Google Drive
# Ensure the file is shared publicly or handle authentication
GOOGLE_DRIVE_MODEL_FILE_ID = "1ilukwSmeVZ7ywBCBt-9LS35JmoYGzMbe"
MODEL_LOCAL_FILENAME = "random_forest_regressor_model.joblib"
GITHUB_DATA_URL = "merged_data.csv"

# --- Data Loading Function (Cached) ---
@st.cache_data # Cache the data loading for performance
def load_data(url):
    """Loads data from a given URL."""
    try:
        df = pd.read_csv('merged_data.csv')
        st.success("Data loaded successfully from GitHub!")
        return df
    except Exception as e:
        st.error(f"Error loading data from {url}: {e}")
        return None

# --- Model Loading Function (Cached) ---
@st.cache_resource # Cache the model loading as it's a resource
def load_model(file_id, output_path):
    """Downloads and loads the model from Google Drive."""
    try:
        # Check if the model file already exists locally
        if not os.path.exists(output_path):
            st.info(f"Downloading model from Google Drive (ID: {file_id})...")
            gdown.download(id=file_id, output=output_path, quiet=False)
            st.success("Model downloaded successfully!")
        else:
            st.info("Model file already exists locally. Loading...")

        # Load the model using joblib
        # Ensure the necessary classes (Pipeline, ColumnTransformer, etc.) are imported
        # in the global scope for joblib to find them.
        model = joblib.load(output_path)
        st.success("Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Error loading model from Google Drive or local file: {e}")
        st.warning("Please ensure the Google Drive file ID is correct and the file is publicly accessible.")
        return None


# --- Page Functions ---

def data_overview_page(df):
    """Displays data overview information."""
    st.title("📚 Data Overview")

    if df is not None:
        st.subheader("Dataset Shape")
        st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

        st.subheader("First 5 Rows")
        st.dataframe(df.head())

        st.subheader("Data Types and Missing Values")
        # Use io.StringIO to capture df.info() output
        st.text("DataFrame Info (Data Types and Non-Null Counts):")
        buffer = io.StringIO()
        df.info(buf=buffer)
        st.text(buffer.getvalue())


        st.subheader("Missing Values Count")
        missing_values = df.isnull().sum()
        st.dataframe(missing_values.rename("Missing Count"))

        st.subheader("Statistical Summary")
        st.dataframe(df.describe())

    else:
        st.warning("Data could not be loaded. Please check the GitHub URL.")


def eda_page(df):
    """Performs and displays Exploratory Data Analysis."""
    st.title("📈 Exploratory Data Analysis (EDA)")

    if df is not None:
        # Drop rows with missing PM2.5 for analysis consistency with modeling
        df_cleaned = df.dropna(subset=['PM2.5'])
        st.write(f"Using {df_cleaned.shape[0]} rows for EDA after dropping missing PM2.5 values.")

        if df_cleaned.empty:
             st.warning("No data available for EDA after dropping missing PM2.5 values.")
             return

        st.subheader("Distribution of Numerical Features")
        numerical_columns = df_cleaned.select_dtypes(include=['float64', 'int64']).columns.tolist()
        # Remove the index column 'NO.' if it exists and PM2.5 as it's the target
        if 'NO.' in numerical_columns:
             numerical_columns.remove('NO.')
        if 'PM2.5' in numerical_columns:
             numerical_columns.remove('PM2.5') # Don't plot distribution of target here

        if numerical_columns:
            selected_numerical_col = st.selectbox("Select a numerical column to plot distribution:", numerical_columns)
            if selected_numerical_col:
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.histplot(df_cleaned[selected_numerical_col], kde=True, bins=30, ax=ax)
                ax.set_title(f'Distribution of {selected_numerical_col}')
                st.pyplot(fig)
        else:
            st.info("No numerical features found for distribution plots.")


        st.subheader("Correlation Heatmap")
        # Select only numerical columns for correlation
        numerical_df = df_cleaned.select_dtypes(include=['number'])
        if not numerical_df.empty:
            correlation_matrix = numerical_df.corr()
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, ax=ax)
            ax.set_title('Correlation Heatmap')
            st.pyplot(fig)
        else:
            st.info("No numerical features found for correlation heatmap.")


        st.subheader("Distribution by Category (Boxplots)")
        categorical_columns = df_cleaned.select_dtypes(include=['object']).columns.tolist()
        if 'Category' in categorical_columns:
            numerical_columns_for_boxplot = df_cleaned.select_dtypes(include=['float64', 'int64']).columns.tolist()
            if 'NO.' in numerical_columns_for_boxplot:
                 numerical_columns_for_boxplot.remove('NO.')

            if numerical_columns_for_boxplot:
                selected_numerical_col_boxplot = st.selectbox("Select a numerical column for boxplot by Category:", numerical_columns_for_boxplot)
                if selected_numerical_col_boxplot:
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.boxplot(data=df_cleaned, x='Category', y=selected_numerical_col_boxplot, ax=ax)
                    ax.set_title(f'Distribution of {selected_numerical_col_boxplot} by Category')
                    st.pyplot(fig)
            else:
                st.info("No numerical features available for boxplots.")
        else:
            st.info("'Category' column not found for boxplots.")


        st.subheader("Pairplot (Sampled Data)")
        st.write("Generating pairplot for a sample of the data due to potential size.")
        # Select a subset of relevant numerical columns for pairplot
        pairplot_cols = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']
        # Filter to only include columns that exist in the dataframe
        existing_pairplot_cols = [col for col in pairplot_cols if col in df_cleaned.columns]

        if 'Category' in df_cleaned.columns and existing_pairplot_cols:
             # Take a sample if the dataset is large
             sample_size = min(1000, df_cleaned.shape[0])
             df_sample = df_cleaned.sample(sample_size, random_state=42)
             # Ensure 'Category' is included if it exists
             cols_for_pairplot = existing_pairplot_cols + ['Category'] if 'Category' in df_sample.columns else existing_pairplot_cols
             fig = sns.pairplot(df_sample[cols_for_pairplot], hue='Category' if 'Category' in df_sample.columns else None)
             st.pyplot(fig)
        elif existing_pairplot_cols:
             # If no category, just plot numerical features
             sample_size = min(1000, df_cleaned.shape[0])
             df_sample = df_cleaned.sample(sample_size, random_state=42)
             fig = sns.pairplot(df_sample[existing_pairplot_cols])
             st.pyplot(fig)
        else:
             st.info("Not enough relevant numerical or 'Category' columns for pairplot.")


    else:
        st.warning("Data could not be loaded. Please check the GitHub URL.")


def modelling_prediction_page(df, model):
    """Handles model loading, prediction, and evaluation."""
    st.title("🧠 Modelling and Prediction")

    if df is not None and model is not None:
        st.subheader("Model Evaluation and Prediction")

        target_column = 'PM2.5'

        if target_column not in df.columns:
            st.error(f"Target column '{target_column}' not found in the loaded data.")
            return

        # Drop rows where the target variable (PM2.5) is missing for evaluation
        data_df_cleaned = df.dropna(subset=[target_column])

        if data_df_cleaned.empty:
             st.warning("No data available for evaluation after dropping missing target values.")
             return

        # Define features (X) and target (y) for evaluation
        features = data_df_cleaned.columns.tolist()
        if 'NO.' in features:
            features.remove('NO.') # Remove index column
        features.remove(target_column)

        X_eval = data_df_cleaned[features]
        y_eval = data_df_cleaned[target_column]

        # Ensure the feature columns in the data match the features the model was trained on
        # This is a crucial check! The columns in X_eval must match the columns
        # the model's preprocessor expects.
        # A robust way is to save the list of training columns and check against it here.
        # For this example, we assume the loaded data has the correct columns.

        try:
            # Make predictions using the loaded model
            y_pred_eval = model.predict(X_eval)

            # Evaluate the model
            mse = mean_squared_error(y_eval, y_pred_eval)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_eval, y_pred_eval)

            st.write("Evaluation Metrics on Loaded Data:")
            st.write(f"Mean Squared Error (MSE): {mse:.4f}")
            st.write(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
            st.write(f"R-squared (R2): {r2:.4f}")

            # --- Display Predictions (Optional) ---
            st.subheader("Sample Predictions vs Actual")
            results_df = pd.DataFrame({'Actual PM2.5': y_eval, 'Predicted PM2.5': y_pred_eval})
            st.dataframe(results_df.head()) # Display first few predictions

            # --- Feature Importance Plot (if model is a pipeline with RF step) ---
            if isinstance(model, Pipeline) and isinstance(model.steps[-1][1], RandomForestRegressor):
                 st.subheader("Feature Importance")
                 try:
                     importances = model.named_steps['model'].feature_importances_

                     # Attempt to get feature names from the preprocessor
                     # This part can be complex depending on the preprocessor structure
                     # Assuming the preprocessor is the first step in the pipeline
                     preprocessor_step = model.named_steps['preprocessor']

                     # Get names of features after preprocessing
                     # This method works for ColumnTransformer
                     try:
                         all_feature_names = preprocessor_step.get_feature_names_out()
                     except AttributeError:
                         st.warning("Could not get processed feature names from the preprocessor. Feature importance plot may show generic names.")
                         # Fallback: create generic names if get_feature_names_out fails
                         all_feature_names = [f'feature_{i}' for i in range(len(importances))]


                     # Sort feature importances
                     indices = np.argsort(importances)[::-1]
                     top_n = min(20, len(all_feature_names)) # Plot top 20 or fewer
                     top_indices = indices[:top_n]

                     fig, ax = plt.subplots(figsize=(12, 8))
                     ax.set_title(f'Top {top_n} Feature Importances')
                     ax.barh(range(top_n), importances[top_indices], align="center")
                     # Ensure feature names list is long enough
                     if len(all_feature_names) > 0:
                         ax.set_yticks(range(top_n))
                         ax.set_yticklabels([all_feature_names[i] for i in top_indices])
                     ax.set_xlabel('Relative Importance')
                     ax.invert_yaxis()
                     st.pyplot(fig)

                 except Exception as e:
                     st.error(f"Error plotting feature importance: {e}")
            else:
                 st.info("Feature importance plot is available only if the loaded model is a scikit-learn Pipeline ending with a RandomForestRegressor.")


        except Exception as e:
            st.error(f"Error during prediction or evaluation: {e}")

    elif df is None:
        st.warning("Data could not be loaded. Please check the GitHub URL.")
    elif model is None:
         st.warning("Model could not be loaded. Please check the Google Drive ID and access permissions.")
    else:
        st.info("Waiting to load data and model...")


# --- Main Application Logic ---

def main():
    """Main function to set up the Streamlit app and navigation."""
    st.set_page_config(
        page_title="Air Quality Prediction App",
        page_icon="💨",
        layout="wide"
    )

    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Data Overview", "Exploratory Data Analysis (EDA)", "Modelling and Prediction"])

    # Load data and model once and pass to page functions
    data_df = load_data(GITHUB_DATA_URL)
    model = load_model(GOOGLE_DRIVE_MODEL_FILE_ID, MODEL_LOCAL_FILENAME)


    # --- Page Routing ---
    if page == "Data Overview":
        data_overview_page(data_df)
    elif page == "Exploratory Data Analysis (EDA)":
        eda_page(data_df)
    elif page == "Modelling and Prediction":
        modelling_prediction_page(data_df, model)

    # --- Footer (Optional) ---
    st.sidebar.markdown("---")
    st.sidebar.write("Developed by Mirukshan") 


if __name__ == "__main__":
    main()

