import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import numpy as np
import os # Import os for file path handling
import gdown # Import gdown for downloading from Google Drive
import datetime # For default year/month/day

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
GOOGLE_DRIVE_MODEL_FILE_ID = "1ilukwSmeVZ7ywBCBt-9LS35JmoYGzMbe" # Example ID, replace with your actual ID
MODEL_LOCAL_FILENAME = "random_forest_regressor_model.joblib"
DATA_FILENAME = "merged_data.csv"

# --- Data Loading Function (Cached) ---
@st.cache_data
def load_data(file_path):
    """Loads data from a local CSV file."""
    try:
        if not os.path.exists(file_path):
            st.error(f"Error: Data file '{file_path}' not found. Please make sure it's in the correct location.")
            return None
        df = pd.read_csv(file_path)
        st.success(f"Data loaded successfully from '{file_path}'!")
        return df
    except Exception as e:
        st.error(f"Error loading data from {file_path}: {e}")
        return None

# --- Model Loading Function (Cached) ---
@st.cache_resource
def load_model(file_id, output_path):
    """Downloads (if needed) and loads the model."""
    try:
        if not os.path.exists(output_path):
            st.info(f"Model file '{output_path}' not found locally.")
            if file_id:
                st.info(f"Downloading model from Google Drive (ID: {file_id})...")
                gdown.download(id=file_id, output=output_path, quiet=False)
                st.success("Model downloaded successfully!")
            else:
                st.error("Google Drive File ID not provided and model not found locally.")
                return None
        else:
            st.info(f"Model file '{output_path}' found locally. Loading...")
        model = joblib.load(output_path)
        st.success("Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        if file_id:
            st.warning("Please ensure the Google Drive file ID is correct and the file is publicly accessible if downloading.")
        return None

# --- Page Functions ---

def data_overview_page(df):
    """Displays data overview information."""
    st.title("📊 Data Overview")
    if df is not None:
        st.subheader("Dataset Shape")
        st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
        st.subheader("First 5 Rows")
        st.dataframe(df.head())
        st.subheader("Data Types and Missing Values")
        st.text("DataFrame Info (Data Types and Non-Null Counts):")
        buffer = io.StringIO()
        df.info(buf=buffer)
        st.text(buffer.getvalue())
        st.subheader("Missing Values Count")
        missing_values = df.isnull().sum()
        st.dataframe(missing_values.rename("Missing Count"))
        st.subheader("Statistical Summary")
        st.dataframe(df.describe(include='all'))
    else:
        st.warning("Data could not be loaded. Please check the data file path and ensure it exists.")

def eda_page(df):
    """Performs and displays Exploratory Data Analysis."""
    st.title("🔍 Exploratory Data Analysis (EDA)")
    if df is not None:
        target_column_name = 'PM2.5'
        if target_column_name not in df.columns:
            st.error(f"Target column '{target_column_name}' not found in the data for EDA. Please check your CSV.")
            return

        df_cleaned = df.dropna(subset=[target_column_name])
        st.write(f"Using {df_cleaned.shape[0]} rows for EDA after dropping missing {target_column_name} values.")
        if df_cleaned.empty:
             st.warning(f"No data available for EDA after dropping missing {target_column_name} values.")
             return

        st.subheader("Distribution of Numerical Features")
        numerical_columns = df_cleaned.select_dtypes(include=np.number).columns.tolist()
        if 'NO.' in numerical_columns: numerical_columns.remove('NO.')
        if target_column_name in numerical_columns: numerical_columns.remove(target_column_name)
        if numerical_columns:
            selected_numerical_col = st.selectbox("Select a numerical column to plot distribution:", numerical_columns)
            if selected_numerical_col:
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.histplot(df_cleaned[selected_numerical_col], kde=True, bins=30, ax=ax)
                ax.set_title(f'Distribution of {selected_numerical_col}')
                st.pyplot(fig)
        else:
            st.info("No numerical features found for distribution plots (excluding target/ID).")

        st.subheader("Correlation Heatmap")
        numerical_df_for_corr = df_cleaned.select_dtypes(include=[np.number])
        if not numerical_df_for_corr.empty and len(numerical_df_for_corr.columns) > 1:
            correlation_matrix = numerical_df_for_corr.corr()
            fig, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, ax=ax)
            ax.set_title('Correlation Heatmap of Numerical Features')
            st.pyplot(fig)
        else:
            st.info("Not enough numerical features for a correlation heatmap.")

        categorical_col_for_boxplot = 'Category'
        if categorical_col_for_boxplot in df_cleaned.columns:
            st.subheader(f"Distribution by {categorical_col_for_boxplot} (Boxplots)")
            numerical_cols_for_boxplot = df_cleaned.select_dtypes(include=np.number).columns.tolist()
            if target_column_name in numerical_cols_for_boxplot: pass
            if 'NO.' in numerical_cols_for_boxplot: numerical_cols_for_boxplot.remove('NO.')
            if numerical_cols_for_boxplot:
                selected_numerical_col_boxplot = st.selectbox(
                    f"Select a numerical column for boxplot by {categorical_col_for_boxplot}:",
                    numerical_cols_for_boxplot,
                    index=numerical_cols_for_boxplot.index(target_column_name) if target_column_name in numerical_cols_for_boxplot else 0
                )
                if selected_numerical_col_boxplot:
                    fig, ax = plt.subplots(figsize=(10, 7))
                    sns.boxplot(data=df_cleaned, x=categorical_col_for_boxplot, y=selected_numerical_col_boxplot, ax=ax)
                    ax.set_title(f'Distribution of {selected_numerical_col_boxplot} by {categorical_col_for_boxplot}')
                    plt.xticks(rotation=45, ha='right')
                    st.pyplot(fig)
            else:
                st.info(f"No numerical features available for boxplots by {categorical_col_for_boxplot}.")
        else:
            st.info(f"'{categorical_col_for_boxplot}' column not found for boxplots. Adjust column name if needed.")

        st.subheader("Pairplot (Sampled Data)")
        st.write("Generating pairplot for a sample of the data due to potential size.")
        pairplot_cols_options = df_cleaned.select_dtypes(include=[np.number]).columns.tolist()
        if target_column_name not in pairplot_cols_options and target_column_name in df_cleaned.columns:
            pairplot_cols_options.append(target_column_name)
        default_pairplot_cols = [col for col in ['PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM', target_column_name] if col in df_cleaned.columns]
        selected_pairplot_cols = st.multiselect("Select columns for Pairplot:", pairplot_cols_options, default=default_pairplot_cols[:5])
        hue_col_pairplot = None
        if categorical_col_for_boxplot in df_cleaned.columns:
            if st.checkbox(f"Use '{categorical_col_for_boxplot}' as hue for Pairplot?", key="pairplot_hue_checkbox"):
                hue_col_pairplot = categorical_col_for_boxplot
        if selected_pairplot_cols and len(selected_pairplot_cols) > 1:
             sample_size = min(500, df_cleaned.shape[0])
             df_sample = df_cleaned.sample(sample_size, random_state=42)
             pairplot_display_cols_with_hue = selected_pairplot_cols + ([hue_col_pairplot] if hue_col_pairplot and hue_col_pairplot not in selected_pairplot_cols else [])
             st.write(f"Generating pairplot for: {', '.join(selected_pairplot_cols)}" + (f" with hue '{hue_col_pairplot}'" if hue_col_pairplot else ""))
             fig = sns.pairplot(df_sample[pairplot_display_cols_with_hue], hue=hue_col_pairplot, diag_kind='kde', corner=True)
             st.pyplot(fig)
        else:
             st.info("Please select at least two numerical columns for the pairplot.")
    else:
        st.warning("Data could not be loaded. EDA cannot be performed.")

def modelling_prediction_page(df, model):
    """Handles model evaluation and PM2.5 prediction based on user inputs."""
    st.title("🧠 Modelling and Prediction")
    st.write("Evaluate the model on loaded data and predict PM2.5 levels using custom inputs.")

    if df is None:
        st.warning("Data could not be loaded. Cannot perform evaluation or prediction.")
        return
    if model is None:
         st.warning("Model could not be loaded. Cannot perform evaluation or prediction.")
         return

    # --- Model Evaluation (on loaded data) ---
    st.subheader("Model Evaluation on Loaded Data")
    target_column = 'PM2.5'

    if target_column not in df.columns:
        st.error(f"Target column '{target_column}' not found in the loaded data. Cannot evaluate or predict.")
        return

    # Use a distinct variable for data cleaned for evaluation
    data_df_cleaned_for_eval = df.dropna(subset=[target_column])
    if data_df_cleaned_for_eval.empty:
         st.warning(f"No data available for evaluation/prediction after dropping rows with missing '{target_column}'.")
         return

    features_for_eval = [col for col in data_df_cleaned_for_eval.columns if col != target_column]
    if 'NO.' in features_for_eval: # Assuming 'NO.' is an index or ID column
        features_for_eval.remove('NO.')
    
    if not features_for_eval:
        st.error("No feature columns found in loaded data (after excluding target and ID). Cannot evaluate model.")
        X_eval = pd.DataFrame() # Empty dataframe
        y_eval = pd.Series(dtype='float64')
    else:
        X_eval = data_df_cleaned_for_eval[features_for_eval]
        y_eval = data_df_cleaned_for_eval[target_column]

    if not X_eval.empty:
        try:
            y_pred_eval = model.predict(X_eval)
            mse = mean_squared_error(y_eval, y_pred_eval)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_eval, y_pred_eval)

            st.write("Evaluation Metrics (on loaded data, after cleaning):")
            col1, col2, col3 = st.columns(3)
            col1.metric(label="Mean Squared Error (MSE)", value=f"{mse:.4f}")
            col2.metric(label="Root Mean Squared Error (RMSE)", value=f"{rmse:.4f}")
            col3.metric(label="R-squared (R²)", value=f"{r2:.4f}")

            st.subheader("Sample Predictions vs Actual (from loaded data)")
            results_df = pd.DataFrame({'Actual PM2.5': y_eval, 'Predicted PM2.5': y_pred_eval})
            st.dataframe(results_df.sample(min(10, len(results_df))).reset_index(drop=True))

            # Feature Importance Plot
            if isinstance(model, Pipeline) and hasattr(model.steps[-1][1], 'feature_importances_'):
                 st.subheader("Feature Importance (from model)")
                 try:
                     importances = model.named_steps['model'].feature_importances_
                     processed_feature_names = None
                     if hasattr(model.named_steps.get('preprocessor'), 'get_feature_names_out'):
                         try:
                             processed_feature_names = model.named_steps['preprocessor'].get_feature_names_out()
                         except Exception as e_feat_names:
                             st.warning(f"Could not get feature names from preprocessor: {e_feat_names}. Using generic names.")
                     
                     if processed_feature_names is None or len(processed_feature_names) != len(importances):
                         processed_feature_names = [f'feature_{i}' for i in range(len(importances))]
                     
                     if processed_feature_names and len(processed_feature_names) == len(importances):
                         indices = np.argsort(importances)[::-1]
                         top_n = min(20, len(processed_feature_names))
                         top_indices = indices[:top_n]
                         fig, ax = plt.subplots(figsize=(12, max(6, top_n * 0.4)))
                         ax.set_title(f'Top {top_n} Feature Importances')
                         bars = ax.barh(range(top_n), importances[top_indices], align="center")
                         ax.set_yticks(range(top_n))
                         ax.set_yticklabels([processed_feature_names[i] for i in top_indices])
                         ax.invert_yaxis()
                         ax.set_xlabel('Relative Importance')
                         for bar in bars:
                             ax.text(bar.get_width(), bar.get_y() + bar.get_height()/2, f'{bar.get_width():.3f}', va='center', ha='left')
                         st.pyplot(fig)
                 except Exception as e_imp:
                     st.error(f"Error plotting feature importance: {e_imp}")
            else:
                 st.info("Feature importance plot is available if the model is a scikit-learn Pipeline with a final estimator that has 'feature_importances_'.")
        except Exception as e_eval:
            st.error(f"Error during model evaluation on loaded data: {e_eval}")
            st.warning("This could happen if the loaded data's columns don't match what the model's preprocessor expects (e.g., missing 'year', 'month', 'day', 'station' if model was trained on them and they are not in merged_data.csv).")
    else:
        st.info("Skipping model evaluation as no valid features were found in the loaded data.")


    # --- Prediction Section (User Input) ---
    st.divider()
    st.subheader("🎯 Make a PM2.5 Prediction")
    st.write("Enter the values for the features below to get a PM2.5 prediction.")

    prediction_input_features = ['year', 'month', 'day', 'hour', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP', 'RAIN', 'wd', 'WSPM', 'station', 'Category']
    st.caption(f"Input fields for prediction: {', '.join(prediction_input_features)}")

    input_data = {}
    num_input_cols = 3 
    input_streamlit_cols = st.columns(num_input_cols)
    feature_idx = 0
    
    # Use data_df_cleaned_for_eval for populating defaults/options if columns exist
    # This is the cleaned version of the loaded data.
    df_for_defaults = data_df_cleaned_for_eval 

    for feature_name in prediction_input_features:
        current_streamlit_col = input_streamlit_cols[feature_idx % num_input_cols]
        feature_present_in_data = feature_name in df_for_defaults.columns
        feature_series = df_for_defaults[feature_name] if feature_present_in_data else pd.Series(dtype='object') # Empty series if not present

        # Specific handling for each feature type
        if feature_name == 'year':
            current_year = datetime.date.today().year
            default_val = int(feature_series.mean()) if feature_present_in_data and pd.notna(feature_series.mean()) else current_year
            input_data[feature_name] = current_streamlit_col.number_input(
                label="Year", value=default_val, min_value=current_year - 20, max_value=current_year + 5, step=1, format="%d", key=f"num_{feature_name}"
            )
        elif feature_name == 'month':
            default_val = int(feature_series.mean()) if feature_present_in_data and pd.notna(feature_series.mean()) else datetime.date.today().month
            default_val = max(1, min(int(default_val), 12))
            input_data[feature_name] = current_streamlit_col.slider(
                label="Month", min_value=1, max_value=12, value=default_val, step=1, key=f"slider_{feature_name}"
            )
        elif feature_name == 'day':
            default_val = int(feature_series.mean()) if feature_present_in_data and pd.notna(feature_series.mean()) else datetime.date.today().day
            default_val = max(1, min(int(default_val), 31))
            input_data[feature_name] = current_streamlit_col.slider(
                label="Day", min_value=1, max_value=31, value=default_val, step=1, key=f"slider_{feature_name}"
            )
        elif feature_name == 'hour':
            default_val = int(feature_series.mean()) if feature_present_in_data and pd.notna(feature_series.mean()) else 12
            default_val = max(0, min(int(default_val), 23))
            input_data[feature_name] = current_streamlit_col.slider(
                label="Hour of Day", min_value=0, max_value=23, value=default_val, step=1, key=f"slider_{feature_name}"
            )
        elif feature_name in ['PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']:
            default_val_num = 0.0
            if feature_present_in_data and pd.api.types.is_numeric_dtype(feature_series) and pd.notna(feature_series.mean()):
                default_val_num = float(feature_series.mean())
            else: # Generic defaults if column not in data or not numeric
                generic_defaults = {'PM10': 50.0, 'SO2': 10.0, 'NO2': 30.0, 'CO': 0.8, 'O3': 60.0, 'TEMP': 15.0, 'PRES': 1012.0, 'DEWP': 5.0, 'RAIN': 0.0, 'WSPM': 2.0}
                default_val_num = generic_defaults.get(feature_name, 0.0)
            input_data[feature_name] = current_streamlit_col.number_input(
                label=f"{feature_name}", value=default_val_num, format="%.2f", key=f"num_{feature_name}"
            )
        elif feature_name in ['wd', 'station', 'Category']:
            unique_vals = []
            default_cat_index = 0
            if feature_present_in_data:
                unique_vals = sorted(list(set(str(val) for val in feature_series.unique() if pd.notna(val))))
                if unique_vals and not feature_series.mode().empty:
                    try:
                        mode_val = str(feature_series.mode()[0])
                        if mode_val in unique_vals:
                            default_cat_index = unique_vals.index(mode_val)
                    except: pass # Keep default_cat_index = 0
            
            if not unique_vals: # Fallback options
                if feature_name == 'wd': unique_vals = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW', 'NNE', 'ENE', 'ESE', 'SSE', 'SSW', 'WSW', 'WNW', 'NNW', 'CALM']
                elif feature_name == 'station': unique_vals = [f"Station{chr(65+i)}" for i in range(5)] # StationA to StationE
                elif feature_name == 'Category': unique_vals = ['Urban', 'Suburban', 'Rural', 'Industrial', 'Traffic']
            
            if unique_vals:
                input_data[feature_name] = current_streamlit_col.selectbox(
                    label=f"{feature_name}", options=unique_vals, index=default_cat_index, key=f"select_{feature_name}"
                )
            else: # Should be rare with fallbacks
                 input_data[feature_name] = current_streamlit_col.text_input(label=f"{feature_name} (Enter manually)", key=f"text_{feature_name}")
        else: # Should not happen if prediction_input_features is exhaustive
            input_data[feature_name] = current_streamlit_col.text_input(label=f"Unknown feature: {feature_name}", key=f"unknown_{feature_name}")

        feature_idx += 1

    if st.button("Predict PM2.5", key="predict_button"):
        valid_inputs = True
        for feature_name in prediction_input_features:
            if input_data.get(feature_name) is None :
                st.error(f"Input for '{feature_name}' is missing or invalid.")
                valid_inputs = False
                break
        
        if valid_inputs:
            try:
                input_df = pd.DataFrame([input_data])[prediction_input_features] # Ensure correct order
                
                # Attempt to convert columns to numeric where appropriate, if they are not already
                # This is a safeguard. Ideally, Streamlit inputs provide correct types.
                for col in ['year', 'month', 'day', 'hour', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']:
                    if col in input_df.columns:
                        input_df[col] = pd.to_numeric(input_df[col], errors='coerce')

                # Check for NaNs after conversion (e.g. if a text input for a number was non-numeric)
                if input_df.isnull().any().any():
                    st.error("Some numerical inputs could not be converted to numbers. Please check your entries.")
                else:
                    prediction = model.predict(input_df)
                    st.success(f"Predicted PM2.5 Concentration: **{prediction[0]:.2f} µg/m³**")

            except Exception as e_pred:
                st.error(f"An error occurred during prediction: {e_pred}")
                st.warning("Ensure all input values are valid. The model's preprocessor might also expect specific data types or ranges.")
                if 'input_df' in locals():
                    st.caption("Data sent for prediction (first row):")
                    st.dataframe(input_df.head(1))
                    st.caption("Data types of input sent:")
                    st.text(input_df.dtypes)

# --- Main Application Logic ---
def main():
    st.set_page_config(page_title="Air Quality Prediction App", page_icon="🌬️", layout="wide")
    st.sidebar.title("Navigation")
    page_options = ["Data Overview", "Exploratory Data Analysis (EDA)", "Modelling and Prediction"]
    page = st.sidebar.radio("Go to", page_options, key="nav_radio")

    data_df = load_data(DATA_FILENAME)
    model = load_model(GOOGLE_DRIVE_MODEL_FILE_ID, MODEL_LOCAL_FILENAME)

    if page == "Data Overview":
        data_overview_page(data_df)
    elif page == "Exploratory Data Analysis (EDA)":
        eda_page(data_df)
    elif page == "Modelling and Prediction":
        modelling_prediction_page(data_df, model)

    st.sidebar.markdown("---")
    st.sidebar.info("Air Quality App v1.2")

if __name__ == "__main__":
    main()
