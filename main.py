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
        # For EDA, explicitly remove 'No' or 'NO.' if they exist, as they are likely IDs
        numerical_columns = df_cleaned.select_dtypes(include=np.number).columns.tolist()
        ids_to_remove_eda = ['NO.', 'No', target_column_name] # Add variations if needed
        numerical_columns = [col for col in numerical_columns if col not in ids_to_remove_eda]

        if numerical_columns:
            selected_numerical_col = st.selectbox("Select a numerical column to plot distribution:", numerical_columns, key="eda_num_dist_select")
            if selected_numerical_col:
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.histplot(df_cleaned[selected_numerical_col], kde=True, bins=30, ax=ax)
                ax.set_title(f'Distribution of {selected_numerical_col}')
                st.pyplot(fig)
        else:
            st.info("No numerical features (excluding target/IDs) found for distribution plots.")

        st.subheader("Correlation Heatmap")
        # For correlation, also exclude ID columns
        numerical_df_for_corr = df_cleaned.select_dtypes(include=[np.number])
        cols_to_drop_corr = [col for col in ['NO.', 'No'] if col in numerical_df_for_corr.columns]
        if cols_to_drop_corr:
            numerical_df_for_corr = numerical_df_for_corr.drop(columns=cols_to_drop_corr)

        if not numerical_df_for_corr.empty and len(numerical_df_for_corr.columns) > 1:
            correlation_matrix = numerical_df_for_corr.corr()
            fig, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, ax=ax)
            ax.set_title('Correlation Heatmap of Numerical Features')
            st.pyplot(fig)
        else:
            st.info("Not enough numerical features for a correlation heatmap (after excluding IDs).")

        categorical_col_for_boxplot = 'Category' # Example, change if needed
        if categorical_col_for_boxplot in df_cleaned.columns:
            st.subheader(f"Distribution by {categorical_col_for_boxplot} (Boxplots)")
            numerical_cols_for_boxplot = df_cleaned.select_dtypes(include=np.number).columns.tolist()
            # Keep target for this plot, remove IDs
            ids_to_remove_boxplot = ['NO.', 'No']
            numerical_cols_for_boxplot = [col for col in numerical_cols_for_boxplot if col not in ids_to_remove_boxplot]

            if numerical_cols_for_boxplot:
                default_boxplot_col = target_column_name if target_column_name in numerical_cols_for_boxplot else numerical_cols_for_boxplot[0]
                selected_numerical_col_boxplot = st.selectbox(
                    f"Select a numerical column for boxplot by {categorical_col_for_boxplot}:",
                    numerical_cols_for_boxplot,
                    index=numerical_cols_for_boxplot.index(default_boxplot_col),
                    key="eda_boxplot_select"
                )
                if selected_numerical_col_boxplot:
                    fig, ax = plt.subplots(figsize=(10, 7))
                    sns.boxplot(data=df_cleaned, x=categorical_col_for_boxplot, y=selected_numerical_col_boxplot, ax=ax)
                    ax.set_title(f'Distribution of {selected_numerical_col_boxplot} by {categorical_col_for_boxplot}')
                    plt.xticks(rotation=45, ha='right')
                    st.pyplot(fig)
            else:
                st.info(f"No numerical features available for boxplots by {categorical_col_for_boxplot} (after excluding IDs).")
        else:
            st.info(f"'{categorical_col_for_boxplot}' column not found in the loaded data. Adjust column name if needed or ensure it's in your CSV.")

        st.subheader("Pairplot (Sampled Data)")
        st.write("Generating pairplot for a sample of the data due to potential size.")
        pairplot_cols_options = df_cleaned.select_dtypes(include=[np.number]).columns.tolist()
        # Exclude IDs from pairplot options
        ids_to_remove_pairplot = ['NO.', 'No']
        pairplot_cols_options = [col for col in pairplot_cols_options if col not in ids_to_remove_pairplot]

        if target_column_name not in pairplot_cols_options and target_column_name in df_cleaned.columns:
            pairplot_cols_options.append(target_column_name) # Ensure target is an option if not already numeric

        default_pairplot_cols = [col for col in ['PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM', target_column_name] if col in pairplot_cols_options]
        
        selected_pairplot_cols = st.multiselect("Select columns for Pairplot:", pairplot_cols_options, default=default_pairplot_cols[:5], key="eda_pairplot_multiselect")
        hue_col_pairplot = None
        if categorical_col_for_boxplot in df_cleaned.columns: 
            if st.checkbox(f"Use '{categorical_col_for_boxplot}' as hue for Pairplot?", key="pairplot_hue_checkbox"):
                hue_col_pairplot = categorical_col_for_boxplot
        if selected_pairplot_cols and len(selected_pairplot_cols) > 1:
             sample_size = min(500, df_cleaned.shape[0])
             df_sample = df_cleaned.sample(sample_size, random_state=42)
             pairplot_display_cols_with_hue = selected_pairplot_cols + ([hue_col_pairplot] if hue_col_pairplot and hue_col_pairplot not in selected_pairplot_cols else [])
             
             # Ensure all columns for pairplot exist in df_sample
             pairplot_display_cols_with_hue = [col for col in pairplot_display_cols_with_hue if col in df_sample.columns]

             if pairplot_display_cols_with_hue and len(pairplot_display_cols_with_hue) >1 :
                st.write(f"Generating pairplot for: {', '.join(selected_pairplot_cols)}" + (f" with hue '{hue_col_pairplot}'" if hue_col_pairplot and hue_col_pairplot in pairplot_display_cols_with_hue else ""))
                fig = sns.pairplot(df_sample[pairplot_display_cols_with_hue], hue=hue_col_pairplot if hue_col_pairplot in pairplot_display_cols_with_hue else None, diag_kind='kde', corner=True)
                st.pyplot(fig)
             else:
                st.info("Not enough valid columns selected for pairplot after filtering.")
        else:
             st.info("Please select at least two numerical columns for the pairplot.")
    else:
        st.warning("Data could not be loaded. EDA cannot be performed.")

def modelling_prediction_page(df, model):
    """Handles model evaluation and PM2.5 prediction based on user inputs."""
    st.title("🧠 Modelling and Prediction")
    st.write("Evaluate the model on loaded data and predict PM2.5 levels using custom inputs.")

    if df is None: 
        st.warning("Data (merged_data.csv) could not be loaded. Cannot perform evaluation or prediction.")
        return
    if model is None:
         st.warning("Model could not be loaded. Cannot perform evaluation or prediction.")
         return

    # --- Model Evaluation (on loaded data) ---
    st.subheader("Model Evaluation on Loaded Data")
    target_column = 'PM2.5'

    if target_column not in df.columns: 
        st.error(f"Target column '{target_column}' not found in the loaded data (merged_data.csv). Cannot evaluate or predict.")
        return

    data_df_cleaned_for_eval = df.dropna(subset=[target_column])
    if data_df_cleaned_for_eval.empty:
         st.warning(f"No data available for evaluation/prediction after dropping rows with missing '{target_column}' from merged_data.csv.")
         return

    # For evaluation, features are derived from the loaded CSV.
    # The model might expect 'No' if it was trained with it.
    # If 'No' is in df.columns, it should be part of X_eval if the model expects it.
    # If 'NO.' (with period) was used during training and removed, that's different.
    # The key is that X_eval's columns must match what the *preprocessor inside the model* expects.
    
    features_for_eval = [col for col in data_df_cleaned_for_eval.columns if col != target_column]
    # If 'NO.' (with period) was explicitly removed during training and is NOT what the model expects,
    # but 'No' (without period) IS expected, then we should NOT remove 'No' here.
    # The error message implies 'No' is expected.
    # We only remove 'NO.' if it's an explicit index not used by the model.
    if 'NO.' in features_for_eval: # This removes 'NO.' (with period) if it exists
        features_for_eval.remove('NO.')
    
    if not features_for_eval:
        st.error("No feature columns found in loaded data (merged_data.csv, after excluding target and potentially 'NO.'). Cannot evaluate model.")
        X_eval = pd.DataFrame()
        y_eval = pd.Series(dtype='float64')
    else:
        X_eval = data_df_cleaned_for_eval[features_for_eval]
        y_eval = data_df_cleaned_for_eval[target_column]

    if not X_eval.empty:
        try:
            # Check if X_eval is missing 'No' when the model expects it.
            # This check is a bit indirect. The model.predict will fail if columns are truly mismatched.
            st.write(f"Columns being sent for model evaluation: {X_eval.columns.tolist()}")
            y_pred_eval = model.predict(X_eval) # This line will error if X_eval is missing 'No' and model expects it.
            mse = mean_squared_error(y_eval, y_pred_eval)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_eval, y_pred_eval)

            st.write("Evaluation Metrics (on loaded data, after cleaning):")
            col1_eval, col2_eval, col3_eval = st.columns(3)
            col1_eval.metric(label="Mean Squared Error (MSE)", value=f"{mse:.4f}")
            col2_eval.metric(label="Root Mean Squared Error (RMSE)", value=f"{rmse:.4f}")
            col3_eval.metric(label="R-squared (R²)", value=f"{r2:.4f}")

            st.subheader("Sample Predictions vs Actual (from loaded data)")
            results_df = pd.DataFrame({'Actual PM2.5': y_eval, 'Predicted PM2.5': y_pred_eval})
            st.dataframe(results_df.sample(min(10, len(results_df))).reset_index(drop=True))

            if isinstance(model, Pipeline) and hasattr(model.steps[-1][1], 'feature_importances_'):
                 st.subheader("Feature Importance (from model)")
                 try:
                     importances = model.named_steps['model'].feature_importances_
                     processed_feature_names = None
                     if hasattr(model.named_steps.get('preprocessor'), 'get_feature_names_out'):
                         try:
                             processed_feature_names = model.named_steps['preprocessor'].get_feature_names_out()
                         except Exception as e_feat_names:
                             st.warning(f"Could not get feature names from preprocessor for importance plot: {e_feat_names}. Using generic names.")
                     
                     if processed_feature_names is None or len(processed_feature_names) != len(importances):
                         # Fallback if names can't be retrieved or length mismatch
                         st.warning("Feature names from preprocessor did not match importance count. Using generic feature names for plot.")
                         processed_feature_names = X_eval.columns.tolist() # Use columns from X_eval as a fallback
                         if len(processed_feature_names) != len(importances): # If still mismatch, use generic
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
                         for bar_val in bars: 
                             ax.text(bar_val.get_width(), bar_val.get_y() + bar_val.get_height()/2, f'{bar_val.get_width():.3f}', va='center', ha='left')
                         st.pyplot(fig)
                 except Exception as e_imp:
                     st.error(f"Error plotting feature importance: {e_imp}")
            else:
                 st.info("Feature importance plot is available if the model is a scikit-learn Pipeline with a final estimator that has 'feature_importances_'.")
        except Exception as e_eval:
            st.error(f"Error during model evaluation on loaded data: {e_eval}")
            st.warning("This could happen if the loaded data's columns (after processing) don't exactly match what the model's preprocessor expects. Check if 'No' is correctly handled for evaluation based on your model training.")
    else:
        st.info("Skipping model evaluation as no valid features were found in the loaded data.")

    # --- Prediction Section (User Input - Hardcoded Approach) ---
    st.divider()
    st.subheader("🎯 Make a PM2.5 Prediction")
    st.write("Enter the values for the features below to get a PM2.5 prediction.")

    # Add 'No' to the list of features the model expects for prediction.
    prediction_input_features_ordered = ['No', 'year', 'month', 'day', 'hour', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP', 'RAIN', 'wd', 'WSPM', 'station', 'Category']
    st.caption(f"Input fields for prediction: {', '.join(prediction_input_features_ordered)}")

    input_data = {}
    df_for_options = df 

    st.markdown("#### Identifier and Date/Time") # Group 'No' with date/time
    # Create 5 columns for No, Year, Month, Day, Hour
    cols_id_datetime = st.columns(5) 
    input_data['No'] = cols_id_datetime[0].number_input('No (Identifier)', value=1, min_value=0, step=1, format="%d", key="no_input")
    current_year = datetime.date.today().year
    input_data['year'] = cols_id_datetime[1].number_input('Year', min_value=current_year - 20, max_value=current_year + 5, value=current_year, step=1, format="%d", key="year_input")
    input_data['month'] = cols_id_datetime[2].slider('Month', 1, 12, datetime.date.today().month, key="month_slider")
    input_data['day'] = cols_id_datetime[3].slider('Day', 1, 31, datetime.date.today().day, key="day_slider")
    input_data['hour'] = cols_id_datetime[4].slider('Hour', 0, 23, 12, key="hour_slider")


    st.markdown("#### Pollutant Levels (µg/m³ or specific units)")
    cols_pollutants1 = st.columns(3)
    input_data['PM10'] = cols_pollutants1[0].number_input('PM10 (µg/m³)', value=50.0, format="%.2f", key="pm10_input")
    input_data['SO2'] = cols_pollutants1[1].number_input('SO2 (µg/m³)', value=10.0, format="%.2f", key="so2_input")
    input_data['NO2'] = cols_pollutants1[2].number_input('NO2 (µg/m³)', value=30.0, format="%.2f", key="no2_input")
    
    cols_pollutants2 = st.columns(3)
    input_data['CO'] = cols_pollutants2[0].number_input('CO (mg/m³)', value=0.8, format="%.2f", key="co_input") 
    input_data['O3'] = cols_pollutants2[1].number_input('O3 (µg/m³)', value=60.0, format="%.2f", key="o3_input")
    
    st.markdown("#### Meteorological Conditions")
    cols_meteo1 = st.columns(3)
    input_data['TEMP'] = cols_meteo1[0].number_input('Temperature (°C)', value=15.0, format="%.1f", key="temp_input")
    input_data['PRES'] = cols_meteo1[1].number_input('Pressure (hPa)', value=1012.0, format="%.1f", key="pres_input")
    input_data['DEWP'] = cols_meteo1[2].number_input('Dewpoint (°C)', value=5.0, format="%.1f", key="dewp_input")

    cols_meteo2 = st.columns(3) 
    input_data['WSPM'] = cols_meteo2[0].number_input('Wind Speed (m/s)', value=2.0, format="%.1f", key="wspm_input")
    input_data['RAIN'] = cols_meteo2[1].number_input('Rain (mm/h)', value=0.0, format="%.1f", key="rain_input")
    
    wd_options = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW', 'NNE', 'ENE', 'ESE', 'SSE', 'SSW', 'WSW', 'WNW', 'NNW', 'CALM', 'Variable'] 
    if df_for_options is not None and 'wd' in df_for_options.columns:
        unique_wd = sorted(list(set(str(val) for val in df_for_options['wd'].unique() if pd.notna(val))))
        if unique_wd: wd_options = unique_wd
    input_data['wd'] = cols_meteo2[2].selectbox('Wind Direction (wd)', wd_options, key="wd_select")

    st.markdown("#### Location Information")
    cols_location = st.columns(2)
    station_options = [f"Station_{i}" for i in range(1, 13)] 
    if df_for_options is not None and 'station' in df_for_options.columns:
        unique_stations = sorted(list(set(str(val) for val in df_for_options['station'].unique() if pd.notna(val))))
        if unique_stations: station_options = unique_stations
    input_data['station'] = cols_location[0].selectbox('Monitoring Station', station_options, key="station_select")

    category_options = ['Urban', 'Suburban', 'Rural', 'Industrial', 'Traffic', 'Residential', 'Background'] 
    if df_for_options is not None and 'Category' in df_for_options.columns: 
        unique_categories = sorted(list(set(str(val) for val in df_for_options['Category'].unique() if pd.notna(val))))
        if unique_categories: category_options = unique_categories
    elif df_for_options is not None and 'category' in df_for_options.columns: 
        unique_categories = sorted(list(set(str(val) for val in df_for_options['category'].unique() if pd.notna(val))))
        if unique_categories: category_options = unique_categories
    input_data['Category'] = cols_location[1].selectbox('Location Category', category_options, key="category_select")

    if st.button("Predict PM2.5", key="predict_button_hardcoded"):
        try:
            input_df = pd.DataFrame([input_data])[prediction_input_features_ordered]
            numerical_cols_for_conversion = ['No', 'year', 'month', 'day', 'hour', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']
            for col in numerical_cols_for_conversion:
                if col in input_df.columns:
                    input_df[col] = pd.to_numeric(input_df[col], errors='coerce')

            if input_df.isnull().any().any():
                st.error("Some numerical inputs could not be converted to numbers or are missing. Please check your entries.")
                nan_cols = input_df.columns[input_df.isnull().any()].tolist()
                st.text(f"Problematic columns: {', '.join(nan_cols)}")
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
    st.sidebar.info("Air Quality App v1.4") # Incremented version

if __name__ == "__main__":
    main()
