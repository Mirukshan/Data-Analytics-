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
# Replace with the public shareable ID of your trained model file on Google Drive
GOOGLE_DRIVE_MODEL_FILE_ID = "1ilukwSmeVZ7ywBCBt-9LS35JmoYGzMbe" # Example ID, replace with your actual ID
MODEL_LOCAL_FILENAME = "random_forest_regressor_model.joblib"
# Assumes 'merged_data.csv' is in the same directory or accessible via this path
DATA_FILENAME = "merged_data.csv"

# --- Data Loading Function (Cached) ---
@st.cache_data # Cache the data loading for performance
def load_data(file_path):
    """Loads data from a local CSV file."""
    try:
        # Check if the data file exists
        if not os.path.exists(file_path):
            st.error(f"Error: Data file '{file_path}' not found. Please make sure it's in the correct location.")
            # Attempt to download if a placeholder URL was intended (example, not active)
            # st.info(f"Attempting to download from a default URL as fallback...")
            # example_url = "https://raw.githubusercontent.com/plotly/datasets/master/auto-mpg.csv" # Replace with actual if needed
            # df = pd.read_csv(example_url)
            # df.to_csv(file_path, index=False) # Save it locally for next time
            # st.success(f"Data downloaded and saved as {file_path}!")
            return None

        df = pd.read_csv(file_path)
        st.success(f"Data loaded successfully from '{file_path}'!")
        return df
    except Exception as e:
        st.error(f"Error loading data from {file_path}: {e}")
        return None

# --- Model Loading Function (Cached) ---
@st.cache_resource # Cache the model loading as it's a resource
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
        st.dataframe(df.describe(include='all')) # include='all' for mixed types
    else:
        st.warning("Data could not be loaded. Please check the data file path and ensure it exists.")


def eda_page(df):
    """Performs and displays Exploratory Data Analysis."""
    st.title("🔍 Exploratory Data Analysis (EDA)")

    if df is not None:
        # Drop rows with missing PM2.5 for analysis consistency with modeling
        # Ensure 'PM2.5' is the correct target column name in your CSV
        target_column_name = 'PM2.5' # Or whatever your target column is named
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
        # Remove ID-like columns or the target itself if desired for individual distribution plots
        if 'NO.' in numerical_columns: # Assuming 'NO.' is an index or ID
             numerical_columns.remove('NO.')
        if target_column_name in numerical_columns:
             numerical_columns.remove(target_column_name)

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
            fig, ax = plt.subplots(figsize=(12, 10)) # Increased size for better readability
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, ax=ax)
            ax.set_title('Correlation Heatmap of Numerical Features')
            st.pyplot(fig)
        else:
            st.info("Not enough numerical features for a correlation heatmap.")

        # Example: Boxplot for 'Category' if it exists
        # Adjust 'Category' to your actual categorical column name if different
        categorical_col_for_boxplot = 'Category' # Example, change if needed
        if categorical_col_for_boxplot in df_cleaned.columns:
            st.subheader(f"Distribution by {categorical_col_for_boxplot} (Boxplots)")
            numerical_cols_for_boxplot = df_cleaned.select_dtypes(include=np.number).columns.tolist()
            if target_column_name in numerical_cols_for_boxplot: # Often useful to plot target by category
                pass # Keep target for this plot
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
                    plt.xticks(rotation=45, ha='right') # Rotate labels if long
                    st.pyplot(fig)
            else:
                st.info(f"No numerical features available for boxplots by {categorical_col_for_boxplot}.")
        else:
            st.info(f"'{categorical_col_for_boxplot}' column not found for boxplots. Adjust column name if needed.")

        st.subheader("Pairplot (Sampled Data)")
        st.write("Generating pairplot for a sample of the data due to potential size.")
        pairplot_cols_options = df_cleaned.select_dtypes(include=[np.number]).columns.tolist()
        if target_column_name not in pairplot_cols_options and target_column_name in df_cleaned.columns:
            pairplot_cols_options.append(target_column_name) # Ensure target is an option

        default_pairplot_cols = [col for col in ['PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM', target_column_name] if col in df_cleaned.columns]

        selected_pairplot_cols = st.multiselect("Select columns for Pairplot:", pairplot_cols_options, default=default_pairplot_cols[:5]) # Default to first 5 relevant

        hue_col_pairplot = None
        if categorical_col_for_boxplot in df_cleaned.columns: # Use the same categorical column as for boxplot or let user choose
            if st.checkbox(f"Use '{categorical_col_for_boxplot}' as hue for Pairplot?"):
                hue_col_pairplot = categorical_col_for_boxplot

        if selected_pairplot_cols and len(selected_pairplot_cols) > 1:
             sample_size = min(500, df_cleaned.shape[0]) # Reduced sample for faster pairplots
             df_sample = df_cleaned.sample(sample_size, random_state=42)
             
             pairplot_display_cols = selected_pairplot_cols
             if hue_col_pairplot and hue_col_pairplot not in pairplot_display_cols:
                 pairplot_display_cols_with_hue = selected_pairplot_cols + [hue_col_pairplot]
             else:
                 pairplot_display_cols_with_hue = selected_pairplot_cols

             st.write(f"Generating pairplot for: {', '.join(selected_pairplot_cols)}" + (f" with hue '{hue_col_pairplot}'" if hue_col_pairplot else ""))
             fig = sns.pairplot(df_sample[pairplot_display_cols_with_hue], hue=hue_col_pairplot, diag_kind='kde', corner=True)
             st.pyplot(fig)
        else:
             st.info("Please select at least two numerical columns for the pairplot.")
    else:
        st.warning("Data could not be loaded. EDA cannot be performed.")


def modelling_prediction_page(df, model):
    """Handles model loading, prediction, and evaluation."""
    st.title("🧠 Modelling and Prediction")

    if df is None:
        st.warning("Data could not be loaded. Cannot perform evaluation or prediction.")
        return
    if model is None:
         st.warning("Model could not be loaded. Cannot perform evaluation or prediction.")
         return

    # --- Model Evaluation (existing part) ---
    st.subheader("Model Evaluation on Loaded Data")
    target_column = 'PM2.5' # Ensure this matches your target column name in the CSV

    if target_column not in df.columns:
        st.error(f"Target column '{target_column}' not found in the loaded data. Cannot evaluate or predict.")
        return

    data_df_cleaned = df.dropna(subset=[target_column])
    if data_df_cleaned.empty:
         st.warning(f"No data available for evaluation/prediction after dropping rows with missing '{target_column}'.")
         return

    # Define features (X) and target (y) for evaluation
    # These are all columns EXCEPT the target and any explicit ID columns like 'NO.'
    features_for_eval = [col for col in data_df_cleaned.columns if col != target_column]
    if 'NO.' in features_for_eval: # Assuming 'NO.' is an index or ID column
        features_for_eval.remove('NO.')
    
    if not features_for_eval:
        st.error("No feature columns found after excluding target and ID. Cannot evaluate.")
        return

    X_eval = data_df_cleaned[features_for_eval]
    y_eval = data_df_cleaned[target_column]

    try:
        y_pred_eval = model.predict(X_eval)
        mse = mean_squared_error(y_eval, y_pred_eval)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_eval, y_pred_eval)

        st.write("Evaluation Metrics on a sample of the loaded data (after cleaning):")
        st.metric(label="Mean Squared Error (MSE)", value=f"{mse:.4f}")
        st.metric(label="Root Mean Squared Error (RMSE)", value=f"{rmse:.4f}")
        st.metric(label="R-squared (R²)", value=f"{r2:.4f}")


        st.subheader("Sample Predictions vs Actual")
        results_df = pd.DataFrame({'Actual PM2.5': y_eval, 'Predicted PM2.5': y_pred_eval})
        st.dataframe(results_df.sample(min(10, len(results_df))).reset_index(drop=True)) # Display a sample

        # Feature Importance Plot
        if isinstance(model, Pipeline) and hasattr(model.steps[-1][1], 'feature_importances_'):
             st.subheader("Feature Importance")
             try:
                 importances = model.named_steps['model'].feature_importances_ # Assumes last step is 'model'
                 
                 # Get feature names after preprocessing
                 # This is the tricky part and depends heavily on your pipeline structure
                 processed_feature_names = None
                 if hasattr(model.named_steps.get('preprocessor'), 'get_feature_names_out'):
                     try:
                         processed_feature_names = model.named_steps['preprocessor'].get_feature_names_out()
                     except Exception as e_feat_names:
                         st.warning(f"Could not get feature names from preprocessor: {e_feat_names}. Using generic names.")
                 
                 if processed_feature_names is None or len(processed_feature_names) != len(importances):
                     # Fallback if names can't be retrieved or length mismatch
                     processed_feature_names = [f'feature_{i}' for i in range(len(importances))]
                     if len(processed_feature_names) != len(importances): # Should not happen with fallback
                         st.error("Mismatch in number of importances and feature names. Cannot plot.")
                         pass # Skip plotting
                 
                 if processed_feature_names and len(processed_feature_names) == len(importances):
                     indices = np.argsort(importances)[::-1]
                     top_n = min(20, len(processed_feature_names))
                     top_indices = indices[:top_n]

                     fig, ax = plt.subplots(figsize=(12, max(6, top_n * 0.4))) # Adjust height
                     ax.set_title(f'Top {top_n} Feature Importances')
                     bars = ax.barh(range(top_n), importances[top_indices], align="center")
                     ax.set_yticks(range(top_n))
                     ax.set_yticklabels([processed_feature_names[i] for i in top_indices])
                     ax.invert_yaxis()
                     ax.set_xlabel('Relative Importance')
                     # Add values on bars
                     for bar in bars:
                         ax.text(bar.get_width(), bar.get_y() + bar.get_height()/2,
                                 f'{bar.get_width():.3f}',
                                 va='center', ha='left')
                     st.pyplot(fig)

             except Exception as e_imp:
                 st.error(f"Error plotting feature importance: {e_imp}")
        else:
             st.info("Feature importance plot is available if the model is a scikit-learn Pipeline with a final estimator that has 'feature_importances_'.")

    except Exception as e_eval:
        st.error(f"Error during model evaluation: {e_eval}")
        st.warning("Ensure the loaded data's columns match what the model was trained on (excluding the target).")


    # --- Prediction Section (New) ---
    st.divider() # Visual separator
    st.subheader("🎯 Make a PM2.5 Prediction")
    st.write("Enter the values for the features below to get a PM2.5 prediction.")
    st.caption(f"The model expects the following features: {', '.join(X_eval.columns.tolist())}")


    # X_eval.columns gives the feature names the model's pipeline expects as input
    prediction_input_features = X_eval.columns.tolist()

    if not prediction_input_features:
        st.warning("No features available for prediction input. This might be due to an issue with data loading or feature definition in the evaluation step.")
    else:
        input_data = {}
        # Using 2 columns for a cleaner layout of input fields
        num_input_cols = 2 # You can adjust this, e.g., to 3
        input_streamlit_cols = st.columns(num_input_cols)
        feature_idx = 0

        for feature_name in prediction_input_features:
            current_streamlit_col = input_streamlit_cols[feature_idx % num_input_cols]
            # Use data_df_cleaned (which X_eval is based on) for defaults and unique values
            if feature_name in data_df_cleaned.columns:
                feature_series = data_df_cleaned[feature_name]

                if pd.api.types.is_numeric_dtype(feature_series.dtype):
                    default_val = float(feature_series.mean()) if pd.notna(feature_series.mean()) else 0.0
                    min_val = float(feature_series.min()) if pd.notna(feature_series.min()) else default_val - 10 # Heuristic
                    max_val = float(feature_series.max()) if pd.notna(feature_series.max()) else default_val + 10 # Heuristic
                    
                    # Ensure default_val is within min_val and max_val for slider/number_input
                    default_val = max(min_val, min(default_val, max_val))


                    if feature_name.lower() == 'hour': # Special handling for 'hour'
                         input_data[feature_name] = current_streamlit_col.slider(
                            label=f"Select {feature_name}",
                            min_value=0, max_value=23, # Standard hour range
                            value=int(default_val) if 0 <= int(default_val) <= 23 else 12,
                            step=1
                        )
                    else:
                        input_data[feature_name] = current_streamlit_col.number_input(
                            label=f"Enter {feature_name}",
                            min_value=min_val,
                            max_value=max_val,
                            value=default_val,
                            format="%.2f" # Consistent formatting
                        )
                else: # Assumed categorical
                    # Get unique values, ensuring they are strings and sorted for consistency
                    unique_vals = sorted(list(set(str(val) for val in feature_series.unique() if pd.notna(val))))
                    default_cat_index = 0
                    if unique_vals: # If there are any unique values
                        # Try to set a sensible default, e.g., mode or first item
                        try:
                            mode_val = str(feature_series.mode()[0]) if not feature_series.mode().empty else unique_vals[0]
                            if mode_val in unique_vals:
                                default_cat_index = unique_vals.index(mode_val)
                        except: # Fallback if mode fails or not in list
                            default_cat_index = 0
                    
                    input_data[feature_name] = current_streamlit_col.selectbox(
                        label=f"Select {feature_name}",
                        options=unique_vals,
                        index=default_cat_index,
                        key=f"select_{feature_name}" # Unique key for selectbox
                    )
            else:
                # This case should ideally not be reached if prediction_input_features comes from X_eval.columns
                current_streamlit_col.warning(f"Data for feature '{feature_name}' not found in cleaned dataset. Using placeholder.")
                input_data[feature_name] = current_streamlit_col.text_input(f"Enter {feature_name} (data missing)", value="NA")


            feature_idx += 1

        if st.button("Predict PM2.5", key="predict_button"):
            # Ensure all inputs are valid before creating DataFrame
            valid_inputs = True
            for feature_name in prediction_input_features:
                if input_data.get(feature_name) is None : # Or add more specific checks
                    st.error(f"Input for '{feature_name}' is missing or invalid.")
                    valid_inputs = False
                    break
            
            if valid_inputs:
                try:
                    # Create DataFrame from inputs.
                    # The order of columns is critical and must match X_eval.columns.
                    input_df = pd.DataFrame([input_data])[prediction_input_features]

                    # The scikit-learn pipeline (model) is expected to handle preprocessing
                    # (e.g., OneHotEncoding for categoricals, Scaling for numericals)
                    # based on how it was trained.
                    
                    # Optional: Log the input DataFrame for debugging
                    # st.write("Input DataFrame to model:")
                    # st.dataframe(input_df)
                    # st.write("Input DataFrame dtypes:")
                    # st.text(input_df.dtypes)


                    prediction = model.predict(input_df)
                    st.success(f"Predicted PM2.5 Concentration: **{prediction[0]:.2f} µg/m³**")

                except Exception as e_pred:
                    st.error(f"An error occurred during prediction: {e_pred}")
                    st.warning("Please ensure all input values are valid and that the model's preprocessor is robust to the input types (e.g., strings for categories if trained that way).")
                    if 'input_df' in locals():
                        st.caption("Data sent for prediction (first row):")
                        st.dataframe(input_df.head(1))
                        st.caption("Data types of input sent:")
                        st.text(input_df.dtypes)


# --- Main Application Logic ---
def main():
    """Main function to set up the Streamlit app and navigation."""
    st.set_page_config(
        page_title="Air Quality Prediction App",
        page_icon="🌬️", # Changed icon
        layout="wide"
    )

    st.sidebar.title("Navigation")
    page_options = ["Data Overview", "Exploratory Data Analysis (EDA)", "Modelling and Prediction"]
    page = st.sidebar.radio("Go to", page_options)

    # Load data and model once
    # Ensure DATA_FILENAME points to your actual data file.
    data_df = load_data(DATA_FILENAME)
    # Ensure GOOGLE_DRIVE_MODEL_FILE_ID is set if you want to download from Drive,
    # or that MODEL_LOCAL_FILENAME exists locally.
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
    st.sidebar.info("Air Quality App v1.1")


if __name__ == "__main__":
    main()
