import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import numpy as np

# Import necessary components for the pipeline (needed for loading the model)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

import io # Required for capturing df.info() output


# --- Page Functions ---

def data_overview_page():
    """Displays data overview information."""
    st.title("📊 Data Overview")

    st.write("Upload your merged dataset (`merged_data.csv`) to view its basic information.")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("File uploaded successfully!")

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

        except Exception as e:
            st.error(f"Error loading or processing file: {e}")
    else:
        st.info("Please upload a CSV file to see the data overview.")


def eda_page():
    """Performs and displays Exploratory Data Analysis."""
    st.title("📈 Exploratory Data Analysis (EDA)")

    st.write("Upload your merged dataset (`merged_data.csv`) to perform EDA.")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv", key="eda_uploader")

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("File uploaded successfully!")

            # Drop rows with missing PM2.5 for analysis consistency with modeling
            # This ensures EDA reflects the data used for training
            df_cleaned = df.dropna(subset=['PM2.5'])
            st.write(f"Using {df_cleaned.shape[0]} rows for EDA after dropping missing PM2.5 values.")


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


        except Exception as e:
            st.error(f"Error loading or processing file for EDA: {e}")
    else:
        st.info("Please upload a CSV file to perform EDA.")


def modelling_prediction_page():
    """Handles model loading, prediction, and evaluation."""
    st.title("🧠 Modelling and Prediction")

    st.write("Upload your merged dataset (`merged_data.csv`) and the trained model file (`random_forest_regressor_model.pkl`) to make predictions and evaluate the model.")

    # --- File Uploaders ---
    uploaded_data_file = st.file_uploader("Choose the merged data CSV file", type="csv", key="model_data_uploader")
    uploaded_model_file = st.file_uploader("Choose the trained model PKL file", type="pkl", key="model_uploader")

    data_df = None
    model = None

    # --- Load Data ---
    if uploaded_data_file is not None:
        try:
            data_df = pd.read_csv(uploaded_data_file)
            st.success("Data file loaded successfully!")
        except Exception as e:
            st.error(f"Error loading data file: {e}")

    # --- Load Model ---
    if uploaded_model_file is not None:
        try:
            # Need to explicitly define the objects used in the pipeline for joblib to load correctly
            # This is a common issue when loading pipelines
            # Ensure the necessary classes (Pipeline, ColumnTransformer, etc.) are imported
            model = joblib.load(uploaded_model_file)
            st.success("Model file loaded successfully!")
            st.write("Loaded model type:", type(model)) # Helps verify the loaded object
            # Optional: Display some info about the loaded model if it's a pipeline
            if isinstance(model, Pipeline):
                 st.write("Model is a scikit-learn Pipeline.")
                 st.write("Pipeline steps:", [step[0] for step in model.steps])
                 # Check if the last step is a RandomForestRegressor
                 if isinstance(model.steps[-1][1], RandomForestRegressor):
                      st.write("The model is a RandomForestRegressor.")
                 else:
                      st.warning("The loaded model is a Pipeline, but the final step is not a RandomForestRegressor.")
            elif isinstance(model, RandomForestRegressor):
                 st.write("Model is a scikit-learn RandomForestRegressor.")
            else:
                 st.warning("The loaded file does not appear to be a scikit-learn Pipeline or RandomForestRegressor.")

        except Exception as e:
            st.error(f"Error loading model file: {e}")

    # --- Prediction and Evaluation ---
    if data_df is not None and model is not None:
        st.subheader("Model Evaluation and Prediction")

        target_column = 'PM2.5'

        if target_column not in data_df.columns:
            st.error(f"Target column '{target_column}' not found in the uploaded data.")
            return

        # Drop rows where the target variable (PM2.5) is missing for evaluation
        data_df_cleaned = data_df.dropna(subset=[target_column])

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
        # This is a crucial check!
        # If the model is a pipeline, the preprocessor step should have a way to get feature names
        # This is tricky if the pipeline was trained on different columns.
        # A robust approach is to ensure the uploaded data has the same columns used during training.
        # For simplicity here, we assume the uploaded data has the necessary columns.
        # In a real app, you might save the list of feature columns used during training.

        try:
            # Make predictions using the loaded model
            y_pred_eval = model.predict(X_eval)

            # Evaluate the model
            mse = mean_squared_error(y_eval, y_pred_eval)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_eval, y_pred_eval)

            st.write("Evaluation Metrics on Uploaded Data:")
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

    elif data_df is None and uploaded_data_file is not None:
        st.warning("Please upload the model file (.pkl) to proceed with evaluation and prediction.")
    elif model is None and uploaded_model_file is not None:
         st.warning("Please upload the data file (.csv) to proceed with evaluation and prediction.")
    else:
        st.info("Please upload both the data file (.csv) and the trained model file (.pkl).")


# --- Main Application Logic ---

def main():
    """Main function to set up the Streamlit app and navigation."""
    st.set_page_config(
        page_title="Air Quality Prediction App",
        page_icon="💨",
        layout="wide"
    )

    # --- Navigation ---
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Data Overview", "Exploratory Data Analysis (EDA)", "Modelling and Prediction"])

    # --- Page Routing ---
    if page == "Data Overview":
        data_overview_page()
    elif page == "Exploratory Data Analysis (EDA)":
        eda_page()
    elif page == "Modelling and Prediction":
        modelling_prediction_page()

    # --- Footer (Optional) ---
    st.sidebar.markdown("---")
    st.sidebar.write("Developed by Mirukshan") 


if __name__ == "__main__":
    main()

