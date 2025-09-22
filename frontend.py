import streamlit as st
import pandas as pd
import requests
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Medical Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Helper function to format symptom names ---
def format_symptom(symptom):
    return symptom.replace('_', ' ').title()

# --- Load Symptom List ---
# It's better to load this from the source to ensure consistency
try:
    training_data = pd.read_csv('Training.csv')
    symptoms_list = [col.strip() for col in training_data.columns if col.strip().lower() != 'prognosis']
    formatted_symptoms_list = [format_symptom(s) for s in symptoms_list]
except FileNotFoundError:
    st.error("Error: 'Training.csv' not found. Please make sure the file is in the correct directory.")
    st.stop() # Stop the app if the essential data is missing


# --- UI Layout ---

# Sidebar for information
with st.sidebar:
    st.title("About")
    st.info(
        "This application uses a Machine Learning model to predict potential diseases based on the symptoms you provide. "
        "The model has been trained on a comprehensive dataset to achieve high accuracy."
    )
    st.warning("**Disclaimer:** This is an AI-powered informational tool and not a substitute for professional medical advice. Please consult a doctor for any health concerns.")

# Main content
st.title("AI Medical Assistant 🩺")
st.markdown("Enter your symptoms below to receive a potential diagnosis and detailed recommendations for diet, lifestyle, and medication.")

# Symptom selection using a multi-select box
st.subheader("Step 1: Select Your Symptoms")
selected_formatted_symptoms = st.multiselect(
    label="Start typing to search for symptoms...",
    options=formatted_symptoms_list,
    help="You can select multiple symptoms from the dropdown list."
)

# Convert formatted symptoms back to the original format for the API
selected_original_symptoms = [symptoms_list[formatted_symptoms_list.index(s)] for s in selected_formatted_symptoms]

st.subheader("Step 2: Get Your Prediction")
if st.button("Predict Disease", type="primary", use_container_width=True):
    if not selected_original_symptoms:
        st.warning("Please select at least one symptom before predicting.")
    else:
        # API request payload
        payload = {"symptoms": selected_original_symptoms}

        # Show a spinner while waiting for the API response
        with st.spinner("Analyzing your symptoms... Please wait."):
            try:
                # Make the request to the FastAPI backend
                api_url = "http://127.0.0.1:8000/predict"
                response = requests.post(api_url, json=payload)
                response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

                results = response.json()

                # --- Display Results ---
                st.success(f"**Predicted Disease:** {results['predicted_disease']}")

                recommendations = results['recommendations']

                # Display Description in an expander
                with st.expander("**Disease Description**", expanded=True):
                    st.write(recommendations.get('description', 'No description available.'))

                # Create columns for the rest of the recommendations
                col1, col2 = st.columns(2)

                with col1:
                    with st.expander("**Recommended Precautions**", expanded=True):
                        precautions = recommendations.get('precautions', [])
                        if precautions:
                            for item in precautions:
                                st.markdown(f"- {item}")
                        else:
                            st.write("No specific precautions listed.")

                    with st.expander("**Suggested Medications**", expanded=True):
                        medications = recommendations.get('medications', [])
                        if medications:
                             for item in medications:
                                st.markdown(f"- {item}")
                        else:
                            st.write("No specific medications listed.")

                with col2:
                    with st.expander("**Dietary Recommendations**", expanded=True):
                        diets = recommendations.get('diets', [])
                        if diets:
                            for item in diets:
                                st.markdown(f"- {item}")
                        else:
                            st.write("No specific diet recommendations listed.")

                    with st.expander("**Workouts & Lifestyle**", expanded=True):
                        workouts = recommendations.get('workouts', [])
                        if workouts:
                            for item in workouts:
                                st.markdown(f"- {item}")
                        else:
                            st.write("No specific workout recommendations listed.")


            except requests.exceptions.ConnectionError:
                st.error("Connection Error: Could not connect to the backend API. Please make sure the FastAPI server is running.")
            except requests.exceptions.RequestException as e:
                st.error(f"An API error occurred: {e}")
