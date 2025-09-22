import pandas as pd
import numpy as np
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

# Initialize the FastAPI app
app = FastAPI(title="Medical Recommendation System API",
              description="An API that predicts diseases based on symptoms and provides recommendations.")

# --- Load necessary files ---
# Load the trained model and label encoder
try:
    model = joblib.load('random_forest_model.pkl')
    le = joblib.load('label_encoder.pkl')
except FileNotFoundError:
    raise RuntimeError("Model files not found. Please run 'train_and_save_model.py' first.")

# Load the recommendation datasets
description = pd.read_csv('description.csv')
diets = pd.read_csv('diets.csv')
medications = pd.read_csv('medications.csv')
precautions = pd.read_csv('precautions_df.csv')
workout = pd.read_csv('workout_df.csv')

# Get the list of all symptoms from the training data columns
# This is needed to create the input vector for the model
training_data = pd.read_csv('Training.csv')
training_data.columns = training_data.columns.str.strip()
if 'Unnamed: 133' in training_data.columns:
    training_data = training_data.drop('Unnamed: 133', axis=1)
symptom_list = training_data.drop('prognosis', axis=1).columns.values


# --- Pydantic Model for Input ---
# This defines the expected structure of the request body
class SymptomsInput(BaseModel):
    symptoms: List[str]
    class Config:
        schema_extra = {
            "example": {
                "symptoms": ["itching", "skin_rash", "nodal_skin_eruptions"]
            }
        }


# --- Helper function for recommendations ---
def get_recommendations(disease: str) -> dict:
    """Fetches all recommendations for a given disease."""
    # Description
    desc = description.loc[description['Disease'] == disease, 'Description'].values
    desc = desc[0] if len(desc) > 0 else "No description available."

    # Precautions
    prec_rows = precautions.loc[precautions['Disease'] == disease, ['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']].values.flatten()
    prec = [p for p in prec_rows if pd.notna(p)]

    # Medications
    meds = medications.loc[medications['Disease'] == disease, 'Medication'].values
    meds = eval(meds[0]) if len(meds) > 0 else []

    # Diets
    diet_list = diets.loc[diets['Disease'] == disease, 'Diet'].values
    diet_list = eval(diet_list[0]) if len(diet_list) > 0 else []

    # Workouts
    work = workout.loc[workout['disease'] == disease, 'workout'].values.tolist()

    return {
        "description": desc,
        "precautions": prec,
        "medications": meds,
        "diets": diet_list,
        "workouts": work
    }

# --- API Endpoint ---

@app.post("/home")
def home() :
    return "Medicine recommendation System"

@app.post("/predict")
def predict_disease(symptoms_input: SymptomsInput):
    """
    Predicts the disease based on a list of symptoms and returns full recommendations.
    """
    symptoms = symptoms_input.symptoms

    # Create a binary vector from the input symptoms
    symptom_vector = np.zeros(len(symptom_list))
    for symptom in symptoms:
        # Clean symptom name to match training columns
        clean_symptom = symptom.strip().replace(" ", "_")
        if clean_symptom in symptom_list:
            index = np.where(symptom_list == clean_symptom)[0][0]
            symptom_vector[index] = 1

    # Predict the disease label
    predicted_label = model.predict([symptom_vector])[0]

    # Decode the label to get the disease name
    predicted_disease = le.inverse_transform([predicted_label])[0]

    # Get the detailed recommendations
    recommendations = get_recommendations(predicted_disease)

    # Structure the final response
    response = {
        "predicted_disease": predicted_disease,
        "recommendations": recommendations
    }

    return response

# To run the app, save this file as main.py and run: uvicorn main:app --reload
