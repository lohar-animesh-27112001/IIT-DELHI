import os
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, computed_field
from fastapi.responses import FileResponse
from typing import Annotated, Literal
import pickle
import pandas as pd
# import sklearn
# from sklearn.pipeline import Pipeline

# import the model.pkl file in rb-> read binary mode
with open("model.pkl", "rb") as file:
    pipeline_model = pickle.load(file)

# MLflow
MODEL_VERSION = "1.0.0"  # Example version, adjust as needed

app = FastAPI()

# CORS middleware to allow requests from the frontend
# This is necessary if your frontend is hosted on a different domain or port
# For example, if your frontend is running on http://localhost:5500, you can specify that domain
# If you want to allow all origins, you can use ["*"] but it's not recommended for production
app.add_middleware(
    # http://127.0.0.1:5500/ML+FastAPI/FrontEnd/index.html
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500"],  # You can specify ["http://localhost:5500"] if needed
    allow_credentials=True, # Allows cookies to be sent with requests
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],  # Allows all headers
)


# Serve static files (HTML, CSS, JS)
app.mount("/static", StaticFiles(directory="../FrontEnd"), name="static")

# Serve index.html at root
@app.get("/")
def read_root():
    return FileResponse("../FrontEnd/index.html")

@app.get("/health")
def health_check():
    """
    Health check endpoint to verify if the API is running.
    Returns a simple JSON response indicating the service is up.
    AWS service health checks can use this endpoint to ensure the API is operational.
    """
    return JSONResponse(status_code=200, content={"status": "ok", "message": "API is running smoothly!", "version": MODEL_VERSION})

# pydantic model to validate data
class UserInput(BaseModel):
    age: Annotated[int, Field(..., ge=0, lt=120, title="Age of the patient", description="Age should be between 0 and 120")]
    weight: Annotated[float, Field(..., gt=0, lt=200, title="Weight of the patient", description="Weight should be between 0 and 200 kg")]
    height: Annotated[float, Field(..., gt=0, lt=300, title="Height of the patient", description="Height should be between 0 and 300 cm")]
    income_lpa: Annotated[float, Field(..., gt=0, title="Income of the patient", description="Income should be positive")]
    smoker: Annotated[bool, Field(..., title="Smoking status", description="Whether the patient is a smoker or not")]
    city: Annotated[str, Field(..., title="City of the patient", description="City where the patient resides")]
    occupation: Annotated[Literal['retired', 'freelancer', 'student', 'government_job',
       'business_owner', 'unemployed', 'private_job'], Field(..., title="Occupation of the patient", description="Occupation of the patient")]
    
    @computed_field
    @property
    def bmi(self) -> float:
        """Calculate Body Mass Index (BMI)"""
        return round(self.weight / (self.height ** 2), 2)
    
    @computed_field
    @property
    def lifestyle_risk(self) -> str:
        if self.smoker and self.bmi > 30:
            return "high"
        elif self.smoker or self.bmi > 27:
            return "medium"
        else:
            return "low"

    @computed_field
    @property
    def age_group(self) -> str:
        if self.age < 25:
            return "young"
        elif self.age < 45:
            return "adult"
        elif self.age < 60:
            return "middle_aged"
        return "senior"
    
    @computed_field
    @property
    def city_tier(self) -> int:
        tier_1_cities = ["Mumbai", "Delhi", "Bangalore", "Chennai", "Kolkata", "Hyderabad", "Pune"]
        tier_2_cities = [
            "Jaipur", "Chandigarh", "Indore", "Lucknow", "Patna", "Ranchi", "Visakhapatnam", "Coimbatore",
            "Bhopal", "Nagpur", "Vadodara", "Surat", "Rajkot", "Jodhpur", "Raipur", "Amritsar", "Varanasi",
            "Agra", "Dehradun", "Mysore", "Jabalpur", "Guwahati", "Thiruvananthapuram", "Ludhiana", "Nashik",
            "Allahabad", "Udaipur", "Aurangabad", "Hubli", "Belgaum", "Salem", "Vijayawada", "Tiruchirappalli",
            "Bhavnagar", "Gwalior", "Dhanbad", "Bareilly", "Aligarh", "Gaya", "Kozhikode", "Warangal",
            "Kolhapur", "Bilaspur", "Jalandhar", "Noida", "Guntur", "Asansol", "Siliguri"
        ]
        if self.city in tier_1_cities:
            return 1
        elif self.city in tier_2_cities:
            return 2
        else:
            return 3

@app.post("/predict")
def predict_insurance_premium(data: UserInput):
    input_df = pd.DataFrame([{
        'bmi': data.bmi,
        'age_group': data.age_group,
        'lifestyle_risk': data.lifestyle_risk,
        'city_tier': data.city_tier,
        'income_lpa': data.income_lpa,
        'occupation': data.occupation
    }])
    prediction = pipeline_model.predict(input_df)[0]
    return JSONResponse(status_code=200, content={"insurance_premium_category": prediction})