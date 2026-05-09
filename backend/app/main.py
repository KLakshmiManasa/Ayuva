from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from jose import jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta
from fastapi.middleware.cors import CORSMiddleware

import pickle
import numpy as np
import shap

# -----------------------------
# APP INIT (MUST BE FIRST)
# -----------------------------
app = FastAPI()

# -----------------------------
# CORS (for frontend)
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# AUTH CONFIG
# -----------------------------
SECRET_KEY = "ayuva_secret"
ALGORITHM = "HS256"

pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")
security = HTTPBearer(auto_error=True)

users_db = {}  # temporary in-memory DB

# -----------------------------
# AUTH FUNCTIONS
# -----------------------------
def hash_password(password):
    password = password.encode('utf-8')[:72]  # truncate safely
    return pwd_context.hash(password)

def verify_password(password, hashed):
    password = password.encode('utf-8')[:72]
    return pwd_context.verify(password, hashed)

def create_token(username):
    payload = {
        "sub": username,
        "exp": datetime.utcnow() + timedelta(hours=10)
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        token = credentials.credentials  # ✅ this already extracts ONLY the token

        print("TOKEN ONLY:", token)  # debug

        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload["sub"]

    except Exception as e:
        print("JWT ERROR:", e)
        raise HTTPException(status_code=401, detail=str(e))

# -----------------------------
# AUTH MODELS
# -----------------------------
class AuthInput(BaseModel):
    username: str
    password: str

# -----------------------------
# AUTH ROUTES
# -----------------------------
@app.post("/signup")
def signup(data: AuthInput):
    try:
        if data.username in users_db:
            return {"error": "User already exists"}

        users_db[data.username] = hash_password(data.password)
        return {"message": "User created"}

    except Exception as e:
        return {"error": str(e)}

@app.post("/login")
def login(data: AuthInput):
    if data.username not in users_db:
        return {"error": "User not found"}

    if not verify_password(data.password, users_db[data.username]):
        return {"error": "Wrong password"}

    token = create_token(data.username)
    return {"token": token}

# -----------------------------
# LOAD MODELS
# -----------------------------
symptom_model = pickle.load(open("../models/disease_model.pkl", "rb"))
ALL_SYMPTOMS = pickle.load(open("../models/symptoms.pkl", "rb"))

diabetes_model = pickle.load(open("../models/diabetes_model.pkl", "rb"))
heart_model = pickle.load(open("../models/heart_model.pkl", "rb"))
liver_model = pickle.load(open("../models/liver_model.pkl", "rb"))

# -----------------------------
# SHAP EXPLAINERS
# -----------------------------
diab_explainer = shap.TreeExplainer(diabetes_model)
heart_explainer = shap.TreeExplainer(heart_model)
liver_explainer = shap.TreeExplainer(liver_model)

# -----------------------------
# INPUT SCHEMA
# -----------------------------
class FullInput(BaseModel):
    symptoms: list

    # Diabetes
    Pregnancies: int
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int

    # Heart
    male: int
    age: int
    education: int
    currentSmoker: int
    cigsPerDay: float
    BPMeds: int
    prevalentStroke: int
    prevalentHyp: int
    diabetes: int
    totChol: float
    sysBP: float
    diaBP: float
    heartRate: float
    glucose: float

    # Liver
    Gender: int
    Total_Bilirubin: float
    Direct_Bilirubin: float
    Alkaline_Phosphatase: float
    Alamine_Aminotransferase: float
    Aspartate_Aminotransferase: float
    Total_Proteins: float
    Albumin: float
    Albumin_and_Globulin_Ratio: float

# -----------------------------
# HELPERS
# -----------------------------
def encode_symptoms(user_symptoms):
    vector = [0] * len(ALL_SYMPTOMS)
    for s in user_symptoms:
        s = s.lower()
        if s in ALL_SYMPTOMS:
            vector[ALL_SYMPTOMS.index(s)] = 1
    return vector

def risk_level(prob):
    prob = float(prob)
    if prob < 0.4:
        return "Low"
    elif prob < 0.7:
        return "Moderate"
    else:
        return "High"

def safe_shap(explainer, input_data):
    shap_values = explainer.shap_values(input_data)

    if isinstance(shap_values, list):
        shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

    return np.array(shap_values).flatten()

def get_top_shap(shap_values, feature_names):
    shap_values = np.array(shap_values).flatten()

    pairs = []
    for i in range(len(feature_names)):
        pairs.append((feature_names[i], float(shap_values[i])))

    pairs.sort(key=lambda x: abs(x[1]), reverse=True)

    return [
        {"feature": f, "impact": round(v, 3)}
        for f, v in pairs[:3]
    ]

# -----------------------------
# MAIN API (PROTECTED)
# -----------------------------
@app.post("/predict_all")
def predict_all(data: FullInput, user: str = Depends(get_current_user)):
    try:
        # SYMPTOMS
        sym_vec = np.array(encode_symptoms(data.symptoms)).reshape(1, -1)
        sym_probs = symptom_model.predict_proba(sym_vec)[0]
        sym_classes = symptom_model.classes_

        symptom_results = sorted(
            zip(sym_classes, sym_probs),
            key=lambda x: float(x[1]),
            reverse=True
        )[:3]

        # DIABETES
        diab_features = [
            "Pregnancies", "Glucose", "BloodPressure",
            "SkinThickness", "Insulin", "BMI",
            "DiabetesPedigreeFunction", "Age"
        ]

        diab_input = np.array([[getattr(data, f) for f in diab_features]])
        diab_prob = float(diabetes_model.predict_proba(diab_input)[0][1])
        diab_shap = safe_shap(diab_explainer, diab_input)

        # HEART
        heart_features = [
            "male", "age", "education", "currentSmoker",
            "cigsPerDay", "BPMeds", "prevalentStroke",
            "prevalentHyp", "diabetes", "totChol",
            "sysBP", "diaBP", "BMI",
            "heartRate", "glucose"
        ]

        heart_input = np.array([[getattr(data, f) for f in heart_features]])
        heart_prob = float(heart_model.predict_proba(heart_input)[0][1])
        heart_shap = safe_shap(heart_explainer, heart_input)

        # LIVER
        liver_features = [
            "Age", "Gender", "Total_Bilirubin",
            "Direct_Bilirubin", "Alkaline_Phosphatase",
            "Alamine_Aminotransferase",
            "Aspartate_Aminotransferase",
            "Total_Proteins", "Albumin",
            "Albumin_and_Globulin_Ratio"
        ]

        liver_input = np.array([[getattr(data, f) for f in liver_features]])
        liver_prob = float(liver_model.predict_proba(liver_input)[0][1])
        liver_shap = safe_shap(liver_explainer, liver_input)

        return {
            "symptom_analysis": [
                {"disease": r[0], "confidence": round(float(r[1]) * 100, 2)}
                for r in symptom_results
            ],

            "diabetes_risk": {
                "level": risk_level(diab_prob),
                "confidence": round(diab_prob * 100, 2),
                "top_factors": get_top_shap(diab_shap, diab_features)
            },

            "heart_risk": {
                "level": risk_level(heart_prob),
                "confidence": round(heart_prob * 100, 2),
                "top_factors": get_top_shap(heart_shap, heart_features)
            },

            "liver_risk": {
                "level": risk_level(liver_prob),
                "confidence": round(liver_prob * 100, 2),
                "top_factors": get_top_shap(liver_shap, liver_features)
            }
        }

    except Exception as e:
     print("ERROR:", e)
     return {"error": str(e)}