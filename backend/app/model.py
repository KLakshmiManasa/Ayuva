import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset
df = pd.read_csv("../datasets/Healthcare.csv")

# -----------------------------
# STEP 1: Clean Symptoms Column
# -----------------------------
df["Symptoms"] = df["Symptoms"].str.lower()

# Convert symptoms into list
df["Symptoms"] = df["Symptoms"].apply(lambda x: x.split(", "))

# -----------------------------
# STEP 2: Get ALL UNIQUE SYMPTOMS
# -----------------------------
all_symptoms = set()

for symptoms in df["Symptoms"]:
    for s in symptoms:
        all_symptoms.add(s)

all_symptoms = sorted(list(all_symptoms))

print("Total symptoms:", len(all_symptoms))

# -----------------------------
# STEP 3: Convert to Binary Matrix
# -----------------------------
def encode(symptom_list):
    vector = [0] * len(all_symptoms)
    for symptom in symptom_list:
        if symptom in all_symptoms:
            index = all_symptoms.index(symptom)
            vector[index] = 1
    return vector

X = df["Symptoms"].apply(encode).tolist()
y = df["Disease"]

# -----------------------------
# STEP 4: Train Model
# -----------------------------
model = RandomForestClassifier()
model.fit(X, y)

# -----------------------------
# STEP 5: Save model + symptoms
# -----------------------------
pickle.dump(model, open("../models/disease_model.pkl", "wb"))
pickle.dump(all_symptoms, open("../models/symptoms.pkl", "wb"))

print("✅ Model trained with real dataset!")