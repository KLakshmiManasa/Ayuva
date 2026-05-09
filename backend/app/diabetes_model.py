import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

# Load dataset
df = pd.read_csv("../datasets/diabetes.csv")

# -----------------------------
# STEP 1: Features & Labels
# -----------------------------
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# -----------------------------
# STEP 2: Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# STEP 3: Train Model
# -----------------------------
model = RandomForestClassifier()
model.fit(X_train, y_train)

# -----------------------------
# STEP 4: Accuracy
# -----------------------------
accuracy = model.score(X_test, y_test)
print("Model Accuracy:", accuracy)

# -----------------------------
# STEP 5: Save Model
# -----------------------------
pickle.dump(model, open("../models/diabetes_model.pkl", "wb"))

print("✅ Diabetes model trained & saved!")