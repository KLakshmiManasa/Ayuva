import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

# Load dataset
df = pd.read_csv("../datasets/heart.csv")

# -----------------------------
# STEP 1: Handle Missing Values
# -----------------------------
df = df.replace("NA", pd.NA)
df = df.apply(pd.to_numeric, errors='coerce')

# Fill missing values with column mean
df = df.fillna(df.mean())

# -----------------------------
# STEP 2: Features & Labels
# -----------------------------
X = df.drop("TenYearCHD", axis=1)
y = df["TenYearCHD"]

# -----------------------------
# STEP 3: Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# STEP 4: Train Model
# -----------------------------
model = RandomForestClassifier()
model.fit(X_train, y_train)

# -----------------------------
# STEP 5: Accuracy
# -----------------------------
accuracy = model.score(X_test, y_test)
print("Heart Model Accuracy:", accuracy)

# -----------------------------
# STEP 6: Save Model
# -----------------------------
pickle.dump(model, open("../models/heart_model.pkl", "wb"))

print("✅ Heart disease model trained & saved!")