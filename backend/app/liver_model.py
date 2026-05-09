import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

# Load dataset
df = pd.read_csv("../datasets/liver.csv")

# -----------------------------
# STEP 1: Clean Data
# -----------------------------

# Convert Gender to numeric
df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})

# Handle missing values
df = df.fillna(df.mean())

# Convert target column
# 1 = disease, 2 = no disease → convert to 1/0
df["Dataset"] = df["Dataset"].apply(lambda x: 1 if x == 1 else 0)

# -----------------------------
# STEP 2: Features & Labels
# -----------------------------
X = df.drop("Dataset", axis=1)
y = df["Dataset"]

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
print("Liver Model Accuracy:", accuracy)

# -----------------------------
# STEP 6: Save Model
# -----------------------------
pickle.dump(model, open("../models/liver_model.pkl", "wb"))

print("✅ Liver disease model trained & saved!")