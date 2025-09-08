import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from io import StringIO
from sklearn.preprocessing import LabelEncoder

DATA_PATH = "./src/MBA.csv"

label_encoder = LabelEncoder()

df = pd.read_csv(DATA_PATH)

df = df.drop(columns=["application_id"], errors="ignore")

for col in ["gpa", "gmat", "work_exp"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["admission"] = df["admission"].fillna("Deny")
df["race"] = df["race"].fillna("Unknown")

for col in ["gpa", "gmat", "work_exp"]:
    df[col] = df[col].fillna(df[col].median())

for col in ["gender", "international", "major", "race", "work_industry", "admission"]:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip()

adm_map = {"Deny": 0, "Waitlist": 1, "Admit": 2}
df["admission"] = df["admission"].map(adm_map)

cat_cols = ["gender", "international", "major", "race", "work_industry"]
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le  

print(df.sample(frac=0.01, random_state=42).to_markdown(index=False))
