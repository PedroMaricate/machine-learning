import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("C:/Users/pedro.maricate/Downloads/baseMBA/MBA.csv")

label_encoder = LabelEncoder()

df["admission"] = df["admission"].fillna("Deny") 
df["race"] = df["race"].fillna("Unknown")

df["gender"] = label_encoder.fit_transform(df["gender"])
df["international"] = label_encoder.fit_transform(df["international"])
df["major"] = label_encoder.fit_transform(df["major"])
df["race"] = label_encoder.fit_transform(df["race"])
df["work_industry"] = label_encoder.fit_transform(df["work_industry"])
df["admission"] = label_encoder.fit_transform(df["admission"])

x = df[[
    "gender", "international", "gpa", "major",
    "race", "gmat", "work_exp", "work_industry"
]]

y = df["admission"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=27, stratify=y
)

print("Dimensão do treino:", x_train.shape)
print("Dimensão do teste:", x_test.shape)
