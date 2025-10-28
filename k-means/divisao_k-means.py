import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score

df = pd.read_csv("./src/MBA.csv")

df["admission"] = df["admission"].fillna("Deny")
y = df["admission"].map({"Deny":0, "Waitlist":1, "Admit":2}).astype(int)

df["race"] = df["race"].fillna("Unknown")
num_cols = ["gpa", "gmat", "work_exp"]
for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(df[c].median())

cat_cols = ["gender","international","major","race","work_industry"]
X_cat = pd.get_dummies(df[cat_cols].astype(str).apply(lambda s: s.str.strip()),
                       drop_first=False, dtype=int)
X_num = df[num_cols].copy()

scaler = StandardScaler()
X_num[num_cols] = scaler.fit_transform(X_num[num_cols])

X = pd.concat([X_num, X_cat], axis=1).values

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = kmeans.fit_predict(X)

sil = silhouette_score(X, labels)
ari = adjusted_rand_score(y, labels)
print(f"Silhouette: {sil:.3f} | ARI: {ari:.3f}")
