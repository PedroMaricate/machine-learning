import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

plt.figure(figsize=(12, 10))

df = pd.read_csv("./src/MBA.csv")

df["admission"] = df["admission"].fillna("Deny")
df["race"] = df["race"].fillna("Unknown")

adm_map = {"Deny": 0, "Waitlist": 1, "Admit": 2}
y = df["admission"].map(adm_map).astype(int)

num_cols = ["gpa", "gmat", "work_exp"]
for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(df[c].median())

cat_cols = ["gender", "international", "major", "race", "work_industry"]
X_cat = pd.get_dummies(df[cat_cols].astype(str).apply(lambda s: s.str.strip()),
                       drop_first=False, dtype=int)

scaler = StandardScaler()
X_num = df[num_cols].copy()
X_num[num_cols] = scaler.fit_transform(X_num[num_cols])

X = pd.concat([X_num, X_cat], axis=1).values

kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=100,
                random_state=42, n_init=10)
labels = kmeans.fit_predict(X)

from sklearn.decomposition import PCA

pca = PCA(n_components=2, random_state=42)
X_2d = pca.fit_transform(X)

plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap='viridis', s=50)
plt.scatter(kmeans.cluster_centers_[:, 0:1], kmeans.cluster_centers_[:, 1:2],
            c='red', marker='*', s=200, label='Centroids (PCA proj.)')
plt.title('K-Means Clustering (MBA Dataset)')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.legend()

buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True)
print(buffer.getvalue())
