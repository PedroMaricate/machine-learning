import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    silhouette_score, adjusted_rand_score
)
from sklearn.decomposition import PCA

df = pd.read_csv("./src/MBA.csv")

df["admission"] = df["admission"].fillna("Deny")
df["race"] = df["race"].fillna("Unknown")

adm_map = {"Deny": 0, "Waitlist": 1, "Admit": 2}
y = df["admission"].map(adm_map).astype(int)

num_cols = ["gpa", "gmat", "work_exp"]
for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(df[c].median())

cat_cols = ["gender", "international", "major", "race", "work_industry"]
X_cat = pd.get_dummies(
    df[cat_cols].astype(str).apply(lambda s: s.str.strip()),
    drop_first=False, dtype=int
)

scaler = StandardScaler()
X_num = df[num_cols].copy()
X_num[num_cols] = scaler.fit_transform(X_num[num_cols])

X = pd.concat([X_num, X_cat], axis=1).values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

y_pred_knn = knn.predict(X_test)

print("=== Avaliação KNN ===")
print("Accuracy:", accuracy_score(y_test, y_pred_knn))
print("\nClassification Report:\n", classification_report(y_test, y_pred_knn))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_knn))

kmeans = KMeans(n_clusters=3, init="k-means++", n_init=10, random_state=42)
labels = kmeans.fit_predict(X)

print("\n=== Avaliação K-Means ===")
print("Silhouette Score:", silhouette_score(X, labels))
print("Adjusted Rand Index (comparando com admission):", adjusted_rand_score(y, labels))
print("\nConfusion Matrix (clusters vs classes):\n", confusion_matrix(y, labels))

pca = PCA(n_components=2, random_state=42)
X_2d = pca.fit_transform(X)

plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_2d[:,0], y=X_2d[:,1], hue=labels, palette="viridis", s=50)
plt.title("Clusters K-Means (projeção PCA 2D)")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")

buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True)
print(buffer.getvalue()) 
