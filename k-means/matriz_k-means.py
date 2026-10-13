# docs/metricas-avaliacao/matriz-confusao-kmeans.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, silhouette_score, adjusted_rand_score
import seaborn as sns
import matplotlib.pyplot as plt
from io import StringIO

# === Carregar e preparar dados (mesmo pré-processamento) ===
df = pd.read_csv("./src/MBA.csv")
df["admission"] = df["admission"].fillna("Deny")
df["race"] = df["race"].fillna("Unknown")

adm_map = {"Deny": 0, "Waitlist": 1, "Admit": 2}
y = df["admission"].map(adm_map).astype(int)

num_cols = ["gpa", "gmat", "work_exp"]
cat_cols = ["gender", "international", "major", "race", "work_industry"]

for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(df[c].median())

X_cat = pd.get_dummies(
    df[cat_cols].astype(str).apply(lambda s: s.str.strip()),
    drop_first=False, dtype=int
)
scaler = StandardScaler()
X_num = df[num_cols].copy()
X_num[num_cols] = scaler.fit_transform(X_num[num_cols])

X = pd.concat([X_num, X_cat], axis=1).values

# === K-Means ===
kmeans = KMeans(n_clusters=3, init="k-means++", n_init=10, max_iter=300, random_state=42)
labels = kmeans.fit_predict(X)

sil = silhouette_score(X, labels)
ari = adjusted_rand_score(y, labels)
print(f"Silhouette Score: {sil:.4f}")
print(f"Adjusted Rand Index: {ari:.4f}")

# === 1) Matriz "bruta": classes reais vs clusters ===
cm_raw = confusion_matrix(y, labels)
plt.figure(figsize=(6, 4))
sns.heatmap(cm_raw, annot=True, fmt="d", cmap="Purples")
plt.title("Matriz de Confusão - K-Means (classes vs clusters)")
plt.xlabel("Cluster")
plt.ylabel("Classe Real")
plt.tight_layout()
buf = StringIO()
plt.savefig(buf, format="svg", transparent=True)
print(buf.getvalue())
plt.close()

# === 2) Matriz "mapeada": cluster -> classe majoritária ===
cluster_map = {}
classes = np.unique(y)
for c in np.unique(labels):
    idx = (labels == c)
    if idx.sum() == 0:
        cluster_map[c] = classes[0]
        continue
    counts = np.bincount(y[idx], minlength=classes.max()+1)
    cluster_map[c] = counts.argmax()
print("Mapeamento cluster → classe:", cluster_map)

y_pred_from_clusters = np.array([cluster_map[c] for c in labels])
cm_map = confusion_matrix(y, y_pred_from_clusters)
plt.figure(figsize=(6, 4))
sns.heatmap(cm_map, annot=True, fmt="d", cmap="Greens")
plt.title("Matriz de Confusão - K-Means (clusters mapeados → classes)")
plt.xlabel("Classe Prevista (via cluster)")
plt.ylabel("Classe Real")
plt.tight_layout()
buf = StringIO()
plt.savefig(buf, format="svg", transparent=True)
print(buf.getvalue())
plt.close()
