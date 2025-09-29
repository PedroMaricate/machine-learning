import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score  # <-- métrica adicionada

plt.figure(figsize=(12, 10))

# --- Carregar e preparar a base ---
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

# --- Treinamento do K-Means ---
kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=100,
                random_state=42, n_init=10)
labels = kmeans.fit_predict(X)

# --- Métricas de avaliação (sem rótulos verdadeiros) ---
# 1) Silhouette Score (quanto maior, melhor; faixa [-1, 1])
sil_score = silhouette_score(X, labels)

# 2) Inertia (WCSS) - soma das distâncias quadráticas aos centróides (quanto menor, melhor)
inertia = kmeans.inertia_

# --- PCA apenas para visualização em 2D ---
pca = PCA(n_components=2, random_state=42)
X_2d = pca.fit_transform(X)

# Variância explicada pelo PCA (útil para relatar quanta info o 2D preserva)
explained = pca.explained_variance_ratio_
explained_total = explained.sum()

# Projetar centróides para o espaço 2D do PCA
centers_2d = pca.transform(kmeans.cluster_centers_)

# --- Gráfico ---
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap='viridis', s=50)
plt.scatter(centers_2d[:, 0], centers_2d[:, 1],
            c='red', marker='*', s=200, label='Centroids (PCA proj.)')
plt.title('K-Means Clustering (MBA Dataset)')
plt.xlabel(f'PCA 1 ({explained[0]*100:.1f}% var.)')
plt.ylabel(f'PCA 2 ({explained[1]*100:.1f}% var.)')
plt.legend()

# --- Impressões das métricas ---
print(f"Silhouette Score: {sil_score:.4f}")
print(f"Inertia (WCSS): {inertia:.2f}")
print(f"Variância explicada PCA por componente: {explained}")
print(f"Variância explicada acumulada (PCA 2D): {explained_total:.4f}")

# Exportar o SVG do gráfico (para embutir em página)
buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True)
print(buffer.getvalue())
# --- Fim do código ---