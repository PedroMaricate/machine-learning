import numpy as np
import joblib
from sklearn.metrics import silhouette_score, adjusted_rand_score, confusion_matrix

ART = "docs/k-means/artifacts"
data = joblib.load(f"{ART}/kmeans_artifacts.pkl")

labels = data["labels"]
X = data["X"]
y = data["y"]

# métricas
sil = silhouette_score(X, labels)
ari = adjusted_rand_score(y, labels)

# mapeamento cluster->classe (voto majoritário)
cm_raw = confusion_matrix(y, labels)  # linhas=classes, colunas=clusters
cluster_to_class = cm_raw.argmax(axis=0)  # para cada cluster, qual classe mais aparece

y_pred = np.array([cluster_to_class[c] for c in labels])
cm_mapped = confusion_matrix(y, y_pred)

print("=== Avaliação K-Means ===")
print(f"Silhouette Score: {sil:.4f}")
print(f"Adjusted Rand Index: {ari:.4f}")
print("Matriz de Confusão (classes vs clusters):")
print(cm_raw)
print("Matriz de Confusão (clusters mapeados → classes):")
print(cm_mapped)
