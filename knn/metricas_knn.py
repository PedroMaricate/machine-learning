import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

ART = "docs/knn/artifacts"
data = joblib.load(f"{ART}/knn_eval.pkl")  # carrega só arrays
y_test = data["y_test"]
y_pred = data["y_pred"]

print("=== Avaliação KNN ===")
print(f"Acurácia: {accuracy_score(y_test, y_pred):.4f}")
print("Relatório de Classificação:")
print(classification_report(y_test, y_pred, zero_division=0))
print("Matriz de Confusão:")
print(confusion_matrix(y_test, y_pred))
