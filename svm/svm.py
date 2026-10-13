import matplotlib

try:
    matplotlib.use("TkAgg")
except:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt


import pandas as pd
import matplotlib.pyplot as plt

from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

DATA_PATH = "./src/MBA.csv"

# === Carregamento da base ===
df = pd.read_csv(DATA_PATH)

# Remove identificador sem valor preditivo
df = df.drop(columns=["application_id"], errors="ignore")

# Garante tipos numéricos
for col in ["gpa", "gmat", "work_exp"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Tratamento de valores ausentes
df["admission"] = df["admission"].fillna("Deny")
df["race"] = df["race"].fillna("Unknown")

for col in ["gpa", "gmat", "work_exp"]:
    df[col] = df[col].fillna(df[col].median())

# Limpeza de espaços / tipos nas categóricas
for col in ["gender", "international", "major", "race", "work_industry", "admission"]:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip()

# Mapeia a variável alvo para valores ordenados
adm_map = {"Deny": 0, "Waitlist": 1, "Admit": 2}
df["admission"] = df["admission"].map(adm_map)

# Codificação das variáveis categóricas
cat_cols = ["gender", "international", "major", "race", "work_industry"]
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

# === Separação entre features (X) e target (y) ===
X = df.drop(columns=["admission"])
y = df["admission"]

# === Divisão treino / teste com estratificação ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=27,
    stratify=y
)

# === Normalização (essencial para SVM) ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === Modelo SVM com kernel RBF ===
svm_clf = SVC(kernel="rbf", C=1.0, gamma="scale")
svm_clf.fit(X_train_scaled, y_train)

# === Avaliação ===
y_pred = svm_clf.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=["Deny", "Waitlist", "Admit"])
cm = confusion_matrix(y_test, y_pred)

# Imprime métricas em HTML simples
print(f"Acurácia do modelo SVM (RBF): {accuracy:.2%}<br>")
print("Relatório de classificação:<br>")
print(report.replace("\n", "<br>"))

# === Gráfico da matriz de confusão em SVG ===
fig, ax = plt.subplots(figsize=(6, 4))
im = ax.imshow(cm, interpolation="nearest")
ax.set_title("Matriz de Confusão - SVM (RBF)")
ax.set_xlabel("Classe Predita")
ax.set_ylabel("Classe Verdadeira")

classes = ["Deny", "Waitlist", "Admit"]
ax.set_xticks(range(len(classes)))
ax.set_yticks(range(len(classes)))
ax.set_xticklabels(classes)
ax.set_yticklabels(classes)

# Escreve os valores dentro dos quadradinhos
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, cm[i, j], ha="center", va="center")

plt.tight_layout()

# Exporta como SVG para o template
buffer = StringIO()
plt.savefig(buffer, format="svg")
plt.close(fig)

print(buffer.getvalue())
