import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# =========================
# 1. CARREGAMENTO E EXPLORAÇÃO
# =========================

df = pd.read_csv("MBA.csv")

print("Formato original:", df.shape)
print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())

# Mantém apenas linhas com decisão registrada
df_labeled = df[df["admission"].notna()].copy()
df_labeled["race"] = df_labeled["race"].fillna("Unknown")

print("\nDistribuição da variável alvo (após filtragem):")
print(df_labeled["admission"].value_counts())

# Gráfico simples da distribuição das classes
class_counts = df_labeled["admission"].value_counts()
plt.bar(class_counts.index, class_counts.values)
plt.title("Distribuição das classes de admissão")
plt.xlabel("Classe")
plt.ylabel("Quantidade de candidatos")

os.makedirs("docs/assets", exist_ok=True)
plt.savefig("docs/assets/svm_distribuicao_classes.png", bbox_inches="tight")
plt.close()

# =========================
# 2. PRÉ-PROCESSAMENTO
# =========================

X = df_labeled.drop(columns=["admission", "application_id"])
y = df_labeled["admission"]

numeric_cols = X.select_dtypes(include=["int64", "float64", "bool"]).columns.tolist()
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

print("\nColunas numéricas:", numeric_cols)
print("Colunas categóricas:", cat_cols)

numeric_transformer = Pipeline(steps=[
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, cat_cols)
    ]
)

# =========================
# 3. DIVISÃO TREINO/TESTE
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTamanho treino:", X_train.shape)
print("Tamanho teste:", X_test.shape)

# =========================
# 4. TREINAMENTO DO SVM
# =========================

kernels = ["linear", "rbf", "poly"]
models = {}
results = {}

for kernel in kernels:
    print(f"\nTreinando modelo com kernel = {kernel}...")
    clf = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", SVC(kernel=kernel))
    ])

    clf.fit(X_train, y_train)
    models[kernel] = clf

    # =========================
    # 5. AVALIAÇÃO
    # =========================
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)

    results[kernel] = {
        "accuracy": acc,
        "confusion_matrix": cm,
        "report": report
    }

    print(f"\n===== RESULTADOS PARA KERNEL: {kernel} =====")
    print("Acurácia:", acc)
    print("Matriz de confusão:\n", cm)
    print("Relatório de classificação:\n", report)

# Identifica o melhor kernel em termos de acurácia
best_kernel = max(results, key=lambda k: results[k]["accuracy"])
print(f"\nMelhor kernel em acurácia: {best_kernel}")
print("Acurácia:", results[best_kernel]["accuracy"])
