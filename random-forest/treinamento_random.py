# docs/random-forest/rf_viz.py
# Gera imagens SVG (matriz de confusão, importâncias e 1 árvore do ensemble)
# para embutir no MkDocs com exec="on" html="1".

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree

# ========================
#  Helpers de renderização
# ========================
def fig_to_svg():
    """Retorna a figura atual como string SVG e fecha a figura."""
    buf = StringIO()
    plt.tight_layout()
    plt.savefig(buf, format="svg", bbox_inches="tight")
    plt.close()
    return buf.getvalue()

def h2(txt):  # título HTML
    print(f"<h2 style='margin:16px 0 8px'>{txt}</h2>")

def h3(txt):
    print(f"<h3 style='margin:12px 0 6px'>{txt}</h3>")

# ==========
# 1) Dados
# ==========
# Ajuste o caminho se necessário
df = pd.read_csv("./src/MBA.csv")

# Pré-processamento essencial (alvo binário + codificação leve)
df["admission"] = df["admission"].fillna("Deny")
df["admission"] = (df["admission"] == "Admit").astype(int)

# imputações simples
for col in ["gmat", "work_exp"]:
    if df[col].isna().any():
        df[col] = df[col].fillna(df[col].median())
if "race" in df.columns and df["race"].isna().any():
    df["race"] = df["race"].fillna("Unknown")

# remove ID se existir
if "application_id" in df.columns:
    df = df.drop(columns=["application_id"])

# one-hot nas categóricas
cat_cols = df.select_dtypes(exclude=["number", "bool"]).columns
X = pd.get_dummies(df.drop(columns=["admission"]), columns=cat_cols, drop_first=True)
y = df["admission"]

# divisão
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ==========
# 2) Modelo
# ==========
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    max_features="sqrt",
    oob_score=True,
    n_jobs=-1,
    random_state=42
)
rf.fit(X_train, y_train)

# ==================
# 3) Métricas texto
# ==================
y_pred = rf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
rep = classification_report(y_test, y_pred, output_dict=True)
rep_df = (pd.DataFrame(rep).T
          .rename_axis("classe")
          .reset_index()[["classe","precision","recall","f1-score","support"]])

# Títulos e números principais
h2("Desempenho do Modelo")
print(f"<p><b>Acurácia (teste):</b> {acc:.4f} &nbsp;&nbsp;|&nbsp;&nbsp; "
      f"<b>OOB score:</b> {rf.oob_score_:.4f}</p>")
h3("Relatório de Classificação")
print(rep_df.to_html(index=False, float_format=lambda x: f"{x:.4f}"))

# =========================
# 4) Matriz de Confusão SVG
# =========================
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=[0,1])
disp.plot(values_format="d", cmap="Blues", colorbar=False)
plt.title("Matriz de Confusão — Random Forest")
print(fig_to_svg())

# ===================================
# 5) Importância das variáveis (Top15)
# ===================================
imp = (pd.Series(rf.feature_importances_, index=X.columns)
         .sort_values(ascending=False)
         .head(15))[::-1]  # invertido p/ barh crescer p/ cima

plt.figure(figsize=(8,5))
sns.barplot(x=imp.values, y=imp.index)
plt.xlabel("Score de Importância (MDI)")
plt.ylabel("Variável")
plt.title("Importância das Variáveis — Top 15")
plt.grid(axis="x", linestyle="--", alpha=0.5)
print(fig_to_svg())

# ==========================================
# 6) Uma árvore do ensemble (visão ilustrativa)
# ==========================================
# Pega um estimador e desenha até profundidade 3 para ficar legível
est = rf.estimators_[0]
plt.figure(figsize=(12,6))
tree.plot_tree(est, max_depth=3, fontsize=8, feature_names=X.columns, class_names=["0","1"], filled=True)
plt.title("Árvore Representativa (estimador 0) — max_depth=3")
print(fig_to_svg())
