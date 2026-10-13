# Divisão dos Dados
import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Carregar a base pré-processada
df = pd.read_csv("./src/MBA.csv")

# 2. Converter variável-alvo para binária, caso ainda não esteja
df["admission"] = df["admission"].apply(lambda x: 1 if x == "Admit" else 0)

# 3. Separar features (X) e alvo (y)
X = df.drop(columns=["admission"])
y = df["admission"]

# 4. Divisão treino/teste (80/20) com estratificação
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# 5. Exibir informações
print("Tamanho do conjunto de treinamento:", X_train.shape)
print("Tamanho do conjunto de teste:", X_test.shape)
print("\nProporção de classes no treino:")
print(y_train.value_counts(normalize=True))
print("\nProporção de classes no teste:")
print(y_test.value_counts(normalize=True))
