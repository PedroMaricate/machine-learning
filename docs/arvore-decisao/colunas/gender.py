import matplotlib

try:
    matplotlib.use("TkAgg")
except:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

# Carrega os dados
df = pd.read_csv("./src/MBA.csv")

genders = (
    df["gender"]
      .astype(str)
      .fillna("Desconhecido")
      .str.strip()
)

counts = genders.value_counts()

# Criação do gráfico
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(counts.index, counts.values)

ax.set_title("Frequência de gênero por aplicações")
ax.set_xlabel("Gênero")
ax.set_ylabel("Número de aplicações")

plt.xticks(rotation=45, ha="right")
plt.tight_layout()

# SALVA COMO IMAGEM (PNG)
plt.savefig(
    "./docs/assets/img/frequencia_genero_aplicacoes.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()
