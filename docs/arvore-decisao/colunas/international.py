import matplotlib.pyplot as plt
import pandas as pd

# Leitura dos dados
df = pd.read_csv("./src/MBA.csv")

internationals = (
    df["international"]
      .astype(str)
      .fillna("Desconhecido")
      .str.strip()
)

counts = internationals.value_counts()

# Criação do gráfico
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(counts.index, counts.values)

ax.set_title("Frequência de alunos internacionais por aplicações")
ax.set_xlabel("Alunos internacionais")
ax.set_ylabel("Número de aplicações")

plt.xticks(rotation=45, ha="right")
plt.tight_layout()

# SALVA COMO IMAGEM (PNG)
plt.savefig(
    "./docs/assets/img/frequencia_alunos_internacionais.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()
