import matplotlib.pyplot as plt
import pandas as pd

# Leitura dos dados
df = pd.read_csv("./src/MBA.csv")

races = (
    df["race"]
      .astype(str)
      .fillna("Desconhecido")
      .str.strip()
)

counts = races.value_counts()

# Criação do gráfico
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(
    counts.index,
    counts.values,
    edgecolor="black"
)

ax.set_title("Frequência de raça por aplicações")
ax.set_xlabel("Raça")
ax.set_ylabel("Número de aplicações")

plt.xticks(rotation=45, ha="right")
plt.tight_layout()

# SALVA COMO IMAGEM (PNG)
plt.savefig(
    "./docs/assets/img/frequencia_raca.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()
