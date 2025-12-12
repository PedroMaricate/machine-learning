import matplotlib.pyplot as plt
import pandas as pd

# Leitura dos dados
df = pd.read_csv("./src/MBA.csv")

industries = (
    df["work_industry"]
      .astype(str)
      .fillna("Desconhecido")
      .str.strip()
)

counts = industries.value_counts()

# Criação do gráfico
fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(
    counts.index,
    counts.values,
    edgecolor="black"
)

ax.set_title("Frequência por setor de experiência profissional")
ax.set_xlabel("Setor de atuação")
ax.set_ylabel("Número de candidatos")

plt.xticks(rotation=45, ha="right")
plt.tight_layout()

# SALVA COMO IMAGEM (PNG)
plt.savefig(
    "./docs/assets/img/frequencia_setor_experiencia_profissional.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()
