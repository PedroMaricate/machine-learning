import matplotlib.pyplot as plt
import pandas as pd

# Leitura dos dados
df = pd.read_csv("./src/MBA.csv")

work_exp = df["work_exp"].dropna()

# Criação do histograma
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(
    work_exp,
    bins=15,
    edgecolor="black"
)

ax.set_title("Distribuição da experiência profissional (anos)")
ax.set_xlabel("Anos de experiência")
ax.set_ylabel("Número de candidatos")

plt.tight_layout()

# SALVA COMO IMAGEM (PNG)
plt.savefig(
    "./docs/assets/img/distribuicao_experiencia_profissional.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()
