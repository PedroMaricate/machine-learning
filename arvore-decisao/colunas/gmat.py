import matplotlib.pyplot as plt
import pandas as pd

# Leitura dos dados
df = pd.read_csv("./src/MBA.csv")

gmat_scores = df["gmat"].dropna()

# Criação do histograma
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(
    gmat_scores,
    bins=20,
    edgecolor="black"
)

ax.set_title("Distribuição das pontuações de GMAT dos candidatos")
ax.set_xlabel("Pontuação GMAT")
ax.set_ylabel("Número de candidatos")

plt.tight_layout()

# SALVA COMO IMAGEM (PNG)
plt.savefig(
    "./docs/assets/img/distribuicao_gmat.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()
