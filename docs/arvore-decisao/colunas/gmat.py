import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO

df = pd.read_csv("C:/Users/pedro.maricate/Downloads/baseMBA/MBA.csv")

gmat_scores = df["gmat"].dropna()

fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(gmat_scores, bins=20, color="skyblue", edgecolor="black")

ax.set_title("Distribuição das pontuações de GMAT dos candidatos")
ax.set_xlabel("Pontuação GMAT")
ax.set_ylabel("Número de candidatos")

plt.tight_layout()

buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True)
print(buffer.getvalue())
