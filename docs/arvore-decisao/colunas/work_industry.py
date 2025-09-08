import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO

df = pd.read_csv("./src/MBA.csv")

industries = (
    df["work_industry"]
      .astype(str)
      .fillna("Desconhecido")
      .str.strip()
)

counts = industries.value_counts()

fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(counts.index, counts.values, color="lightcoral", edgecolor="black")

ax.set_title("Frequência por setor de experiência profissional")
ax.set_xlabel("Setor de atuação")
ax.set_ylabel("Número de candidatos")

plt.xticks(rotation=45, ha="right")

plt.tight_layout()

buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True)
print(buffer.getvalue())
