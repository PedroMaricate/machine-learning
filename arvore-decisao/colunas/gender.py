import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO

df = pd.read_csv("./src/MBA.csv")

genders = (
    df["gender"]
      .astype(str)
      .fillna("Desconhecido")
      .str.strip()
)

counts = genders.value_counts()

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(counts.index, counts.values)

ax.set_title("Frequência de gênero por aplicações")
ax.set_xlabel("Gênero")
ax.set_ylabel("Número de aplicações")

plt.xticks(rotation=45, ha="right")
plt.tight_layout()

buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True)
print(buffer.getvalue())