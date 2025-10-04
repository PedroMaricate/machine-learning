import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO

df = pd.read_csv("./src/MBA.csv")

work_exp = df["work_exp"].dropna()

fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(work_exp, bins=15, color="orange", edgecolor="black")

ax.set_title("Distribuição da experiência profissional (anos)")
ax.set_xlabel("Anos de experiência")
ax.set_ylabel("Número de candidatos")

plt.tight_layout()

buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True)
print(buffer.getvalue())
