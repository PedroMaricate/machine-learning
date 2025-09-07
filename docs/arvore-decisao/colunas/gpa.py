import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import pandas as pd
from io import StringIO

df = pd.read_csv("C:/Users/pedro.maricate/Downloads/baseMBA/MBA.csv")

gpa = (
    df["gpa"]
      .astype(float)
      .dropna()
)

fig, ax = plt.subplots(figsize=(10, 6))
box = ax.boxplot(
    gpa, vert=True, patch_artist=True,
    boxprops=dict(facecolor="skyblue", color="blue"),
    medianprops=dict(color="red", linewidth=2),
    whiskerprops=dict(color="black"),
    capprops=dict(color="black"),
    flierprops=dict(markerfacecolor="orange", marker="o", markersize=6, alpha=0.6)
)

ax.set_title("Distribuição do GPA dos candidatos")
ax.set_ylabel("GPA")

box_patch = mpatches.Patch(color="skyblue", label="Intervalo interquartílico (IQR)")
median_line = mlines.Line2D([], [], color="red", label="Mediana")
outlier_marker = mlines.Line2D([], [], color="orange", marker="o", linestyle="None", label="Outliers")

ax.legend(handles=[box_patch, median_line, outlier_marker], loc="upper right")

plt.tight_layout()

buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True)
print(buffer.getvalue())
