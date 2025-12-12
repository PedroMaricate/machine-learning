import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import pandas as pd

# Leitura dos dados
df = pd.read_csv("./src/MBA.csv")

gpa = (
    df["gpa"]
      .astype(float)
      .dropna()
)

# Criação do boxplot
fig, ax = plt.subplots(figsize=(10, 6))
ax.boxplot(
    gpa,
    vert=True,
    patch_artist=True,
    boxprops=dict(facecolor="skyblue", color="blue"),
    medianprops=dict(color="red", linewidth=2),
    whiskerprops=dict(color="black"),
    capprops=dict(color="black"),
    flierprops=dict(
        markerfacecolor="orange",
        marker="o",
        markersize=6,
        alpha=0.6
    )
)

ax.set_title("Distribuição do GPA dos candidatos")
ax.set_ylabel("GPA")

# Legenda explicativa
box_patch = mpatches.Patch(
    facecolor="skyblue",
    edgecolor="blue",
    label="Intervalo interquartílico (IQR)"
)
median_line = mlines.Line2D(
    [], [], color="red", linewidth=2, label="Mediana"
)
outlier_marker = mlines.Line2D(
    [], [], color="orange", marker="o",
    linestyle="None", label="Outliers"
)

ax.legend(
    handles=[box_patch, median_line, outlier_marker],
    loc="upper right"
)

plt.tight_layout()

# SALVA COMO IMAGEM (PNG)
plt.savefig(
    "./docs/assets/img/distribuicao_gpa_boxplot.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()
