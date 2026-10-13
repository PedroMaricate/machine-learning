import networkx as nx
import numpy as np
from io import StringIO
import matplotlib

# backend seguro (funciona no notebook e no template)
try:
    matplotlib.use("TkAgg")
except Exception:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt

# Ajuste o nome do arquivo aqui:
DATA_PATH = "./src/citations.txt"

# ===== 1) Carregar o grafo =====
G = nx.read_edgelist(
    DATA_PATH,
    nodetype=int,
    create_using=nx.DiGraph(),
)

print(f"Grafo carregado com {G.number_of_nodes()} nós e {G.number_of_edges()} arestas.<br>")

# ===== 2) Implementação simples do PageRank =====
def pagerank_custom(G, d=0.85, tol=1e-6, max_iter=100):
    nodes = list(G.nodes())
    n = len(nodes)
    index = {node: i for i, node in enumerate(nodes)}

    # vetor inicial uniforme
    pr = np.ones(n) / n

    out_deg = np.array([G.out_degree(node) for node in nodes], dtype=float)
    predecessors = {
        i: [index[u] for u in G.predecessors(node)]
        for i, node in enumerate(nodes)
    }

    for _ in range(max_iter):
        pr_old = pr.copy()
        pr = np.ones(n) * (1 - d) / n

        # peso de nós sem saída (dangling nodes)
        dangling_weight = pr_old[out_deg == 0].sum()

        for i in range(n):
            s = 0.0
            for j in predecessors[i]:
                if out_deg[j] > 0:
                    s += pr_old[j] / out_deg[j]
            pr[i] += d * (s + dangling_weight / n)

        if np.abs(pr - pr_old).sum() < tol:
            break

    return {node: pr[index[node]] for node in nodes}

def top_k(pr_dict, k=10):
    return sorted(pr_dict.items(), key=lambda x: x[1], reverse=True)[:k]

# ===== 3) PageRank "na unha" e comparação com networkx =====
d_default = 0.85

pr_custom = pagerank_custom(G, d=d_default)
pr_nx = nx.pagerank(G, alpha=d_default)

nodes = list(pr_custom.keys())
diff = np.mean([abs(pr_custom[v] - pr_nx[v]) for v in nodes])
print(f"Diff médio entre implementação própria e networkx (d={d_default}): {diff:.6f}<br>")

# ===== 4) Top 10 nós para d=0.85 =====
top10 = top_k(pr_custom, k=min(10, len(nodes)))
print(f"<br><b>Top 10 nós (d={d_default}) - implementação própria:</b><br>")
for node, score in top10:
    print(f"Nó {node}: PageRank = {score:.6e}<br>")

# ===== 5) Gráfico dos scores do top 10 =====
labels = [str(node) for node, _ in top10]
scores = [score for _, score in top10]

fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(range(len(scores)), scores)
ax.set_xticks(range(len(scores)))
ax.set_xticklabels(labels, rotation=45, ha="right")
ax.set_ylabel("PageRank")
ax.set_title(f"Top {len(scores)} nós por PageRank (d={d_default})")
plt.tight_layout()

buffer = StringIO()
plt.savefig(buffer, format="svg")
plt.close(fig)

print("<br><b>Gráfico - Top nós por PageRank (d=0.85):</b><br>")
print(buffer.getvalue())

# ===== 6) Comparação dos rankings com diferentes valores de d =====
d_values = [0.5, 0.85, 0.99]

print("<br><br><b>Análise da variação do fator d:</b><br>")
for d in d_values:
    pr_d = pagerank_custom(G, d=d)
    top10_d = top_k(pr_d, k=10)
    print(f"<br><u>Top 10 nós para d = {d}:</u><br>")
    for node, score in top10_d:
        print(f"Nó {node}: PageRank = {score:.6e}<br>")
