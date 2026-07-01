import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pickle

with open("matrices/instancias_index.pkl", "rb") as f:
    paths = pickle.load(f)

data = [dict(np.load(p)) for p in paths]

fig, axes = plt.subplots(2, 3, figsize=(13, 8))

familias = ["grid", "tree", "erdos_renyi"]
for ax, fam in zip(axes[0], familias):
    inst = [d for d in data if str(d["name"]) == fam and int(d["n"]) == 16][0]
    L = inst["L"]
    im = ax.imshow(np.where(np.abs(L) > 1e-9, L, np.nan), cmap="coolwarm")
    ax.set_title(f"L — {fam} (n=16)")
    ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046)


ax = axes[1][0]
for fam in familias:
    pts = sorted([(int(d["n"]), float(d["kappa"])) for d in data if str(d["name"]) == fam])
    ns, ks = zip(*pts)
    ax.plot(ns, ks, marker="o", label=fam)
ax.set_xlabel("n (nº de nós)")
ax.set_ylabel(r"$\kappa = \lambda_{max}/\lambda_2$")
ax.set_title("Número de condição vs tamanho")
ax.legend()
ax.set_yscale("log")

ax = axes[1][1]
inst_grid = [d for d in data if str(d["name"]) == "grid" and int(d["n"]) == 16][0]
ax.plot(np.sort(inst_grid["eigvals"]), marker=".")
ax.set_title("Espectro de L — grid (n=16)")
ax.set_xlabel("índice")
ax.set_ylabel(r"$\lambda_i$")

ax = axes[1][2]
ax.plot(inst_grid["b"], label="b (corrente injetada)", marker="o")
ax.plot(inst_grid["x_chol"], label="x (potencial, solução)", marker="x")
ax.set_title("b vs x — grid (n=16), solução clássica")
ax.legend()

plt.tight_layout()
plt.savefig("instancias_visualizacao.png", dpi=150)
print("salvo em instancias_visualizacao.png")