import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("matrices/resultados_solvers_classicos.pkl", "rb") as f:
    R = pickle.load(f)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

# (a) iterações da GD vs kappa, por família
ax = axes[0]
for fam in ["grid", "tree", "erdos_renyi"]:
    pts = sorted([(r["kappa"], r["iters_gd"]) for r in R if r["name"] == fam])
    ks, its = zip(*pts)
    ax.plot(ks, its, marker="o", label=fam)
ax.set_xlabel(r"$\kappa$")
ax.set_ylabel("iterações até convergir (GD)")
ax.set_title("Descida de Gradiente: iterações vs condicionamento")
ax.set_xscale("log"); ax.set_yscale("log")
ax.legend()

# (b) curvas de convergência: melhor caso (erdos_renyi n=4) vs piorr (tree n=25)
ax = axes[1]
melhor = min(R, key=lambda r: r["kappa"])
pior = max(R, key=lambda r: r["kappa"])
ax.plot(melhor["hist_gd"], label=f"{melhor['name']} n={melhor['n']} (κ={melhor['kappa']:.1f})")
ax.plot(pior["hist_gd"], label=f"{pior['name']} n={pior['n']} (κ={pior['kappa']:.1f})")
ax.set_yscale("log")
ax.set_xlabel("iteração")
ax.set_ylabel(r"$\|r_k\| = \|b - Ax_k\|$")
ax.set_title("Convergência: melhor vs pior condicionamento")
ax.legend()

# (c) tempo de Cholesky vs n (deve crescer ~ n^3)
ax = axes[2]
for fam in ["grid", "tree", "erdos_renyi"]:
    pts = sorted([(r["n"], r["t_chol"] * 1e3) for r in R if r["name"] == fam])
    ns, ts = zip(*pts)
    ax.plot(ns, ts, marker="o", label=fam)
ax.set_xlabel("n")
ax.set_ylabel("tempo Cholesky (ms)")
ax.set_title("Custo do solver direto vs tamanho")
ax.legend()

plt.tight_layout()
plt.savefig("comparacao_solvers.png", dpi=150)
print("salvo em comparacao_solvers.png")