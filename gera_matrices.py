import numpy as np
import networkx as nx
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse.csgraph import laplacian as sp_laplacian
from scipy.sparse import csr_matrix
import pickle
from scipy.sparse.linalg import cg


RNG = np.random.default_rng(42)


# Gera grafo de grade 2D
def grid_graph(side: int) -> nx.Graph:
    """Grade 2D side x side (n = side**2 nós). Estrutura regular, kappa baixo."""
    G = nx.grid_2d_graph(side, side)
    return nx.convert_node_labels_to_integers(G)

# Gera árvore aleatória
def random_tree(n: int) -> nx.Graph:
    """Árvore aleatória uniforme com n nós. Caso esparso extremo (d=1 em folhas)."""
    return nx.random_labeled_tree(n, seed=int(RNG.integers(0, 10**6)))

# Gera grafo de Erdős–Rényi esparso
def erdos_renyi_sparse(n: int, avg_degree: float = 3.0) -> nx.Graph:
    
    p = min(avg_degree / n, 1.0)
    for _ in range(50):
        G = nx.gnp_random_graph(n, p, seed=int(RNG.integers(0, 10**6)))
        if nx.is_connected(G):
            return G
    raise RuntimeError("Não foi possível gerar G(n,p) conexo; aumente avg_degree.")


# Montagem do Laplaciano ponderado (resistores aleatórios)

def weighted_laplacian(G: nx.Graph, r_low=0.5, r_high=2.0):

    n = G.number_of_nodes()
    L = np.zeros((n, n))
    for u, v in G.edges():
        r = RNG.uniform(r_low, r_high)
        w = 1.0 / r
        L[u, u] += w
        L[v, v] += w
        L[u, v] -= w
        L[v, u] -= w
    return L


# Vetor de corrente b, projetado ortogonal a 1 (conservação de carga)

def random_current_vector(n: int) -> np.ndarray:
    b = RNG.normal(size=n)
    b -= b.mean()         
    return b / np.linalg.norm(b)


# Solução clássica de referência

def solve_classical(L: np.ndarray, b: np.ndarray, eps_reg: float = 1e-10):

    n = L.shape[0]
    ones = np.ones(n)
    P1 = np.outer(ones, ones) / n
    L_reg = L + eps_reg * P1

    # Cholesky
    c, low = cho_factor(L_reg)
    x_chol = cho_solve((c, low), b)
    x_chol -= x_chol.mean()  # remove componente no núcleo

    # Gradiente Conjugado
    x_cg, info = cg(L_reg, b, rtol=1e-10, atol=0.0)
    x_cg -= x_cg.mean()

    return x_chol, x_cg


def condition_number(L: np.ndarray):
    # kappa = lambda_max / lambda_2 (menor autovalor não nulo)
    eigvals = np.sort(np.linalg.eigvalsh(L))
    lam2 = eigvals[1]   # eigvals[0] ~ 0
    lam_max = eigvals[-1]
    return lam_max / lam2, eigvals



def build_instance(name: str, G: nx.Graph, out_dir: str = "."):
    n = G.number_of_nodes()
    L = weighted_laplacian(G)
    b = random_current_vector(n)
    x_chol, x_cg = solve_classical(L, b)
    kappa, eigvals = condition_number(L)

    path = f"{out_dir}/matrices/instancia_{name}_n{n}.npz"
    np.savez(
        path,
        L=L, b=b,
        x_chol=x_chol, x_cg=x_cg,
        eigvals=eigvals, kappa=kappa,
        n=n, name=name,
    )
    print(f"[{name:10s}] n={n:4d}  kappa={kappa:8.2f}  -> {path}")
    return path


if __name__ == "__main__":
    sizes = [4, 8, 16]   # grade: side**2; demais: n direto

    instances = []
    for side in [2, 4]:
        instances.append(("grid", grid_graph(side)))
    for n in sizes:
        instances.append(("tree", random_tree(n)))
    for n in sizes:
        instances.append(("erdos_renyi", erdos_renyi_sparse(n, avg_degree=3.0)))

    paths = [build_instance(name, G, out_dir=".") for name, G in instances]

    with open("matrices/instancias_index.pkl", "wb") as f:
        pickle.dump(paths, f)