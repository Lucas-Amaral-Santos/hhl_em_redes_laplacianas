"""
Implementação explícita  dos dois solvers clássicos de
referência citados na Metodologia: Decomposição de Cholesky e
Descida de Gradiente , para resolver L x = b restrito ao
subespaço ortogonal a 1 .

"""
import numpy as np
import pickle
import time




def cholesky_decompose(A: np.ndarray) -> np.ndarray:

    n = A.shape[0]
    G = np.zeros_like(A, dtype=float)
    for j in range(n):
        soma = A[j, j] - np.dot(G[j, :j], G[j, :j])
        if soma <= 0:
            raise np.linalg.LinAlgError(
                f"Matriz não é positiva definida na coluna {j} (soma={soma:.2e})"
            )
        G[j, j] = np.sqrt(soma)
        for i in range(j + 1, n):
            G[i, j] = (A[i, j] - np.dot(G[i, :j], G[j, :j])) / G[j, j]
    return G


def forward_substitution(G: np.ndarray, b: np.ndarray) -> np.ndarray:
    n = len(b)
    y = np.zeros(n)
    for i in range(n):
        y[i] = (b[i] - np.dot(G[i, :i], y[:i])) / G[i, i]
    return y


def backward_substitution(GT: np.ndarray, y: np.ndarray) -> np.ndarray:
    n = len(y)
    x = np.zeros(n)
    for i in reversed(range(n)):
        x[i] = (y[i] - np.dot(GT[i, i + 1:], x[i + 1:])) / GT[i, i]
    return x


def cholesky_solve(A: np.ndarray, b: np.ndarray):
    G = cholesky_decompose(A)
    y = forward_substitution(G, b)
    x = backward_substitution(G.T, y)
    return x, G

# Descida de Gradiente (steepest descent)
def gradient_descent_solve(A: np.ndarray, b: np.ndarray, x0=None,
                            tol: float = 1e-8, max_iter: int = 20_000):
    n = A.shape[0]
    x = np.zeros(n) if x0 is None else x0.copy()
    r = b - A @ x
    hist = [np.linalg.norm(r)]
    for k in range(1, max_iter + 1):
        Ar = A @ r
        denom = r @ Ar
        if denom <= 1e-300:
            break
        alpha = (r @ r) / denom
        x = x + alpha * r
        r = b - A @ x
        hist.append(np.linalg.norm(r))
        if hist[-1] < tol:
            return x, np.array(hist), k
    return x, np.array(hist), max_iter


def regularize(L, eps=1e-10):
    n = L.shape[0]
    P1 = np.ones((n, n)) / n
    return L + eps * P1


def run_all(paths):
    resultados = []
    for p in paths:
        d = dict(np.load(p))
        L, b = d["L"], d["b"]
        n, kappa, name = int(d["n"]), float(d["kappa"]), str(d["name"])
        A = regularize(L)

        t0 = time.perf_counter()
        x_chol, _ = cholesky_solve(A, b)
        t_chol = time.perf_counter() - t0
        x_chol -= x_chol.mean()

        t0 = time.perf_counter()
        x_gd, hist_gd, iters = gradient_descent_solve(A, b)
        t_gd = time.perf_counter() - t0
        x_gd -= x_gd.mean()

        erro_chol = np.linalg.norm(x_chol - d["x_chol"]) / np.linalg.norm(d["x_chol"])
        erro_gd = np.linalg.norm(x_gd - d["x_chol"]) / np.linalg.norm(d["x_chol"])

        resultados.append(dict(
            path=p, name=name, n=n, kappa=kappa,
            t_chol=t_chol, t_gd=t_gd, iters_gd=iters,
            hist_gd=hist_gd, erro_chol=erro_chol, erro_gd=erro_gd,
        ))
        print(f"[{name:11s} n={n:3d}] kappa={kappa:8.2f}  "
              f"Cholesky: {t_chol*1e3:6.2f}ms (erro={erro_chol:.1e})   "
              f"GD: {iters:5d} iters / {t_gd*1e3:7.2f}ms (erro={erro_gd:.1e})")
    return resultados


if __name__ == "__main__":
    with open("matrices/instancias_index.pkl", "rb") as f:
        paths = pickle.load(f)
    resultados = run_all(paths)

    with open("matrices/resultados_solvers_classicos.pkl", "wb") as f:
        pickle.dump(resultados, f)