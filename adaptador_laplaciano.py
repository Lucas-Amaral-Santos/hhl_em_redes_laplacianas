"""
Adapta as matrizes do notebook hhl_generico para o HHL.
O notebook hhl_laplaciano exige:
    - N = 2^n (nº de qubits)
    - matriz Hermitiana SPD (Laplaciano de grafo conexo)
"""
import numpy as np
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import QFT, UnitaryGate
from qiskit.quantum_info import Statevector
from scipy.linalg import expm


def qpe_exact(circuit: QuantumCircuit, U_pows, reg_clock, reg_b, precision: int, inverse=False):
    b_qubits = [reg_b[j] for j in range(reg_b.size)]

    if not inverse:
        circuit.h(reg_clock)
        for i in range(precision):
            circuit.append(U_pows[i].control(1), [reg_clock[i]] + b_qubits)
    else:
        for i in range(precision):
            circuit.append(U_pows[precision-i-1].control(1), [reg_clock[i]] + b_qubits)
        circuit.h(reg_clock)


def build_U_pows_from_exact(H: np.ndarray, t: float, n: int, precision: int):
    H = np.asarray(H, dtype=complex)
    H = 0.5*(H + H.conj().T)
    U = expm(-1j * H * t)

    U_pows = []
    for i in range(precision):
        Ui = np.linalg.matrix_power(U, 2**i)
        U_pows.append(UnitaryGate(Ui, label=f"U^{2**i}"))
    return U_pows


def carregar(path):
    d = dict(np.load(path))
    return {
        "L": d["L"], "b": d["b"], "n": int(d["n"]),
        "kappa": float(d["kappa"]), "x_chol": d["x_chol"],
        "name": str(d["name"]),
    }


def eh_potencia_de_2(N):
    return N >= 2 and (N & (N - 1)) == 0


def remover_nucleo(L, gamma=None):

    N = L.shape[0]
    P1 = np.ones((N, N)) / N
    if gamma is None:
        gamma = float(np.linalg.eigvalsh(L).max())
    return L + gamma * P1, gamma


def preparar(inst, precision):
    """
    Converte uma instância (dict de carregar()) em entradas prontas p/ o HHL
    """
    L, b = inst["L"], inst["b"]
    N = L.shape[0]
    if not eh_potencia_de_2(N):
        raise ValueError(
            f"[{inst['name']} n={N}] N não é potência de 2; "
            f"o notebook exige N=2^n. Pule esta instância (grades 9 e 25)."
        )
    n = int(round(np.log2(N)))

    H, gamma = remover_nucleo(L)

    ev = np.linalg.eigvalsh(L)
    ev_uteis = ev[ev > 1e-9]          
    lam_min = float(ev_uteis.min())
    C = lam_min

    Pp = 2 ** precision
    t = -2.0 * np.pi / (lam_min * Pp)  # convenção e^{-iHt}

    codes = np.round(ev_uteis / lam_min).astype(int)
    codes = np.clip(codes, 1, Pp - 1)
    pares = sorted(set(zip(codes.tolist(), ev_uteis.tolist())))
    lmb = sorted(set(int(c) for c in codes))

    b_norm = b / np.linalg.norm(b)

    return {
        "H": H, "b": b_norm, "n": n, "t": t,
        "lmb": lmb, "C": C, "pares": pares,
        "gamma": gamma, "x_chol": inst["x_chol"], "kappa": inst["kappa"],
        "name": inst["name"],
    }


def sonda_codigos_qpe(H, t, precision, autovetor):


    n = int(round(np.log2(H.shape[0])))
    clk = QuantumRegister(precision, "c"); b = QuantumRegister(n, "b")
    qc = QuantumCircuit(clk, b)
    qc.prepare_state(list(np.asarray(autovetor, float)), [b[j] for j in range(n)])
    Up = build_U_pows_from_exact(H, t, precision)
    qpe_exact(qc, Up, clk, b, precision, inverse=False)
    qc.append(QFT(precision, inverse=True), clk)
    sv = Statevector.from_instruction(qc).data
    pc = np.zeros(2 ** precision)
    for idx, p in enumerate(np.abs(sv) ** 2):
        pc[idx & ((1 << precision) - 1)] += p
    return int(np.argmax(pc)), float(pc.max())


if __name__ == "__main__":
    import glob
    for path in sorted(glob.glob("matrices/instancia_*.npz")):
        inst = carregar(path)
        try:
            out = preparar(inst, precision=6)
            print(f"OK   {out['name']:12s} N={2**out['n']:2d} "
                  f"kappa={out['kappa']:7.2f} |lmb|={len(out['lmb'])} "
                  f"t={out['t']:.4f} C={out['C']:.4f}")
        except ValueError as ex:
            print(f"PULA {ex}")