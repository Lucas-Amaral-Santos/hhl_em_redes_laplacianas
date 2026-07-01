import numpy as np
from math import asin
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RYGate, QFT, UnitaryGate
from adaptador_laplaciano import carregar, preparar
from qiskit_aer import AerSimulator
from scipy.linalg import expm
import glob



def qpe_exact(circuit: QuantumCircuit, U_pows, reg_clock, reg_b, precision: int, inverse=False):
    b_qubits = [reg_b[j] for j in range(reg_b.size)]

    if not inverse:
        circuit.h(reg_clock)
        for i in range(precision):
            circuit.append(U_pows[i].control(1), [reg_clock[i]] + b_qubits)
    else:
        for i in range(precision):
            circuit.append(U_pows[i].inverse().control(1), [reg_clock[i]] + b_qubits)  
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

def rotacoes_genericas(circuit, reg_clock, reg_anc, precision, s_min=1):

    P = 2 ** precision
    for s in range(max(1, s_min), P):
        ang = 2.0 * asin(1.0 / s)
        circuit.append(
            RYGate(ang).control(precision, ctrl_state=int(s)),
            [reg_clock[j] for j in range(precision)] + [reg_anc[0]],
        )
 


def hhl_laplaciano(H, b_vec, precision, t, pares, C, n, measure=True):
    
    anc = QuantumRegister(1, "ancilla")
    clk = QuantumRegister(precision, "clock")
    b = QuantumRegister(n, "b")
    ca = ClassicalRegister(1, "meas_a")
    cb = ClassicalRegister(n, "meas_b")
    qc = QuantumCircuit(anc, clk, b, ca, cb)

    qc.prepare_state(list(np.asarray(b_vec, float)), [b[j] for j in range(n)])

    U_pows = build_U_pows_from_exact(H, t, n,precision)
    qpe_exact(qc, U_pows, clk, b, precision, inverse=False)
    qc.append(QFT(precision, inverse=True), clk)
    rotacoes_genericas(qc, clk, anc, precision, s_min=1)
    if measure:
        qc.measure(anc, ca)
    qc.append(QFT(precision, inverse=False), clk)
    qpe_exact(qc, U_pows, clk, b, precision, inverse=True)
    if measure:
        qc.measure(b, cb)
    return qc

def run_simulation(qc, shots=1000):
    simulator = AerSimulator(method='matrix_product_state')
    result = simulator.run(qc.decompose(reps=10), shots=shots).result()
    return result.get_counts()


for path in sorted(glob.glob("matrices/instancia_*.npz")):
    inst = carregar(path)
    try:
        precision = int(np.ceil(np.log2(inst["kappa"]))) + 1
        P = preparar(inst, precision=precision)
    except ValueError as ex:
        print("PULA:", ex)             
        continue

    qc = hhl_laplaciano(P["H"], P["b"], precision, P["t"], P["pares"], P["C"], P["n"])
    counts = run_simulation(qc, shots=20000)        # roda a simulação

    probs = np.zeros(2 ** P["n"])
    total = sum(counts.values())
    for bits, c in counts.items():
        b_bits = bits.split()[0]                     
        probs[int(b_bits, 2)] += c / total
    alvo = P["x_chol"] ** 2 / np.sum(P["x_chol"] ** 2)
    fid = np.sum(np.sqrt(probs * alvo)) ** 2         
    print(f"{P['name']:12s} N={2**P['n']:2d} kappa={P['kappa']:7.2f} "
          f"fidelidade(|x|^2 vs x_chol)={fid:.4f}")