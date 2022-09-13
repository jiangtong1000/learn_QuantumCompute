import numpy as np
from qiskit import QuantumCircuit
from qiskit.providers.aer import Aer
from scipy import linalg
from qiskit import execute
from qiskit.quantum_info import Statevector


def make_circ_m(thetas, theta_i, theta_j, qcidx):
    assert qcidx in [0, 1]
    circ = QuantumCircuit(9)
    circ.h(0)
    circ.h(1)
    circ.h(2)
    circ.h(5)
    circ.h(6)
    apply_dic = {0: [circ.crx, 1, 3], 1: [circ.crx, 2, 4],
                 2: [circ.rzz, 3, 4], 3: [circ.rx, 3],
                 4: [circ.rx, 4], 5: [circ.rzz, 3, 4]}
    apply_diff_dic = {0: circ.ccx, 1: circ.ccx,
                      2: circ.cz, 3: circ.cx,
                      4: circ.cx, 5: circ.cz}
    for i in range(6):
        if i == theta_i:
            if qcidx == 1:
                circ.x(0)
            func = apply_diff_dic[i]
            if i in [0, 1]:
                func(0, apply_dic[i][1], apply_dic[i][2])
            elif i in [2, 5]:
                func(0, apply_dic[i][1])
                func(0, apply_dic[i][2])
            else:
                func(0, apply_dic[i][1])
            if qcidx == 1:
                circ.x(0)
        func = apply_dic[i][0]
        if len(apply_dic[i]) == 2:
            func(thetas[i], apply_dic[i][1])
        elif len(apply_dic[i]) == 3:
            func(thetas[i], apply_dic[i][1], apply_dic[i][2])
    apply_dic = {0: [circ.crx, 5, 7], 1: [circ.crx, 6, 8],
                 2: [circ.rzz, 7, 8], 3: [circ.rx, 7],
                 4: [circ.rx, 8], 5: [circ.rzz, 7, 8]}
    apply_diff_dic = {0: circ.ccx, 1: circ.ccx,
                      2: circ.cz, 3: circ.cx,
                      4: circ.cx, 5: circ.cz}
    for i in range(6):
        if i == theta_j:
            circ.x(0)
            func = apply_diff_dic[i]
            if i in [0, 1]:
                func(0, apply_dic[i][1], apply_dic[i][2])
            elif i in [2, 5]:
                func(0, apply_dic[i][1])
                func(0, apply_dic[i][2])
            else:
                func(0, apply_dic[i][1])
            circ.x(0)
        func = apply_dic[i][0]
        if len(apply_dic[i]) == 2:
            func(thetas[i], apply_dic[i][1])
        elif len(apply_dic[i]) == 3:
            func(thetas[i], apply_dic[i][1], apply_dic[i][2])
    circ.cswap(0, 3, 7)
    circ.cswap(0, 4, 8)
    circ.h(0)
    return circ


def psi_partial_thetai(thetas, theta_i, part):
    circ = QuantumCircuit(9)
    circ.h(0)
    assert part in ["real", "imag"]
    if part == "imag":
        circ.sdg(0)
    circ.h(1)
    circ.h(2)
    circ.h(5)
    circ.h(6)
    apply_dic = {0: [circ.crx, 1, 3], 1: [circ.crx, 2, 4],
                 2: [circ.rzz, 3, 4], 3: [circ.rx, 3],
                 4: [circ.rx, 4], 5: [circ.rzz, 3, 4]}
    apply_diff_dic = {0: circ.ccx, 1: circ.ccx,
                      2: circ.cz, 3: circ.cx,
                      4: circ.cx, 5: circ.cz}

    for i in range(6):
        if i == theta_i:
            circ.x(0)
            func = apply_diff_dic[i]
            if i in [0, 1]:
                func(0, apply_dic[i][1], apply_dic[i][2])
            elif i in [2, 5]:
                func(0, apply_dic[i][1])
                func(0, apply_dic[i][2])
            else:
                func(0, apply_dic[i][1])
            circ.x(0)
        func = apply_dic[i][0]
        if len(apply_dic[i]) == 2:
            func(thetas[i], apply_dic[i][1])
        elif len(apply_dic[i]) == 3:
            func(thetas[i], apply_dic[i][1], apply_dic[i][2])
    apply_dic = {0: [circ.crx, 5, 7], 1: [circ.crx, 6, 8],
                 2: [circ.rzz, 7, 8], 3: [circ.rx, 7],
                 4: [circ.rx, 8], 5: [circ.rzz, 7, 8]}
    for i in range(6):
        func = apply_dic[i][0]
        if len(apply_dic[i]) == 2:
            func(thetas[i], apply_dic[i][1])
        elif len(apply_dic[i]) == 3:
            func(thetas[i], apply_dic[i][1], apply_dic[i][2])
    return circ


def make_circ_xxzz(thetas, theta_i, oper_idx, qcidx):
    circ = psi_partial_thetai(thetas, theta_i, "real")
    apply_diff_dic = {0: [circ.cx, 7], 1: [circ.cx, 8],
                      2: [circ.cz, 7, 8]}
    if qcidx == 0:
        circ.x(0)
    func = apply_diff_dic[oper_idx][0]
    func(0, apply_diff_dic[oper_idx][1])
    if len(apply_diff_dic[oper_idx]) == 3:
        func(0, apply_diff_dic[oper_idx][2])
    if qcidx == 0:
        circ.x(0)
    circ.cswap(0, 3, 7)
    circ.cswap(0, 4, 8)
    circ.h(0)
    return circ


def make_circ_xy1(thetas, theta_i, oper_idx, qcidx):
    circ = psi_partial_thetai(thetas, theta_i, "real")
    apply_qubit = {0: 7, 1: 8}
    if qcidx == 0:
        circ.cx(0, apply_qubit[oper_idx])
        circ.cy(0, apply_qubit[oper_idx])
    elif qcidx == 1:
        circ.x(0)
        circ.cy(0, apply_qubit[oper_idx])
        circ.cx(0, apply_qubit[oper_idx])
        circ.x(0)
    circ.cswap(0, 3, 7)
    circ.cswap(0, 4, 8)
    circ.h(0)
    return circ


def make_circ_xx(thetas, theta_i, qcidx):
    circ = psi_partial_thetai(thetas, theta_i, "imag")
    apply_gates = {0: [circ.x, 7],
                   1: [circ.y, 7],
                   2: [circ.x, 8],
                   3: [circ.y, 8]}
    func, qubit = apply_gates[qcidx]
    func(qubit)
    circ.cswap(0, 3, 7)
    circ.cswap(0, 4, 8)
    circ.h(0)
    return circ


def make_circ_xy(thetas, theta_i, qcidx):
    circ = psi_partial_thetai(thetas, theta_i, "real")
    if qcidx == 0:
        circ.x(0)
        circ.cx(0, 7)
        circ.x(0)
        circ.cy(0, 7)
    if qcidx == 1:
        circ.x(0)
        circ.cy(0, 7)
        circ.x(0)
        circ.cx(0, 7)
    if qcidx == 2:
        circ.x(0)
        circ.cx(0, 8)
        circ.x(0)
        circ.cy(0, 8)
    if qcidx == 3:
        circ.x(0)
        circ.cy(0, 8)
        circ.x(0)
        circ.cx(0, 8)
    circ.cswap(0, 3, 7)
    circ.cswap(0, 4, 8)
    circ.h(0)
    return circ


def make_circ_rho(thetas, theta_i):
    circ = psi_partial_thetai(thetas, theta_i, "imag")
    circ.cswap(0, 3, 7)
    circ.cswap(0, 4, 8)
    circ.h(0)
    return circ


def measure_qc_statevec(qc):
    backend = Aer.get_backend('statevector_simulator')
    job_result = execute(qc, backend).result().get_statevector(qc)
    probs = Statevector(job_result).probabilities(qargs=[0])
    return probs[0] - probs[1]


def measure_z0(thetas):
    circ = QuantumCircuit(4)
    circ.h(0)
    circ.h(1)
    apply_dic = {0: [circ.crx, 0, 2], 1: [circ.crx, 1, 3],
                 2: [circ.rzz, 2, 3], 3: [circ.rx, 2],
                 4: [circ.rx, 3], 5: [circ.rzz, 2, 3]}
    for i in range(6):
        func = apply_dic[i][0]
        if len(apply_dic[i]) == 2:
            func(thetas[i], apply_dic[i][1])
        elif len(apply_dic[i]) == 3:
            func(thetas[i], apply_dic[i][1], apply_dic[i][2])
    backend = Aer.get_backend('statevector_simulator')
    job_result = execute(circ, backend).result().get_statevector(circ)
    probs = Statevector(job_result).probabilities(qargs=[2])
    return probs[0] - probs[1]


def get_grad(thetas):
    M = np.zeros((6, 6))
    for i in range(6):
        for j in range(i, 6):
            res = []
            qc = make_circ_m(thetas, i, j, 0)
            res.append(measure_qc_statevec(qc))
            qc = make_circ_m(thetas, i, j, 1)
            res.append(measure_qc_statevec(qc))
            M[i, j] = 2 * res[0] - 2 * res[1]
            M[j, i] = 2 * res[0] - 2 * res[1]
    V = []
    for i in range(6):
        res = 0
        for j in range(3):
            coeff = 0.25 if j == 2 else 1
            qc = make_circ_xxzz(thetas, i, j, 0)
            res = res - 2 * coeff * measure_qc_statevec(qc)
            qc = make_circ_xxzz(thetas, i, j, 1)
            res = res + 2 * coeff * measure_qc_statevec(qc)
        for j in range(2):
            qc = make_circ_xy1(thetas, i, j, 0)
            res = res - 2 * 0.25 * measure_qc_statevec(qc)
            qc = make_circ_xy1(thetas, i, j, 1)
            res = res - 2 * 0.25 * measure_qc_statevec(qc)
        for j in range(4):
            qc = make_circ_xx(thetas, i, j)
            res = res - 0.25 * 2 * measure_qc_statevec(qc)
        for j in range(4):
            coeff = (-1) ** j
            qc = make_circ_xy(thetas, i, j)
            res = res - 0.25 * 2 * coeff * measure_qc_statevec(qc)
        qc = make_circ_rho(thetas, i)
        res = res + 2 * measure_qc_statevec(qc)
        V.append(res)
    M = M + np.eye(6) * 1.e-8
    grad_vec = linalg.solve(M, V)
    return grad_vec
