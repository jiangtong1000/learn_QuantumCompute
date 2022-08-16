# -*- coding: utf-8 -*-

import numpy as np
from qiskit import QuantumCircuit
from qiskit import transpile
from qiskit.providers.aer import AerSimulator, Aer
from scipy import linalg
from qiskit import execute

shots = 50000


def measure_A(theta1, theta2, printqc=False):
    circ = QuantumCircuit(2)
    circ.h(0)
    circ.cx(0, 1)
    circ.rx(theta1, 1)
    circ.x(0)
    circ.cz(0, 1)
    circ.rz(theta2, 1)
    circ.h(0)
    meas = QuantumCircuit(2, 1)
    meas.barrier(range(2))
    meas.measure(0, 0)
    qc = meas.compose(circ, front=True)
    if printqc:
        return qc
    else:
        backend = AerSimulator()
        qc_compiled = transpile(qc, backend)
        job_sim = backend.run(qc_compiled, shots=shots)
        job_result = job_sim.result()
        counts = job_result.get_counts()
        if "1" not in counts:
            return 1
        elif "0" not in counts:
            return -1
        else:
            return (counts['0']-counts['1']) / (counts['0'] + counts['1'])


def measure_C1(theta1, theta2, printqc=False):
    circ = QuantumCircuit(2)
    circ.h(0)
    circ.cx(0, 1)
    circ.x(0)
    circ.rx(theta1, 1)
    circ.rz(theta2, 1)
    circ.cy(0, 1)
    circ.h(0)
    meas = QuantumCircuit(2, 1)
    meas.barrier(range(2))
    meas.measure(0, 0)
    qc = meas.compose(circ, front=True)
    if printqc:
        return qc
    else:
        backend = AerSimulator()
        qc_compiled = transpile(qc, backend)
        job_sim = backend.run(qc_compiled, shots=shots)
        job_result = job_sim.result()
        counts = job_result.get_counts()
        if "1" not in counts:
            return 1
        elif "0" not in counts:
            return -1
        else:
            return (counts['0']-counts['1']) / (counts['0'] + counts['1'])


def measure_C2(theta1, theta2, printqc=False):
    circ = QuantumCircuit(2)
    circ.h(0)
    circ.rx(theta1, 1)
    circ.cz(0, 1)
    circ.x(0)
    circ.rz(theta2, 1)
    circ.cy(0, 1)
    circ.h(0)
    meas = QuantumCircuit(2, 1)
    meas.barrier(range(2))
    meas.measure(0, 0)
    qc = meas.compose(circ, front=True)
    if printqc:
        return qc
    else:
        backend = AerSimulator()
        qc_compiled = transpile(qc, backend)
        job_sim = backend.run(qc_compiled, shots=shots)
        job_result = job_sim.result()
        counts = job_result.get_counts()
        if "1" not in counts:
            return 1
        elif "0" not in counts:
            return -1
        else:
            return (counts['0']-counts['1']) / (counts['0'] + counts['1'])


def measure_partial_zi(theta1, theta2, printqc=False):
    circ = QuantumCircuit(2)
    circ.h(0)
    circ.rx(theta1, 1)
    circ.cz(0, 1)
    circ.h(0)
    circ.rz(theta2, 1)
    meas = QuantumCircuit(2, 1)
    meas.barrier(range(2))
    meas.measure(0, 0)
    qc = meas.compose(circ, front=True)
    if printqc:
        return qc
    else:
        backend = AerSimulator()
        qc_compiled = transpile(qc, backend)
        job_sim = backend.run(qc_compiled, shots=shots)
        job_result = job_sim.result()
        counts = job_result.get_counts()
        if "1" not in counts:
            return 1
        elif "0" not in counts:
            return -1
        else:
            return (counts['0']-counts['1']) / (counts['0'] + counts['1'])
def measure_e(theta1, theta2, printqc=False):
    circ = QuantumCircuit(2)
    circ.h(0)
    circ.rx(theta1, 1)
    circ.rz(theta2, 1)
    circ.cy(0, 1)
    circ.h(0)
    meas = QuantumCircuit(2, 1)
    meas.barrier(range(2))
    meas.measure(0, 0)
    qc = meas.compose(circ, front=True)
    if printqc:
        return qc
    else:
        backend = AerSimulator()
        qc_compiled = transpile(qc, backend)
        job_sim = backend.run(qc_compiled, shots=shots)
        job_result = job_sim.result()
        counts = job_result.get_counts()
        if "1" not in counts:
            return 1
        elif "0" not in counts:
            return -1
        else:
            return (counts['0']-counts['1']) / (counts['0'] + counts['1'])


def get_state(theta1, theta2, printqc=False):
    circ = QuantumCircuit(1)
    circ.rx(theta1, 0)
    circ.rz(theta2, 0)
#     circ.measure(0, 0)
    backend = Aer.get_backend('statevector_simulator')
    sv1 = execute(circ, backend).result().get_statevector(circ)
    return sv1


def exact_evolve(t, printqc=False):
    circ = QuantumCircuit(1)
    theta1 = 0.1734; theta2= 0.3909
    circ.rx(theta1, 0)
    circ.rz(theta2, 0)
    circ.ry(t*2, 0)
#     circ.measure(0, 0)
    backend = Aer.get_backend('statevector_simulator')
    sv1 = execute(circ, backend).result().get_statevector(circ)
    return sv1


def grad_theta(theta1, theta2, correct):
    A12 = measure_A(theta1, theta2)
    C1 = measure_C1(theta1, theta2)
    C2 = measure_C2(theta1, theta2)
    B = measure_partial_zi(theta1, theta2)
    E = measure_e(theta1, theta2)
    if correct:
        mat = np.array([[0.25, A12*0.25], [A12*0.25, 0.25-0.25*B**2]])
        vec = np.array([C1 * (0.5), C2 * (0.5)-0.5*B*E])
    else:
        mat = np.array([[0.25, A12*0.25], [A12*0.25, 0.25]])
        vec = np.array([C1 * (0.5), C2 * (0.5)])
    return linalg.solve(mat, vec)
