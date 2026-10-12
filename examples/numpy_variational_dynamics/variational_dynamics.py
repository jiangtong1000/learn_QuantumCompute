# Copyright (c) 2026 Tong Jiang
# SPDX-License-Identifier: MIT

"""Variational quantum dynamics with McLachlan's principle, in plain NumPy.

See README.md in this directory for model definitions and numerical settings.

Exact expectation values stand in for the Hadamard- and swap-test estimates a quantum
computer would provide, so the script isolates the variational equations themselves.

1. Pure state: |phi> = Rz(tz) Rx(tx) |0>, H = Y, with and without the global-phase
   correction (Yuan et al., Quantum 3, 191 (2019), Fig. 2).
2. Mixed state: two-qubit Ising model H = X0 + X1 + Z0 Z1 / 4 with amplitude damping
   L_k = |0><1| on each qubit, using a purified ansatz on two system and two ancilla qubits.

Run with: python3 variational_dynamics.py   (needs only NumPy)
"""
import numpy as np

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.diag([1.0 + 0j, -1.0])


def rx(t):
    return np.cos(t / 2) * I2 - 1j * np.sin(t / 2) * X


def rz(t):
    return np.cos(t / 2) * I2 - 1j * np.sin(t / 2) * Z


def rk4(f, y, dt):
    k1 = f(y)
    k2 = f(y + dt / 2 * k1)
    k3 = f(y + dt / 2 * k2)
    k4 = f(y + dt * k3)
    return y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


# ------------------------------------------------------------------ pure state
KET0 = np.array([1, 0], dtype=complex)


def phi(th):
    return rz(th[1]) @ rx(th[0]) @ KET0


def dphi(th):
    """Analytic derivatives, using dR/dt = -(i/2) sigma R for a Pauli rotation."""
    return [rz(th[1]) @ (-0.5j * X) @ rx(th[0]) @ KET0,
            (-0.5j * Z) @ rz(th[1]) @ rx(th[0]) @ KET0]


def pure_rhs(th, H, corrected):
    """Solve sum_j M_ij dtheta_j/dt = V_i for the parameter velocities."""
    p, d = phi(th), dphi(th)
    n = len(th)
    M = np.array([[np.vdot(d[i], d[j]).real for j in range(n)] for i in range(n)])
    V = np.array([np.vdot(d[i], H @ p).imag for i in range(n)])
    if corrected:
        a = np.array([np.vdot(p, d[i]).imag for i in range(n)])  # <phi|d_i phi> = i a_i
        energy = np.vdot(p, H @ p).real
        M = M - np.outer(a, a)
        V = V + a * energy
    return np.linalg.solve(M, V)


def run_pure(T=4.0, dt=0.01):
    th0 = np.array([0.1734, 0.3909])
    psi0 = phi(th0)
    times = np.arange(0, T + 1e-9, dt)
    fidelity = {}
    for corrected in (True, False):
        th, fid = th0.copy(), []
        for t in times:
            exact = np.cos(t) * psi0 - 1j * np.sin(t) * (Y @ psi0)  # exp(-iYt) psi0
            fid.append(abs(np.vdot(exact, phi(th))) ** 2)
            th = rk4(lambda y: pure_rhs(y, Y, corrected), th, dt)
        fidelity[corrected] = np.array(fid)
    return times, fidelity


# ------------------------------------------------------------------ mixed state
# Qubit order (a0, a1, s0, s1): ancillas a0, a1 purify the system qubits s0, s1.
P0 = np.diag([1, 0]).astype(complex)
P1 = np.diag([0, 1]).astype(complex)


def kron(*ops):
    out = np.array([[1.0 + 0j]])
    for o in ops:
        out = np.kron(out, o)
    return out


def on(single, q):
    ops = [I2] * 4
    ops[q] = single
    return kron(*ops)


def crx(t, control, target):
    return on(P0, control) + on(P1, control) @ on(rx(t), target)


def rzz(t, q1, q2):
    return np.cos(t / 2) * np.eye(16) - 1j * np.sin(t / 2) * (on(Z, q1) @ on(Z, q2))


# One layer: ancilla-controlled X rotations create the mixing; then ZZ, X, X, ZZ rotations.
GATES = [lambda t: crx(t, 0, 2), lambda t: crx(t, 1, 3), lambda t: rzz(t, 2, 3),
         lambda t: on(rx(t), 2), lambda t: on(rx(t), 3), lambda t: rzz(t, 2, 3)]
PLUS = np.array([1, 1], dtype=complex) / np.sqrt(2)
PSI0 = np.kron(np.kron(PLUS, PLUS), np.kron(KET0, KET0))  # |+>|+>|0>|0>


def rho_system(th):
    v = PSI0
    for gate, t in zip(GATES, th):
        v = gate(t) @ v
    m = np.outer(v, v.conj()).reshape(4, 4, 4, 4)  # (ancillas, system, ancillas', system')
    return np.einsum('aiaj->ij', m)


H_SYS = kron(X, I2) + kron(I2, X) + 0.25 * kron(Z, Z)
LOWER = (X + 1j * Y) / 2  # |0><1|: decay towards |0>
JUMPS = [kron(LOWER, I2), kron(I2, LOWER)]


def lindblad(r):
    out = -1j * (H_SYS @ r - r @ H_SYS)
    for L in JUMPS:
        LdL = L.conj().T @ L
        out += L @ r @ L.conj().T - 0.5 * (LdL @ r + r @ LdL)
    return out


def mixed_rhs(th, h=1e-6):
    """M_ij = Tr[(d_i rho)^+ d_j rho], V_i = Tr[(d_i rho)^+ L(rho)]."""
    r = rho_system(th)
    d = []
    for i in range(len(th)):
        e = np.zeros_like(th)
        e[i] = h
        d.append((rho_system(th + e) - rho_system(th - e)) / (2 * h))
    n = len(th)
    M = np.array([[np.trace(d[i].conj().T @ d[j]).real for j in range(n)] for i in range(n)])
    V = np.array([np.trace(d[i].conj().T @ lindblad(r)).real for i in range(n)])
    return np.linalg.lstsq(M + 1e-10 * np.eye(n), V, rcond=None)[0]


def run_mixed(T=10.0, dt=0.01):
    Z0 = kron(Z, I2)
    times = np.arange(0, T + 1e-9, dt)
    th = np.zeros(6)
    r = np.diag([1, 0, 0, 0]).astype(complex)  # |00><00|, matching th = 0
    z_exact, z_var = [], []
    for _ in times:
        z_var.append(np.trace(Z0 @ rho_system(th)).real)
        z_exact.append(np.trace(Z0 @ r).real)
        th = rk4(mixed_rhs, th, dt)
        r = rk4(lindblad, r, dt)
    return times, np.array(z_exact), np.array(z_var)


if __name__ == "__main__":
    t, fid = run_pure()
    print(f"pure state, t = {t[-1]:.1f}: fidelity {fid[True][-1]:.4f} with correction, "
          f"{fid[False][-1]:.4f} without (minimum {fid[False].min():.3f})")
    t2, z_exact, z_var = run_mixed()
    print(f"mixed state, t = {t2[-1]:.1f}: max |<Z0> error| = {np.abs(z_exact - z_var).max():.4f}, "
          f"final <Z0> {z_exact[-1]:.4f} reference vs {z_var[-1]:.4f} variational")
