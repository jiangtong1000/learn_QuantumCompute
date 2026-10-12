# Variational quantum dynamics in NumPy

A small classical reference implementation of McLachlan variational dynamics for
the models explored in this repository's original 2022 Qiskit examples.

## Run

From the repository root:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r examples/numpy_variational_dynamics/requirements.txt
python examples/numpy_variational_dynamics/variational_dynamics.py
```

Tested with Python 3.12.8 and NumPy 2.4.4. The script computes both examples and
prints summaries; it does not generate figures. Typical output is:

```text
pure state, t = 4.0: fidelity 1.0000 with correction, 0.5833 without (minimum 0.133)
mixed state, t = 10.0: max |<Z0> error| = 0.0045, final <Z0> 0.1220 reference vs 0.1219 variational
```

## What it computes

- **Pure state:** a one-qubit `Rz Rx` ansatz under `H = Y`, with and without the
  global-phase correction, compared with the analytic solution. This reproduces
  the setup of Fig. 2 in [Yuan et al., Quantum 3, 191 (2019)](https://doi.org/10.22331/q-2019-10-07-191).
- **Mixed state:** `H = X0 + X1 + Z0 Z1 / 4`, with local amplitude damping
  `L_k = |0><1|`. A purified ansatz uses two system and two ancilla qubits.
  Its `⟨Z0⟩` is compared with direct numerical integration of the full Lindblad equation.

Both runs use RK4 with `dt = 0.01`, to times 4 and 10. The mixed-state solver uses
central finite differences with step `1e-6` and diagonal regularization `1e-10`.
These settings are checked for these small examples, not arbitrary models.
NumPy evaluates all overlaps and expectations directly; there are no quantum
measurement circuits or sampling noise.

## Provenance and reuse

Project credit: **Tong Jiang / learn_QuantumCompute**. The 2026 NumPy rewrite and
release preparation used AI assistance. The historical Qiskit implementations
are [test1.py](../../test1.py) and [evolve_mix.py](../../evolve_mix.py).

The [MIT license](LICENSE) covers the files in this directory. The original
`test1.py` and `evolve_mix.py` also have an MIT grant, including the historical
versions listed in [LICENSING.md](../../LICENSING.md). Other repository files
are outside these grants. NumPy is a separately licensed dependency.
When citing the example, identify its repository path and commit, and credit the
scientific reference above. This citation request is separate from the MIT terms.
