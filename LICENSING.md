# Licensing and provenance

## Original Qiskit scripts

Tong Jiang confirms that he independently wrote the implementations in
`test1.py` and `evolve_mix.py`. They implement methods described in
[Yuan et al., Quantum 3, 191 (2019)](https://doi.org/10.22331/q-2019-10-07-191).
The paper is the scientific source for the methods; the scripts are Tong Jiang's
own implementations.

As of October 12, 2026, Tong Jiang licenses these two scripts under the
[MIT License](LICENSES/MIT.txt), with the notice `Copyright (c) 2022 Tong Jiang`.
This grant includes the following historical snapshots, even though those
snapshots do not contain an embedded license header:

| File | Historical commit | Git blob |
| --- | --- | --- |
| [test1.py](https://github.com/jiangtong1000/learn_QuantumCompute/blob/af36a484fe02cd21d3c44ebeba785825570a1c54/test1.py) | `af36a484fe02cd21d3c44ebeba785825570a1c54` | `e9309d8f1a036908b6307e659f336482703ce700` |
| [evolve_mix.py](https://github.com/jiangtong1000/learn_QuantumCompute/blob/d4647e192f7321b83c8b28df9fb4aa940232a585/evolve_mix.py) | `d4647e192f7321b83c8b28df9fb4aa940232a585` | `2ef496ecf5d3cdf125297b3e20c0b2505eaf6702` |

When redistributing those snapshots, include the MIT license and copyright
notice above. Their historical Git contents are unchanged.

## NumPy reference example

The code and documentation in `examples/numpy_variational_dynamics/` are covered
by that directory's [MIT license](examples/numpy_variational_dynamics/LICENSE),
with the notice `Copyright (c) 2026 Tong Jiang`. Its README records the numerical
settings and AI assistance in the rewrite and release preparation.

## Scope and credit

The grants above cover only the named scripts and the NumPy example directory.
They do not apply to other repository files, notebooks, figures, external papers,
or dependencies. Imported libraries, including Qiskit, NumPy, and SciPy, retain
their own licenses.

Preferred project credit: **Tong Jiang / learn_QuantumCompute**. For research
attribution, please identify the repository, file path, and commit used, and cite
the relevant scientific reference. This citation request is separate from the
MIT license conditions.
