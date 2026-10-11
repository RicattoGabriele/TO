# 2D topology optimization with linearized buckling analysis

2D continuum topology optimization solver based on the three-field SIMP method (design variables -> spatial filter -> Heaviside projection), coupled with linearized buckling analysis (LBA), Kreisselmeier-Steinhauser (KS) multi-modal aggregation, and Pareto multi-objective scalarization. Includes an interactive Streamlit dashboard and a discrete ground structure truss solver.

## Implemented features

- Vectorized Q4 plane stress FEM (`physics_engine/simp_engine_2d.py`):
  - 4-node bilinear quadrilateral elements with pre-computed 2x2 Gauss stiffness matrix ($k_0$).
  - Sparse CSC assembly (`scipy.sparse.csc_matrix`) and reusable sparse LU factorization (`scipy.sparse.linalg.factorized`), computed once per iteration and shared across static equilibrium, shift-invert Arnoldi iterations, and adjoint solves.
- Three-field SIMP and exact chain rule:
  - Cone density filter ($\tilde{x} = Hx / H_s$) followed by a smoothed Heaviside projection ($x_{\text{Phys}}$) with automatic $\beta$-continuation.
  - Full chain rule applied to both objective sensitivities ($\partial f / \partial x_e$) and volume sensitivities ($\partial V / \partial x_e$).
  - Optimality criteria (OC) bisection loop evaluating the true projected physical volume at every sub-iteration, with support for passive solid and void regions.
- Linearized buckling analysis (LBA) and KS aggregation:
  - Geometric stiffness matrix $K_G(x_{\text{Phys}}, u)$ assembled from the non-linear Green-Lagrange strain tensor ($u_{,x}^2 + v_{,x}^2$) using pre-computed centroidal matrices ($T_x, T_y, T_{xy}$).
  - Differential penalization ($p_G = 6 > p_K = 3$ and zero geometric stiffness in void regions) to suppress spurious buckling modes in low-density elements.
  - Option to optimize either the fundamental mode ($n_{\text{eig}} = 1$) or the Kreisselmeier-Steinhauser (KS) multi-modal aggregate ($n_{\text{eig}} > 1$) to avoid mode switching when eigenvalues coalesce.
- Pareto trade-off and post-processing (`app.py`):
  - Continuous weighting $\alpha$ between compliance ($C$) and critical buckling load factor ($\text{BLF} = \lambda_1$) using dynamic $L_1$ gradient normalization.
  - Automated Pareto frontier sweep across $\alpha \in [0, 1]$.
  - Dual-view visualization: black-and-white structural layout alongside a void-masked minimum principal stress ($\sigma_{II}$) map (`Jet_r` colormap: red for compression, blue for tension).

## Man-in-the-loop development workflow

This repository was developed with a strict human-in-the-loop approach: mathematical formulation, literature selection, and physical validation are handled by the engineer, while the AI agent is used strictly for code translation, vectorization, and UI wiring.

```text
[1. Theoretical formulation (human)]
   Literature review (Sigmund, Ferrari, Wang, Marler & Arora)
   Analytical derivation of FEM, LBA, adjoint sensitivities, and Heaviside chain rule
        |
        v
[2. Engineering specification (human)]
   Definition of matrix equations, boundary conditions, and algorithmic structure (OC / KS / Pareto)
        |
        v
[3. Code implementation & vectorization (AI)]
   Translation into vectorized NumPy/SciPy routines, sparse LU caching, and Streamlit UI
        |
        v
[4. Physical validation & code review (human)]
   Verification against closed-form benchmarks, Green-Lagrange tensor check,
   chain-rule consistency check, and exact physical volume conservation in OC
        |
        +---> Numerical or syntax bug ----------> back to [3. Code implementation (AI)]
        +---> Physical or mathematical issue ---> back to [1. Theoretical formulation (human)]
```

## Repository structure

```text
├── app.py                          # Streamlit dashboard (2D SIMP, LBA, KS, Pareto sweep)
├── Documentation.md                # Theoretical documentation with all implemented equations
├── Documentation.tex               # LaTeX source of the theoretical documentation
├── AGENTS.md                       # Project rules and numerical standards
├── requirements.txt                # Python dependencies
├── physics_engine/
│   ├── simp_engine_2d.py           # Q4 FEM + 3-field SIMP + LBA + KS aggregator
│   ├── ground_structure.py         # Discrete truss ground structure solver
│   └── __init__.py
└── tests/
    ├── test_simp_engine.py         # Unit tests for stiffness, OC convergence, and volume constraint
    └── test_case_study_ground_structure.py
```

## Quick start

Install dependencies:
```bash
pip install -r requirements.txt
```

Run the interactive dashboard:
```bash
python -m streamlit run app.py
```

Run validation tests:
```bash
pytest tests/
```
