# 2D topology optimization with linearized buckling analysis

2D continuum topology optimization solver based on the SIMP method, with linearized buckling analysis and Kreisselmeier-Steinhauser multi-modal aggregation. Includes an interactive Streamlit dashboard.

## Implemented features

- Vectorized Q4 plane stress FEM (`physics_engine/simp_engine_2d.py`):
  - 4-node bilinear quadrilateral elements with pre-computed 2x2 Gauss stiffness matrix ($k_0$).
  - Sparse CSC assembly (`scipy.sparse.csc_matrix`) and reusable sparse LU factorization (`scipy.sparse.linalg.factorized`), computed once per iteration and shared across static equilibrium, shift-invert Arnoldi iterations, and adjoint solves.
- Three-field SIMP and exact chain rule:
  - Cone density filter ($\tilde{x} = Hx / H_s$) followed by a smoothed Heaviside projection ($x_{\text{Phys}}$) with automatic $\beta$-continuation.
  - Full chain rule applied to both objective sensitivities ($\partial f / \partial x_e$) and volume sensitivities ($\partial V / \partial x_e$).
  - Optimality criteria (OC) bisection loop evaluating the true projected physical volume at every sub-iteration, with support for passive solid and void regions.
- LBA and KS aggregation:
  - Geometric stiffness matrix $K_G(x_{\text{Phys}}, u)$ assembled from the non-linear Green-Lagrange strain tensor ($u_{,x}^2 + v_{,x}^2$) using pre-computed centroidal matrices ($T_x, T_y, T_{xy}$).
  - Differential penalization ($p_G = 6 > p_K = 3$ and zero geometric stiffness in void regions) to suppress spurious buckling modes in low-density elements.
  - Option to optimize either the fundamental mode ($n_{\text{eig}} = 1$) or the Kreisselmeier-Steinhauser (KS) multi-modal aggregate ($n_{\text{eig}} > 1$) to avoid mode switching when eigenvalues coalesce.
- Buckling-compliance trade-off and post-processing (`app.py`):
  - Continuous weighting $\alpha$ between compliance ($C$) and critical buckling load factor ($\text{BLF} = \lambda_1$) using dynamic $L_1$ gradient normalization.
  - Sweeping $\alpha \in [0, 1]$.
  - Dual-view visualization: black-and-white structural layout alongside a void-masked minimum principal stress ($\sigma_{II}$) map (`Jet_r` colormap: red for compression, blue for tension).

## Man-in-the-loop development workflow

This repository was developed with a human-in-the-loop approach: mathematical formulation, literature selection, and physical validation are handled by the engineer, while the AI agent is used strictly for code translation, vectorization, and UI wiring.

```text
[1. Theoretical formulation (human)]
   Literature review
   Analytical derivation of FEM, LBA, etc...
        |
        v
[2. Engineering specification (human)]
   Definition of matrix equations, boundary conditions, and algorithmic structure
        |
        v
[3. Code implementation & vectorization (Agents)]
   Translation into vectorized NumPy/SciPy routines and Streamlit UI
        |
        v
[4. Physical validation & code review (human)]
   Verification against closed-form benchmarks, Green-Lagrange tensor check,
   chain-rule consistency check, and exact physical volume conservation in OC
        |
        +---> Numerical or syntax bug ----------> back to [3. Code implementation (AI)]
        +---> Physical or mathematical issue ---> back to [1. Theoretical formulation (human)]
```

## Validation & changelog

Cross-checking the solver against external references (Altair Inspire on an Aluminum 6061-T6 $400 \times 200 \times 5\,\text{mm}$ cantilever and closed-form Timoshenko beam theory) identified two hardcoded UI parameters, a latent $E_0^2$ scaling bug, and an unconverged continuation step:

- **Fixed double-$E_0$ scaling bug (`physics_engine/simp_engine_2d.py`)**: `_build_element_stiffness()` (`k0`) and `_init_buckling_matrices()` (`Cmat0`) previously multiplied by `self.E0` at initialization while `solve()` multiplied again by `E_elements ~ E0` and `E_G ~ E0`, causing elastic stiffness $K$ to scale as $E_0^2$ whenever $E_0 \neq 1$. Both base matrices are now assembled at unit modulus ($E = 1.0$) with dimensionless void ratio $E_{\min}/E_0 = 10^{-9}$, restoring exact invariance of $C \cdot E_0 \cdot t$ and $\text{BLF} / (E_0 \cdot t)$.
- **Exposed physical parameters in UI (`app.py`)**: Replaced hardcoded `E0 = 1.0 MPa`, `thickness = 10.0 mm`, `nu = 0.30`, and `max_iter = 45` with explicit inputs defaulting to Aluminum 6061-T6 (`E0 = 68900 MPa`, `nu = 0.33`, `thickness = 5.0 mm`, `max_iter = 100`).
- **$\beta$-continuation, OC damping, and compliance-based stopping criterion**: Prevented $\beta$ from jumping on the final iteration without equilibration, added OC damping (`oc_damping = 0.3`, `move = 0.1` for $\beta \ge 2$) to eliminate boundary flip-flopping, implemented early stopping on relative compliance stability over 10 iterations evaluated at maximum $\beta$ ($|C(it) - C(it - 10)| / C(it) < \text{tol}_C = 0.001$), checked objective stability over the last 20 iterations ($< 0.1\%$), and restricted `"converged after N iterations"` in `status_message` to runs where convergence genuinely triggered (reporting `"stopped after N iterations"` when `max_iter` is reached without meeting convergence criteria).
- **Filter radius length-scale note**: Added UI and documentation notes clarifying that $r_{\min}$ is specified in $\text{mm}$; setting $r_{\min} = 2.5\,\text{mm}$ on a $dx = 2.0\,\text{mm}$ mesh ($r_{\min}/dx = 1.25$ elements) provides minimal filtering and permits mesh-scale fragments, whereas mesh independence requires $r_{\min} \approx 2.5\text{--}3$ element lengths ($5\text{--}6\,\text{mm}$).
- **Automated `pytest` verification suite (`tests/test_simp_engine.py`)**:
  1. *Frozen-settings regression*: $200 \times 100$ cantilever ($E_0 = 68900\,\text{MPa}, t = 5\,\text{mm}, 45$ iterations, legacy $\beta$) reproduces $C = 392.0\,\text{N}\cdot\text{mm}$ ($30\,\text{mm}$ strip load, $\nu = 0.33$) and $C = 418.01\,\text{N}\cdot\text{mm}$ (point load, $\nu = 0.30$).
  2. *Scale invariance*: $(E_0 = 1, t = 10)$ vs $(E_0 = 68900, t = 5)$ yields identical $C \cdot E_0 \cdot t$, identical $\text{BLF} / (E_0 \cdot t)$, and identical topology `density_matrix`.
  3. *Full solid vs Timoshenko beam*: Full-solid Q4 FEM cantilever matches analytical Timoshenko compliance ($C_{\text{Timoshenko}} = \frac{F^2 L^3}{3 E_0 I} + \frac{6 F^2 L}{5 G A} = 111.42\,\text{N}\cdot\text{mm}$) within $0.6\%$.

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
    ├── test_simp_engine.py         # Unit, regression, scale-invariance, and Timoshenko tests
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
