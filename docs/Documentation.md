# Theoretical documentation: 2D topology optimization engine

This document covers the mathematical and numerical formulations implemented in `physics_engine/simp_engine_2d.py` and `physics_engine/ground_structure.py`. All quantities use a consistent structural unit system: lengths in $\text{mm}$, forces in $\text{N}$, stresses and elastic moduli in $\text{MPa}$ ($\text{N/mm}^2$), and displacements in $\text{mm}$.

## 1. Topology invariance under load reversal

In linear elastic compliance minimization ($\alpha = 0$), applying a load $F$ or its opposite $-F$ produces the exact same optimized topology. From Hooke's law:

$$
K(x_{\text{Phys}}) u = F
$$

Reversing the force to $-F$ yields the displacement field $-u$, leaving the compliance (external work) unchanged:

$$
C_{\text{new}} = (-F)^T (-u) = F^T u = C
$$

Because the base material is isotropic and symmetric in tension and compression ($E_t = E_c = E_0$), compliance minimization alone cannot distinguish between compressed struts and tensile ties. Breaking this symmetry requires introducing geometric non-linearities through linearized buckling analysis (LBA).

## 2. Finite element formulation (Q4 plane stress)

The rectangular design domain of size $\text{nelx} \times \text{nely}$ is discretized into regular 4-node bilinear quadrilateral (Q4) elements of unit side ($1\,\text{mm} \times 1\,\text{mm}$) and unit thickness ($t = 1\,\text{mm}$). Each node has 2 translational degrees of freedom ($u_x, u_y$), giving 8 DOFs per element and $2(\text{nelx}+1)(\text{nely}+1)$ global DOFs.

Under plane stress conditions, the baseline constitutive matrix for unit Young's modulus ($E_0 = 1$) is:

$$
D_0 = \frac{1}{1 - \nu^2}
\begin{bmatrix}
1 & \nu & 0 \\
\nu & 1 & 0 \\
0 & 0 & \frac{1-\nu}{2}
\end{bmatrix}
$$

The $8 \times 8$ solid element stiffness matrix $k_0$ is pre-computed once at initialization via exact $2 \times 2$ Gauss quadrature:

$$
k_0 = \int_{-1}^{1} \int_{-1}^{1} B(\xi, \eta)^T D_0 B(\xi, \eta) \det(J) \, d\xi \, d\eta
$$

## 3. Three-field SIMP formulation (density filter + Heaviside projection)

To prevent checkerboard instabilities, mesh dependency, and blurry grey boundaries, the solver uses three density fields ($x \to \tilde{x} \to x_{\text{Phys}}$):

1. Design variables $x_e \in [0, 1]$, updated by the optimizer.
2. Filtered densities $\tilde{x}_e$, obtained via a spatial cone filter of radius $r_{\min}$:
   $$
   \tilde{x}_e = \frac{\sum_{i \in \mathcal{N}_e} H_{ei} x_i}{\sum_{i \in \mathcal{N}_e} H_{ei}}, \qquad H_{ei} = \max\big(0,\, r_{\min} - \text{dist}(e, i)\big)
   $$
3. Physical densities $x_{\text{Phys}, e}$, obtained by projecting $\tilde{x}_e$ through a smoothed Heaviside step function centered at $\eta = 0.5$ with sharpness parameter $\beta$:
   $$
   x_{\text{Phys}, e}(\tilde{x}_e) = \frac{\tanh(\beta \eta) + \tanh\!\big(\beta (\tilde{x}_e - \eta)\big)}{\tanh(\beta \eta) + \tanh\!\big(\beta (1 - \eta)\big)}
   $$

The derivative of the projection with respect to the filtered density is:

$$
\frac{\partial x_{\text{Phys}, e}}{\partial \tilde{x}_e} = \frac{\beta \left(1 - \tanh^2\!\big(\beta (\tilde{x}_e - \eta)\big)\right)}{\tanh(\beta \eta) + \tanh\!\big(\beta (1 - \eta)\big)}
$$

A continuation scheme starts at $\beta = 1.0$ and doubles $\beta$ every 25 iterations (up to $\beta_{\max} = 8.0$), allowing the layout to form gradually before converging to a crisp $0/1$ topology.

### Chain rule for sensitivities

All physical quantities (stiffness matrices and total volume $V = \sum_e x_{\text{Phys}, e}$) are evaluated on $x_{\text{Phys}}$. Any derivative with respect to $x_{\text{Phys}}$ (both objective $\partial f / \partial x_{\text{Phys}}$ and volume $\partial V / \partial x_{\text{Phys}} = 1$) is mapped back to the design variables $x$ via the chain rule:

$$
\frac{\partial f}{\partial x_i} = \sum_{e \in \mathcal{N}_i} \frac{H_{ie}}{\sum_k H_{ek}} \left( \frac{\partial f}{\partial x_{\text{Phys}, e}} \cdot \frac{\partial x_{\text{Phys}, e}}{\partial \tilde{x}_e} \right)
$$

### SIMP interpolation and optimality criteria (OC) bisection

Element elastic stiffness follows the standard SIMP interpolation:

$$
E_e(x_{\text{Phys}, e}) = E_{\min} + x_{\text{Phys}, e}^p (E_0 - E_{\min}), \qquad E_{\min} = 10^{-9} E_0, \quad p = 3.0
$$

The optimality criteria (OC) update uses a bisection loop on the Lagrange multiplier $\lambda_{\text{bisect}}$:

$$
B_e = \sqrt{\max\!\left(0,\, -\frac{\partial f / \partial x_e}{\lambda_{\text{bisect}} \, (\partial V / \partial x_e)}\right)}, \qquad x_e^{\text{new}} = \text{clip}\!\left(x_e B_e,\, \max(0, x_e - m),\, \min(1, x_e + m)\right)
$$

Inside each bisection step, the candidate vector $x^{\text{new}}$ is passed through both the spatial filter $H$ and the Heaviside projection so that the volume constraint $\frac{1}{N_e}\sum_e x_{\text{Phys}, e}^{\text{new}} = V_f$ is satisfied on the true physical field, while enforcing passive solid ($\text{pasS} = 1$) and passive void ($\text{pasV} = 0$) masks.

## 4. Linearized buckling analysis (LBA) and KS aggregation

Following Ferrari, Sigmund, and Guest (2021), linear buckling is formulated as a generalized eigenvalue problem:

$$
\left( K(x_{\text{Phys}}) + \lambda K_G(x_{\text{Phys}}, u) \right) \phi = 0 \quad \Longleftrightarrow \quad K_G \phi_i = \mu_i K \phi_i, \qquad \mu_i = \frac{1}{\lambda_i}
$$

where $\lambda_1 = 1 / \mu_1$ is the fundamental buckling load factor (BLF).

### Green-Lagrange geometric stiffness matrix ($K_G$)

Evaluating the non-linear Green-Lagrange strain terms ($u_{,x}^2 + v_{,x}^2$, $u_{,y}^2 + v_{,y}^2$, and cross terms) at the element centroid ($\xi = 0, \eta = 0$), the $8 \times 8$ element geometric stiffness matrix $k_{G,e}$ is written as a linear combination of three constant base matrices $T_x, T_y, T_{xy}$ weighted by the centroidal Cauchy stresses $(\sigma_{x,e}, \sigma_{y,e}, \tau_{xy,e}) = D_0 B_0 u_e$:

$$
k_{G,e}(\sigma_e) = x_{\text{Phys}, e}^{p_G} \left( \sigma_{x,e} T_x + \sigma_{y,e} T_y + \tau_{xy,e} T_{xy} \right)
$$

This allows assembling the global sparse matrix $K_G$ with pure vectorized array operations.

### Differential penalization against spurious void modes

In low-density regions ($x_{\text{Phys}, e} \to 0$), localized artificial buckling modes appear if elastic and geometric stiffness scale with the same exponent. To suppress them, geometric stiffness uses a higher exponent $p_G = 6.0 > p_K = 3.0$ and zero offset in void elements ($E_G = x_{\text{Phys}, e}^{p_G}$). As $x_{\text{Phys}, e} \to 0$, the local stiffness ratio scales as $x_{\text{Phys}, e}^{p_K - p_G} = x_{\text{Phys}, e}^{-3} \to \infty$, pushing void modes out of the fundamental spectrum.

### Shift-invert eigensolver and sparse LU reuse

To extract the $N_{\text{eig}} = 4$ most critical buckling modes ($\mu_1 \ge \mu_2 \ge \dots \ge \mu_{N_{\text{eig}}} > 0$), the solver runs `scipy.sparse.linalg.eigsh` on the operator $v \mapsto K_{\text{free}}^{-1} K_{G,\text{free}} v$. The elastic matrix $K_{\text{free}}$ is factorized once per iteration via SuperLU (`sla.factorized(K_free)`), and the cached LU factors are reused across all Arnoldi iterations and all $N_{\text{eig}}$ adjoint right-hand sides.

### Kreisselmeier-Steinhauser (KS) multi-modal aggregation

Optimizing a single eigenvalue $\mu_1$ leads to oscillations when adjacent modes cross ($\mu_1 \approx \mu_2$). The first $N_{\text{eig}}$ eigenvalues are aggregated via the log-sum-exp stabilized Kreisselmeier-Steinhauser function ($\rho_{\text{KS}} = 50$):

$$
J_{\text{KS}}(\mu) = \mu_1 + \frac{1}{\rho_{\text{KS}}} \ln \left( \sum_{i=1}^{N_{\text{eig}}} \exp\!\big(\rho_{\text{KS}} (\mu_i - \mu_1)\big) \right)
$$

with modal weights:

$$
w_{\text{KS}, i} = \frac{\exp\!\big(\rho_{\text{KS}} (\mu_i - \mu_1)\big)}{\sum_{j=1}^{N_{\text{eig}}} \exp\!\big(\rho_{\text{KS}} (\mu_j - \mu_1)\big)}, \qquad \sum_{i=1}^{N_{\text{eig}}} w_{\text{KS}, i} = 1
$$

For each mode $\phi_i$ normalized to $\phi_i^T K \phi_i = 1$, the adjoint vector $w_i$ solves:

$$
K w_i = \sum_e x_{\text{Phys}, e}^{p_G} B_0^T D_0^T
\begin{bmatrix}
\phi_{i,e}^T T_x \phi_{i,e} \\
\phi_{i,e}^T T_y \phi_{i,e} \\
\phi_{i,e}^T T_{xy} \phi_{i,e}
\end{bmatrix}
$$

giving the aggregated sensitivity with respect to physical density:

$$
\frac{\partial J_{\text{KS}}}{\partial x_{\text{Phys}, e}} = \sum_{i=1}^{N_{\text{eig}}} w_{\text{KS}, i} \left( \phi_{i,e}^T \frac{\partial k_{G,e}}{\partial x_{\text{Phys}, e}} \phi_{i,e} - \mu_i \phi_{i,e}^T \frac{\partial k_{E,e}}{\partial x_{\text{Phys}, e}} \phi_{i,e} - w_{i,e}^T \frac{\partial k_{E,e}}{\partial x_{\text{Phys}, e}} u_e \right)
$$

### Multi-objective Pareto scalarization

Following Marler & Arora (2004), the chain-rule-filtered compliance gradient $g_c = \partial C / \partial x$ and buckling gradient $g_{\text{KS}} = \partial J_{\text{KS}} / \partial x$ are normalized by their $L_1$ norms and combined via the weight $\alpha \in [0, 0.85]$:

$$
s_{\text{eff}, e} = (1 - \alpha) \frac{g_{c,e}}{\|g_c\|_1} + \alpha \frac{\min(0, g_{\text{KS}, e})}{\|\min(0, g_{\text{KS}})\|_1}
$$

## 5. Principal stress evaluation and dual-view visualization

From the centroidal stress vector $\sigma_e = [\sigma_{x,e}, \sigma_{y,e}, \tau_{xy,e}]^T = D_0 B_0 u_e$, the in-plane principal stresses are:

$$
\sigma_{I, e} = \frac{\sigma_{x,e} + \sigma_{y,e}}{2} + \sqrt{\left(\frac{\sigma_{x,e} - \sigma_{y,e}}{2}\right)^2 + \tau_{xy,e}^2}, \qquad
\sigma_{II, e} = \frac{\sigma_{x,e} + \sigma_{y,e}}{2} - \sqrt{\left(\frac{\sigma_{x,e} - \sigma_{y,e}}{2}\right)^2 + \tau_{xy,e}^2}
$$

The post-processor displays two synchronized views:
1. Binary black-and-white topology ($x_{\text{Phys}, e} \ge \rho_{\text{threshold}}$).
2. Void-masked minimum principal stress $\sigma_{II, e}$ using the `Jet_r` colormap (voids masked as `NaN`, compressed members in red/orange, neutral/tensile members in blue).

## 6. Discrete ground structure solver (`physics_engine/ground_structure.py`)

Alongside the continuum SIMP solver, the repository includes a 2D discrete truss topology optimizer based on the ground structure method:
1. Generates a nodal grid and all candidate bar connections up to a maximum length $L_{\max}$, filtering out collinear overlaps via GCD direction vectors.
2. Assembles the global bar stiffness matrix in global coordinates using direction cosines $(c, s) = (\cos\theta, \sin\theta)$.
3. Iteratively updates cross-sectional areas $A_i$ via Fully Stressed Design (FSD) and strain-energy rescaling under total volume conservation, pruning bars that drop below the cutoff threshold.

## References

1. Ferrari, F., Sigmund, O., & Guest, J. K. (2021). Topology optimization with linearized buckling criteria in 250 lines of Matlab. *Structural and Multidisciplinary Optimization*, 63, 3045-3066.
2. Sigmund, O. (2001). A 99 line topology optimization code written in Matlab. *Structural and Multidisciplinary Optimization*, 21(2), 120-127.
3. Wang, F., Lazarov, B. S., & Sigmund, O. (2011). On projection methods, convergence and robust formulations in topology optimization. *Structural and Multidisciplinary Optimization*, 43(6), 767-784.
4. Kreisselmeier, G., & Steinhauser, R. (1979). Systematic control design by optimizing a vector performance index. *IFAC Proceedings Volumes*, 12(7), 113-117.
5. Marler, R. T., & Arora, J. S. (2004). Survey of multi-objective optimization methods for engineering. *Structural and Multidisciplinary Optimization*, 26(6), 369-395.
