"""
2D SIMP Continuum Topology Optimization Engine.
Physics-driven material distribution optimizer using Plane Stress Q4 finite elements,
SIMP material interpolation with cubic penalization, spatial sensitivity filtering (Helmholtz-type),
and deterministic Optimality Criteria (OC) update loop.
"""

from dataclasses import dataclass, field
import math
import time
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla


@dataclass
class SIMPResult:
    """Result of 2D SIMP topology optimization."""
    success: bool
    status_message: str
    density_matrix: np.ndarray  # 2D array of shape (nely, nelx) with values in [0, 1]
    compliance: float
    volume_fraction: float
    iterations_run: int
    execution_time_sec: float
    compliance_history: List[float]
    change_history: List[float]
    nelx: int
    nely: int
    dx: float
    dy: float


class SIMPOptimizer2D:
    """
    Solves minimum compliance topology optimization on a 2D continuum domain.
    """

    def __init__(
        self,
        nelx: int = 60,
        nely: int = 30,
        dx: float = 1.0,
        dy: float = 1.0,
        thickness: float = 1.0,
        E0: float = 1.0,
        Emin: float = 1e-9,
        nu: float = 0.3,
        penal: float = 3.0,
        rmin: float = 2.5,
        volfrac: float = 0.4,
    ):
        """
        Args:
            nelx: Number of elements in X direction
            nely: Number of elements in Y direction
            dx: Element size in X (mm)
            dy: Element size in Y (mm)
            thickness: Out-of-plane thickness (mm)
            E0: Young's modulus of solid material (MPa)
            Emin: Minimum stiffness for void elements (avoids singularity)
            nu: Poisson's ratio
            penal: SIMP penalization exponent (typically 3.0)
            rmin: Sensitivity filter radius (in element units or physical mm)
            volfrac: Target volume fraction in [0.05, 0.95]
        """
        self.nelx = int(nelx)
        self.nely = int(nely)
        self.dx = float(dx)
        self.dy = float(dy)
        self.thickness = float(thickness)
        self.E0 = float(E0)
        self.Emin = float(Emin)
        self.nu = float(nu)
        self.penal = float(penal)
        self.rmin = float(rmin)
        self.volfrac = float(volfrac)

        self.num_elements = self.nelx * self.nely
        self.num_nodes = (self.nelx + 1) * (self.nely + 1)
        self.num_dofs = 2 * self.num_nodes

        # Element stiffness matrix k0
        self.k0 = self._build_element_stiffness()

        # Assembly indexing vectors
        self.edofMat = self._build_edof_matrix()

        # Spatial filter weights
        self.H, self.Hs = self._build_spatial_filter()

        # Boundary conditions
        self.fixed_dofs: Set[int] = set()
        self.force_vector = np.zeros(self.num_dofs, dtype=np.float64)

    def _build_element_stiffness(self) -> np.ndarray:
        """
        Exact Q4 plane stress element stiffness matrix using 2x2 Gauss quadrature.
        Returns 8x8 symmetric matrix.
        """
        E = self.E0
        nu = self.nu
        t = self.thickness
        dx = self.dx
        dy = self.dy

        # Plane stress constitutive matrix D
        factor = E / (1.0 - nu**2)
        D = factor * np.array([
            [1.0, nu, 0.0],
            [nu, 1.0, 0.0],
            [0.0, 0.0, (1.0 - nu) / 2.0]
        ], dtype=np.float64)

        k = np.zeros((8, 8), dtype=np.float64)
        gauss_pts = [-1.0 / math.sqrt(3.0), 1.0 / math.sqrt(3.0)]
        det_J = (dx * dy) / 4.0

        for xi in gauss_pts:
            for eta in gauss_pts:
                # Shape function derivatives w.r.t xi, eta
                dNdxi = np.array([
                    -0.25 * (1.0 - eta),
                     0.25 * (1.0 - eta),
                     0.25 * (1.0 + eta),
                    -0.25 * (1.0 + eta)
                ])
                dNdeta = np.array([
                    -0.25 * (1.0 - xi),
                    -0.25 * (1.0 + xi),
                     0.25 * (1.0 + xi),
                     0.25 * (1.0 - xi)
                ])

                # Derivatives w.r.t physical x, y
                dNx = dNdxi * (2.0 / dx)
                dNy = dNdeta * (2.0 / dy)

                # Strain-displacement matrix B (3x8)
                # DOFs ordered: [u1, v1, u2, v2, u3, v3, u4, v4]
                B = np.zeros((3, 8), dtype=np.float64)
                for i in range(4):
                    B[0, 2 * i] = dNx[i]
                    B[1, 2 * i + 1] = dNy[i]
                    B[2, 2 * i] = dNy[i]
                    B[2, 2 * i + 1] = dNx[i]

                k += (B.T @ D @ B) * det_J * t

        return k

    def _build_edof_matrix(self) -> np.ndarray:
        """
        Builds matrix of element DOFs of size (num_elements, 8).
        Node numbering: node (i, j) where i in [0..nelx], j in [0..nely].
        Node ID = i * (nely + 1) + j.
        Element ID = elx * nely + ely.
        """
        edofMat = np.zeros((self.num_elements, 8), dtype=np.int32)
        for elx in range(self.nelx):
            for ely in range(self.nely):
                el_idx = elx * self.nely + ely
                # 4 corner nodes of element (elx, ely):
                # n1: bottom-left (elx, ely)
                # n2: bottom-right (elx+1, ely)
                # n3: top-right (elx+1, ely+1)
                # n4: top-left (elx, ely+1)
                n1 = elx * (self.nely + 1) + ely
                n2 = (elx + 1) * (self.nely + 1) + ely
                n3 = (elx + 1) * (self.nely + 1) + (ely + 1)
                n4 = elx * (self.nely + 1) + (ely + 1)

                edofMat[el_idx, :] = [
                    2 * n1, 2 * n1 + 1,
                    2 * n2, 2 * n2 + 1,
                    2 * n3, 2 * n3 + 1,
                    2 * n4, 2 * n4 + 1
                ]
        return edofMat

    def _build_spatial_filter(self) -> Tuple[sp.csr_matrix, np.ndarray]:
        """
        Builds spatial sensitivity filter matrix H (sparse) to eliminate checkerboards
        and enforce minimum length scale rmin.
        """
        rmin = self.rmin
        nely = self.nely
        nelx = self.nelx
        num_el = self.num_elements

        iH = []
        jH = []
        sH = []

        # Loop over elements and neighbor box of radius ceil(rmin)
        r_ceil = int(math.ceil(rmin))
        for i1 in range(nelx):
            for j1 in range(nely):
                e1 = i1 * nely + j1
                imin = max(0, i1 - r_ceil)
                imax = min(nelx, i1 + r_ceil + 1)
                jmin = max(0, j1 - r_ceil)
                jmax = min(nely, j1 + r_ceil + 1)

                for i2 in range(imin, imax):
                    for j2 in range(jmin, jmax):
                        e2 = i2 * nely + j2
                        dist = math.hypot((i1 - i2) * self.dx, (j1 - j2) * self.dy)
                        weight = max(0.0, rmin - dist)
                        if weight > 0:
                            iH.append(e1)
                            jH.append(e2)
                            sH.append(weight)

        H = sp.csr_matrix((sH, (iH, jH)), shape=(num_el, num_el), dtype=np.float64)
        Hs = np.array(H.sum(axis=1)).flatten()
        return H, Hs

    def node_id(self, i: int, j: int) -> int:
        """Returns node ID from grid coordinate (i in 0..nelx, j in 0..nely)."""
        if not (0 <= i <= self.nelx and 0 <= j <= self.nely):
            raise IndexError(f"Node indices ({i}, {j}) out of range [0..{self.nelx}, 0..{self.nely}]")
        return i * (self.nely + 1) + j

    def clear_boundary_conditions(self):
        """Clears all fixed DOFs and applied forces."""
        self.fixed_dofs.clear()
        self.force_vector.fill(0.0)

    def fix_node(self, i: int, j: int, fix_x: bool = True, fix_y: bool = True):
        """Fix displacements of node (i, j)."""
        nid = self.node_id(i, j)
        if fix_x:
            self.fixed_dofs.add(2 * nid)
        if fix_y:
            self.fixed_dofs.add(2 * nid + 1)

    def fix_wall(self, side: str = "left", fix_x: bool = True, fix_y: bool = True):
        """Fix an entire boundary wall ('left', 'right', 'bottom', 'top')."""
        if side == "left":
            for j in range(self.nely + 1):
                self.fix_node(0, j, fix_x, fix_y)
        elif side == "right":
            for j in range(self.nely + 1):
                self.fix_node(self.nelx, j, fix_x, fix_y)
        elif side == "bottom":
            for i in range(self.nelx + 1):
                self.fix_node(i, 0, fix_x, fix_y)
        elif side == "top":
            for i in range(self.nelx + 1):
                self.fix_node(i, self.nely, fix_x, fix_y)

    def add_load(self, i: int, j: int, fx: float = 0.0, fy: float = 0.0):
        """Apply concentrated point load in Newtons to node (i, j)."""
        nid = self.node_id(i, j)
        self.force_vector[2 * nid] += fx
        self.force_vector[2 * nid + 1] += fy

    def solve(self, max_iter: int = 45, tol: float = 0.01) -> SIMPResult:
        """
        Executes deterministic SIMP topology optimization loop using Optimality Criteria.
        """
        start_time = time.time()

        if len(self.fixed_dofs) < 3:
            raise ValueError(f"Need at least 3 fixed DOFs to prevent rigid body motions. Found {len(self.fixed_dofs)}.")

        if np.all(np.abs(self.force_vector) < 1e-12):
            raise ValueError("No external forces applied to domain.")

        # Initial uniform density matching target volume fraction
        x = np.full(self.num_elements, self.volfrac, dtype=np.float64)
        xPhys = x.copy()

        # Free vs fixed DOFs
        all_dofs = np.arange(self.num_dofs, dtype=np.int32)
        free_mask = np.ones(self.num_dofs, dtype=bool)
        for fdof in self.fixed_dofs:
            free_mask[fdof] = False
        free_dofs = all_dofs[free_mask]

        # Fast sparse matrix row/col indices for K assembly
        # edofMat has shape (num_elements, 8)
        # Each element contributes an 8x8 block
        iK = np.kron(self.edofMat, np.ones((8, 1), dtype=np.int32)).flatten()
        jK = np.kron(self.edofMat, np.ones((1, 8), dtype=np.int32)).flatten()

        c_history = []
        ch_history = []

        compliance = 0.0
        iterations = 0

        for it in range(1, max_iter + 1):
            iterations = it

            # 1. FEA Global Stiffness Matrix Assembly
            # sK = k0.flatten() * (Emin + xPhys^penal * (E0 - Emin))
            E_elements = self.Emin + (xPhys ** self.penal) * (self.E0 - self.Emin)
            sK = (self.k0.flatten()[np.newaxis, :] * E_elements[:, np.newaxis]).flatten()

            K_full = sp.coo_matrix((sK, (iK, jK)), shape=(self.num_dofs, self.num_dofs)).tocsr()
            K_free = K_full[free_dofs, :][:, free_dofs]
            F_free = self.force_vector[free_dofs]

            # 2. Solve displacement field U
            U_free = sla.spsolve(K_free, F_free)
            U = np.zeros(self.num_dofs, dtype=np.float64)
            U[free_dofs] = U_free

            # 3. Compliance & Sensitivity Analysis
            # For each element, U_e = U[edofMat[e, :]]
            # ce = U_e @ k0 @ U_e
            U_e = U[self.edofMat]  # shape (num_elements, 8)
            # Efficient row-wise quadratic form: (U_e @ k0) * U_e summed over cols
            ce = np.sum((U_e @ self.k0) * U_e, axis=1)

            compliance = float(np.sum(E_elements * ce))
            c_history.append(compliance)

            # dc/dx = - penal * (E0 - Emin) * xPhys^(penal - 1) * ce
            dc = - self.penal * (self.E0 - self.Emin) * (xPhys ** (self.penal - 1.0)) * ce

            # 4. Spatial Sensitivity Filtering
            # dc_filtered = H @ (x * dc) / (Hs * max(1e-3, x))
            dc_filtered = np.array(self.H @ (xPhys * dc)).flatten() / (self.Hs * np.maximum(1e-3, xPhys))

            # 5. Optimality Criteria (OC) Density Update with Bisection
            l1 = 1e-9
            l2 = 1e9
            move = 0.2
            xnew = np.zeros_like(x)

            while (l2 - l1) / (l1 + l2) > 1e-4:
                lmid = 0.5 * (l2 + l1)
                # B_e = sqrt(-dc / (lmid * dv)) with dv = 1
                Be = np.sqrt(np.maximum(0.0, -dc_filtered / lmid))
                # Update with move limit
                xnew = np.maximum(
                    0.001,
                    np.maximum(
                        x - move,
                        np.minimum(1.0, np.minimum(x + move, x * Be))
                    )
                )

                if np.mean(xnew) > self.volfrac:
                    l1 = lmid
                else:
                    l2 = lmid

            change = float(np.max(np.abs(xnew - x)))
            ch_history.append(change)
            x = xnew.copy()
            xPhys = x.copy()

            if change < tol:
                break

        # Reshape density to 2D image matrix (nely, nelx)
        # Note: element index e = elx * nely + ely
        # In 2D array, row is y (from top to bottom or bottom to top), col is x
        # Let density_matrix[ely, elx] with ely in 0..nely-1, elx in 0..nelx-1
        density_matrix = np.zeros((self.nely, self.nelx), dtype=np.float64)
        for elx in range(self.nelx):
            for ely in range(self.nely):
                density_matrix[ely, elx] = xPhys[elx * self.nely + ely]

        exec_time = time.time() - start_time
        curr_volfrac = float(np.mean(xPhys))

        return SIMPResult(
            success=True,
            status_message=f"SIMP optimization converged in {iterations} iterations ({exec_time:.2f}s).",
            density_matrix=density_matrix,
            compliance=compliance,
            volume_fraction=curr_volfrac,
            iterations_run=iterations,
            execution_time_sec=exec_time,
            compliance_history=c_history,
            change_history=ch_history,
            nelx=self.nelx,
            nely=self.nely,
            dx=self.dx,
            dy=self.dy,
        )

    def to_plotly_heatmap(
        self,
        res: SIMPResult,
        threshold: float = 0.5,
        colormap: str = "Jet"
    ):
        """
        Creates an interactive Plotly continuous density heatmap with isoline contours,
        support markers, and external load arrows.
        """
        import plotly.graph_objects as go

        x_coords = np.linspace(0, self.nelx * self.dx, self.nelx)
        y_coords = np.linspace(0, self.nely * self.dy, self.nely)

        fig = go.Figure()

        # 1. Continuous Density Heatmap
        fig.add_trace(go.Heatmap(
            z=res.density_matrix,
            x=x_coords,
            y=y_coords,
            colorscale=colormap,
            zmin=0.0,
            zmax=1.0,
            colorbar=dict(
                title=dict(text="Densità Relativa ρ", font=dict(color="#f8fafc", size=11)),
                tickfont=dict(color="#94a3b8", size=10),
                len=0.85
            ),
            hoverongaps=False,
            hovertemplate="X: %{x:.1f} mm<br>Y: %{y:.1f} mm<br>Densità: %{z:.2f}<extra></extra>"
        ))

        # 2. Contour line at threshold (Solid Boundary)
        fig.add_trace(go.Contour(
            z=res.density_matrix,
            x=x_coords,
            y=y_coords,
            contours=dict(
                start=threshold,
                end=threshold,
                size=1.0,
                coloring="none"
            ),
            line=dict(color="#ffffff", width=2.5, dash="solid"),
            showscale=False,
            hoverinfo="none",
            name=f"Confine Solido (ρ = {threshold:.2f})"
        ))

        # 3. Add Boundary Conditions
        fixed_x, fixed_y, fixed_labels = [], [], []
        loaded_x, loaded_y, loaded_labels = [], [], []

        for nid in range(self.num_nodes):
            ix = nid // (self.nely + 1)
            iy = nid % (self.nely + 1)
            x_pos = ix * self.dx
            y_pos = iy * self.dy

            is_fixed_x = (2 * nid in self.fixed_dofs)
            is_fixed_y = (2 * nid + 1 in self.fixed_dofs)
            fx = self.force_vector[2 * nid]
            fy = self.force_vector[2 * nid + 1]

            if is_fixed_x or is_fixed_y:
                fixed_x.append(x_pos)
                fixed_y.append(y_pos)
                fixed_labels.append(f"Incastro ({ix}, {iy}) a ({x_pos:.0f}, {y_pos:.0f}) mm")

            if abs(fx) > 1e-6 or abs(fy) > 1e-6:
                loaded_x.append(x_pos)
                loaded_y.append(y_pos)
                loaded_labels.append(f"Carico ({ix}, {iy}): F=({fx:.0f}, {fy:.0f}) N")

                # Arrow annotation
                mag = math.hypot(fx, fy)
                ax = 0.0 if abs(fx) < 1e-6 else (-35.0 * fx / mag)
                ay = 0.0 if abs(fy) < 1e-6 else (35.0 * fy / mag)

                fig.add_annotation(
                    x=x_pos, y=y_pos,
                    ax=ax, ay=ay,
                    xref="x", yref="y",
                    axref="pixel", ayref="pixel",
                    text=f"F=({fx:.0f}, {fy:.0f})N",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1.5,
                    arrowwidth=2,
                    arrowcolor="#ffffff",
                    font=dict(color="#ffffff", size=10, family="monospace")
                )

        # Support markers
        if fixed_x:
            fig.add_trace(go.Scatter(
                x=fixed_x, y=fixed_y,
                mode="markers",
                marker=dict(size=9, symbol="square", color="#f59e0b", line=dict(color="#78350f", width=1.5)),
                hoverinfo="text",
                hovertext=fixed_labels,
                name="Incastro"
            ))

        # Load markers
        if loaded_x:
            fig.add_trace(go.Scatter(
                x=loaded_x, y=loaded_y,
                mode="markers",
                marker=dict(size=10, symbol="circle", color="#ef4444", line=dict(color="#ffffff", width=1.5)),
                hoverinfo="text",
                hovertext=loaded_labels,
                name="Carico Applicato"
            ))

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0f172a",
            plot_bgcolor="#020617",
            xaxis=dict(
                title="X (mm)",
                scaleanchor="y",
                scaleratio=1,
                gridcolor="#1e293b",
                zerolinecolor="#334155"
            ),
            yaxis=dict(
                title="Y (mm)",
                gridcolor="#1e293b",
                zerolinecolor="#334155"
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(size=10)
            ),
            margin=dict(l=20, r=20, t=30, b=20),
            height=430
        )

        return fig

    def to_3d_params(
        self,
        res: SIMPResult,
        threshold: float = 0.40,
        depth_z_mm: float = 10.0,
        voxel_size_mm: float = 0.6
    ):
        """
        Converts 2D continuous density field into 3D volumetric beams for PicoGK solid synthesis.
        Extracts solid voxel centers, depth extrusions, and orthogonal connectors.
        """
        from PicoGK_Dashboard.core.models import BeamParam, GroundStructureTrussParams

        rho = res.density_matrix
        nely, nelx = rho.shape
        dx, dy = self.dx, self.dy
        r_base = 0.55 * max(dx, dy)
        beams = []

        for ely in range(nely):
            for elx in range(nelx):
                val = rho[ely, elx]
                if val >= threshold:
                    xc = (elx + 0.5) * dx
                    yc = (ely + 0.5) * dy
                    r = r_base * math.sqrt(val)

                    # Extrusion beam in Z
                    beams.append(BeamParam(
                        x1=xc, y1=yc, z1=-depth_z_mm / 2.0,
                        x2=xc, y2=yc, z2=+depth_z_mm / 2.0,
                        radius=r, force=val
                    ))

                    # Horizontal neighbor connection
                    if elx + 1 < nelx and rho[ely, elx + 1] >= threshold:
                        beams.append(BeamParam(
                            x1=xc, y1=yc, z1=0.0,
                            x2=(elx + 1.5) * dx, y2=yc, z2=0.0,
                            radius=r, force=val
                        ))

                    # Vertical neighbor connection
                    if ely + 1 < nely and rho[ely + 1, elx] >= threshold:
                        beams.append(BeamParam(
                            x1=xc, y1=yc, z1=0.0,
                            x2=xc, y2=(ely + 1.5) * dy, z2=0.0,
                            radius=r, force=val
                        ))

        return GroundStructureTrussParams(
            voxel_size_mm=voxel_size_mm,
            depth_z_mm=depth_z_mm,
            beams=beams
        )

