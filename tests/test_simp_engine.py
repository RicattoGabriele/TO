"""
Unit and physical validation tests for 2D SIMP Continuum Topology Optimization Engine.
Tests:
1. Q4 Plane stress element stiffness matrix symmetry and positive semi-definiteness.
2. Deterministic convergence on cantilever benchmark.
3. Plotly heatmap generation.
4. Regression test with frozen 45-iteration settings (E0=68900 MPa, t=5 mm -> C ~ 392.0 N*mm strip load, 418.02 N*mm point load).
5. Exact scale invariance: (E0=1, t=10) vs (E0=68900, t=5) must yield identical C*E0*t, BLF/(E0*t), and density topology.
6. Analytical verification: Full solid Q4 FEM cantilever vs closed-form Timoshenko beam compliance (111.42 N*mm).
7. Honest convergence status reporting and damped beta-continuation schedule.
"""

import os
import sys
import unittest
import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as sla

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from physics_engine.simp_engine_2d import SIMPOptimizer2D, SIMPResult


class TestSIMPEngine2D(unittest.TestCase):
    """Test suite for SIMPOptimizer2D."""

    def test_1_element_stiffness_matrix_properties(self):
        """Verify k0 is symmetric, has 3 zero eigenvalues (rigid body modes in 2D), and is independent of E0."""
        opt1 = SIMPOptimizer2D(nelx=10, nely=5, dx=2.0, dy=2.0, E0=1.0, thickness=5.0)
        opt2 = SIMPOptimizer2D(nelx=10, nely=5, dx=2.0, dy=2.0, E0=68900.0, thickness=5.0)
        k0 = opt1.k0

        self.assertEqual(k0.shape, (8, 8))
        # Symmetry check
        np.testing.assert_allclose(k0, k0.T, atol=1e-12, err_msg="k0 must be strictly symmetric.")
        # k0 must be computed at unit Young's modulus (no double-E0 multiplication)
        np.testing.assert_allclose(opt1.k0, opt2.k0, rtol=1e-14, err_msg="k0 must be for unit E=1.0.")
        np.testing.assert_allclose(opt1.Cmat0, opt2.Cmat0, rtol=1e-14, err_msg="Cmat0 must be for unit E=1.0.")

        # Eigenvalues check: exactly 3 zero eigenvalues for unconstrained 2D Q4 element
        eigvals = np.linalg.eigvalsh(k0)
        zero_eigs = np.sum(np.abs(eigvals) < 1e-10)
        self.assertEqual(zero_eigs, 3, f"Expected 3 rigid body modes, found {zero_eigs}.")
        self.assertTrue(np.all(eigvals[3:] > 0), "Remaining 5 elastic modes must be strictly positive.")

    def test_2_cantilever_simp_optimization_convergence(self):
        """Run cantilever benchmark (40x20 elements) and verify volume and compliance convergence."""
        target_vol = 0.4
        opt = SIMPOptimizer2D(
            nelx=40,
            nely=20,
            dx=2.5,
            dy=2.5,
            rmin=2.5,
            volfrac=target_vol,
            penal=3.0,
        )

        # Fix left wall (x=0)
        opt.fix_wall("left", fix_x=True, fix_y=True)

        # Apply downward point load at bottom right (x=40, y=0)
        opt.add_load(40, 0, fx=0.0, fy=-100.0)

        res = opt.solve(max_iter=25, tol=0.01)

        self.assertTrue(res.success)
        self.assertEqual(res.density_matrix.shape, (20, 40))
        self.assertAlmostEqual(res.volume_fraction, target_vol, places=2)

        # Density bounds
        self.assertGreaterEqual(np.min(res.density_matrix), 0.0)
        self.assertLessEqual(np.max(res.density_matrix), 1.0)

        # Compliance must decrease as SIMP redistributes material
        self.assertGreater(res.compliance_history[0], res.compliance_history[-1])

    def test_3_plotly_heatmap_generation(self):
        """Verify Plotly interactive heatmap generation works without errors."""
        opt = SIMPOptimizer2D(nelx=20, nely=10, dx=1.0, dy=1.0, volfrac=0.5)
        opt.fix_wall("left")
        opt.add_load(20, 5, fx=0.0, fy=-50.0)
        res = opt.solve(max_iter=5, tol=0.05)

        fig = opt.to_plotly_heatmap(res, threshold=0.45)
        self.assertIsNotNone(fig)
        self.assertGreater(len(fig.data), 0)

    def test_4_regression_frozen_45_iters(self):
        """
        Regression test (Test 1):
        With frozen legacy settings (45 iterations, legacy beta schedule, undamped OC),
        E0 = 68900 MPa and t = 5 mm on the 200x100 cantilever (400x200 mm, Vf=0.25, rmin=2.5 mm):
        - 30 mm strip load (16 nodes at x=200, y=0..15, nu=0.33) must yield C ~ 392.0 N*mm
          (and point load at (200, 0) with nu=0.30 must yield C = 418.02 N*mm = 14400598 / 34450).
        """
        # Strip load over 30 mm (nodes j=0..15 at x=200), nu=0.33
        opt_strip = SIMPOptimizer2D(
            nelx=200, nely=100, dx=2.0, dy=2.0,
            thickness=5.0, E0=68900.0, nu=0.33,
            penal=3.0, rmin=2.5, volfrac=0.25
        )
        opt_strip.fix_wall("left", fix_x=True, fix_y=True)
        for j in range(16):
            opt_strip.add_load(200, j, fx=0.0, fy=-1000.0 / 16.0)

        res_strip = opt_strip.solve(max_iter=45, tol=0.01, mode="compliance", legacy_schedule=True)
        assert res_strip.compliance == pytest.approx(392.0, rel=0.005), (
            f"Expected strip-load compliance ~392.0 N*mm, got {res_strip.compliance:.2f} N*mm"
        )

        # Single point load at bottom-right corner (200, 0), nu=0.30
        opt_pt = SIMPOptimizer2D(
            nelx=200, nely=100, dx=2.0, dy=2.0,
            thickness=5.0, E0=68900.0, nu=0.30,
            penal=3.0, rmin=2.5, volfrac=0.25
        )
        opt_pt.fix_wall("left", fix_x=True, fix_y=True)
        opt_pt.add_load(200, 0, fx=0.0, fy=-1000.0)

        res_pt = opt_pt.solve(max_iter=45, tol=0.01, mode="compliance", legacy_schedule=True)
        assert res_pt.compliance == pytest.approx(14400598.0 / 34450.0, abs=0.05), (
            f"Expected point-load compliance 418.01 N*mm, got {res_pt.compliance:.2f} N*mm"
        )

    def test_5_scale_invariance_E0_and_thickness(self):
        """
        Scale invariance test (Test 2):
        Comparing (E0 = 1.0, t = 10.0) against (E0 = 68900.0, t = 5.0) must produce:
        - Identical invariant product C * E0 * t
        - Identical invariant ratio BLF / (E0 * t)
        - Identical density topology
        This test directly catches the double-E0 multiplication bug in k0 and Cmat0.
        """
        def run_case(E0_val: float, t_val: float):
            opt = SIMPOptimizer2D(
                nelx=30, nely=15, dx=2.0, dy=2.0,
                thickness=t_val, E0=E0_val, nu=0.33,
                penal=3.0, rmin=3.0, volfrac=0.35
            )
            opt.fix_wall("left", fix_x=True, fix_y=True)
            opt.add_load(30, 0, fx=0.0, fy=-1000.0)
            res = opt.solve(
                max_iter=15, tol=1e-4, mode="buckling_max",
                alpha_buckling=0.40, nEig=2, legacy_schedule=True
            )
            stresses = opt.compute_principal_stresses(res)
            return opt, res, stresses

        opt1, res1, str1 = run_case(E0_val=1.0, t_val=10.0)
        opt2, res2, str2 = run_case(E0_val=68900.0, t_val=5.0)

        inv_C_1 = res1.compliance * 1.0 * 10.0
        inv_C_2 = res2.compliance * 68900.0 * 5.0
        assert inv_C_1 == pytest.approx(inv_C_2, rel=1e-6), (
            f"Scale invariance violated for C*E0*t: {inv_C_1} vs {inv_C_2}"
        )

        inv_blf_1 = res1.blf_history[-1] / (1.0 * 10.0)
        inv_blf_2 = res2.blf_history[-1] / (68900.0 * 5.0)
        assert inv_blf_1 == pytest.approx(inv_blf_2, rel=1e-5), (
            f"Scale invariance violated for BLF/(E0*t): {inv_blf_1} vs {inv_blf_2}"
        )

        np.testing.assert_allclose(
            res1.density_matrix, res2.density_matrix, atol=1e-5,
            err_msg="Topology density_matrix must be invariant to E0 and t."
        )

        # Principal stresses sigma * t must also be independent of E0
        np.testing.assert_allclose(
            str1["von_mises"] * 10.0, str2["von_mises"] * 5.0, rtol=1e-5,
            err_msg="Physical stress * thickness must be independent of E0."
        )

    def test_6_full_solid_vs_timoshenko(self):
        """
        Analytical verification test (Test 3):
        Full solid cantilever (L=400 mm, H=200 mm, t=5 mm, E0=68900 MPa, nu=0.33, F=-1000 N
        distributed over the right tip edge) against closed-form Timoshenko beam theory:
        C_Timoshenko = F^2 * L^3 / (3 * E0 * I) + 6 * F^2 * L / (5 * G * A) = 111.42 N*mm.
        """
        L, H, t = 400.0, 200.0, 5.0
        E0, nu = 68900.0, 0.33
        F_total = 1000.0
        nelx, nely = 200, 100
        dx, dy = L / nelx, H / nely

        opt = SIMPOptimizer2D(
            nelx=nelx, nely=nely, dx=dx, dy=dy,
            thickness=t, E0=E0, nu=nu, volfrac=1.0
        )
        opt.fix_wall("left", fix_x=True, fix_y=True)

        # Parabolic/trapezoidal shear traction along right edge x = nelx
        for j in range(nely + 1):
            w = 0.5 if (j == 0 or j == nely) else 1.0
            opt.add_load(nelx, j, fx=0.0, fy=-F_total * w / nely)

        # Assemble full solid stiffness matrix (xPhys = 1.0 everywhere)
        E_elements = np.full(opt.num_elements, E0, dtype=np.float64)
        sK = (opt.k0.flatten()[np.newaxis, :] * E_elements[:, np.newaxis]).flatten()
        iK = np.kron(opt.edofMat, np.ones((8, 1), dtype=np.int32)).flatten()
        jK = np.kron(opt.edofMat, np.ones((1, 8), dtype=np.int32)).flatten()

        K_full = sp.coo_matrix((sK, (iK, jK)), shape=(opt.num_dofs, opt.num_dofs)).tocsr()
        all_dofs = np.arange(opt.num_dofs, dtype=np.int32)
        free_mask = np.ones(opt.num_dofs, dtype=bool)
        for fdof in opt.fixed_dofs:
            free_mask[fdof] = False
        free_dofs = all_dofs[free_mask]

        U = np.zeros(opt.num_dofs, dtype=np.float64)
        U[free_dofs] = sla.spsolve(K_full[free_dofs, :][:, free_dofs].tocsc(), opt.force_vector[free_dofs])
        c_fem = float(opt.force_vector @ U)

        # Analytical Timoshenko compliance
        I_beam = t * (H ** 3) / 12.0
        A_beam = t * H
        G_mod = E0 / (2.0 * (1.0 + nu))
        kappa = 5.0 / 6.0
        c_bending = (F_total ** 2) * (L ** 3) / (3.0 * E0 * I_beam)
        c_shear = (F_total ** 2) * L / (kappa * G_mod * A_beam)
        c_timoshenko = c_bending + c_shear

        assert c_timoshenko == pytest.approx(111.42, abs=0.05)
        assert c_fem == pytest.approx(c_timoshenko, rel=0.015), (
            f"FEM full-solid compliance {c_fem:.2f} N*mm deviates from Timoshenko {c_timoshenko:.2f} N*mm by >1.5%"
        )

    def test_7_honest_convergence_message_and_damped_schedule(self):
        """
        Verify that status_message reports 'converged after N iterations' ONLY when change < tol
        actually triggers, and reports 'stopped at max_iter' otherwise.
        """
        opt = SIMPOptimizer2D(nelx=30, nely=15, dx=2.0, dy=2.0, thickness=5.0, E0=68900.0, nu=0.33, rmin=4.0, volfrac=0.35)
        opt.fix_wall("left")
        opt.add_load(30, 0, fx=0.0, fy=-1000.0)

        # 1. Force early stop before convergence
        res_unconverged = opt.solve(max_iter=5, tol=1e-6, mode="compliance")
        self.assertIn("stopped after 5 iterations", res_unconverged.status_message)
        self.assertIn("max_iter=5", res_unconverged.status_message)
        self.assertNotIn("converged", res_unconverged.status_message)
        self.assertFalse(res_unconverged.converged)
        self.assertFalse(res_unconverged.objective_stable_20)

        # 2. Allow convergence with damped schedule and looser tolerance
        res_converged = opt.solve(max_iter=100, tol=0.02, mode="compliance")
        self.assertLess(res_converged.change_history[-1], 0.02)
        self.assertIn(f"converged after {res_converged.iterations_run} iterations", res_converged.status_message)
        self.assertTrue(res_converged.converged)

    def test_8_compliance_stopping_criterion_at_max_beta(self):
        """
        Verify the new stopping criterion: relative compliance change over 10 iterations
        evaluated only at maximum beta (beta >= beta_max).
        Also verify objective stability reporting over the last 20 iterations.
        """
        # Benchmark 55x25 cantilever where border elements oscillate preventing change < 0.01,
        # but objective stabilizes (< 0.1% change over 10 iterations at beta=8.0).
        opt = SIMPOptimizer2D(nelx=55, nely=25, dx=1.0, dy=1.0, volfrac=0.4)
        opt.fix_wall("left")
        opt.add_load(55, 0, fx=0.0, fy=-100.0)

        res = opt.solve(max_iter=100, tol=0.01, tol_c=0.001)

        # Must converge early before max_iter=100 because of stable compliance at max beta
        self.assertTrue(res.converged)
        self.assertLess(res.iterations_run, 100)
        self.assertGreaterEqual(res.iterations_run, 61)  # beta reaches 8.0 at it=61
        self.assertIn(f"converged after {res.iterations_run} iterations", res.status_message)
        self.assertIn("objective stable: ΔC_10=", res.status_message)
        self.assertIn("< 0.1%", res.status_message)

        # Verify relative delta C over 10 iterations is indeed < 0.1%
        rel_dC_10 = abs(res.compliance_history[-1] - res.compliance_history[-11]) / res.compliance_history[-1]
        self.assertLess(rel_dC_10, 0.001)

        # Verify exhausting max_iter without triggering convergence:
        # Run with very tight tolerances so neither change nor tol_c triggers
        res_stopped = opt.solve(max_iter=100, tol=1e-8, tol_c=1e-8)
        self.assertFalse(res_stopped.converged)
        self.assertEqual(res_stopped.iterations_run, 100)
        self.assertIn("stopped after 100 iterations", res_stopped.status_message)
        self.assertIn("max_iter=100", res_stopped.status_message)
        self.assertNotIn("converged", res_stopped.status_message)

        # Over the last 20 iterations (it=81..100), compliance variation is 0.047% < 1‰
        self.assertTrue(res_stopped.objective_stable_20)
        self.assertIn("objective stable (ΔC < 0.1% over the last 20 iterations)", res_stopped.status_message)

    def test_9_stopping_criterion_edge_cases(self):
        """
        Verify edge cases:
        1. tol_c=None runs cleanly without raising TypeError.
        2. Short run max_iter=10 cleanly reports 'stopped after 10 iterations' with objective_stable_20=False.
        """
        opt = SIMPOptimizer2D(nelx=20, nely=10, dx=1.0, dy=1.0, volfrac=0.4)
        opt.fix_wall("left")
        opt.add_load(20, 5, fx=0.0, fy=-50.0)

        # 1. tol_c=None
        res_none = opt.solve(max_iter=5, tol=1e-8, tol_c=None)
        self.assertFalse(res_none.converged)
        self.assertIn("stopped after 5 iterations", res_none.status_message)

        # 2. Short run max_iter=10
        res_10 = opt.solve(max_iter=10, tol=1e-8, tol_c=0.001)
        self.assertFalse(res_10.converged)
        self.assertFalse(res_10.objective_stable_20)
        self.assertEqual(res_10.iterations_run, 10)
        self.assertIn("stopped after 10 iterations", res_10.status_message)
        self.assertNotIn("converged", res_10.status_message)
        self.assertNotIn("objective stable", res_10.status_message)


if __name__ == "__main__":
    unittest.main()
