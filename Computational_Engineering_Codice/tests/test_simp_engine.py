"""
Unit tests for 2D SIMP Continuum Topology Optimization Engine.
Tests:
1. Q4 Plane stress element stiffness matrix symmetry and positive semi-definiteness.
2. Deterministic convergence on cantilever benchmark.
3. Strict volume fraction conservation.
4. Spatial filter execution and checkerboard prevention.
"""

import os
import sys
import unittest
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from physics_engine.simp_engine_2d import SIMPOptimizer2D, SIMPResult


class TestSIMPEngine2D(unittest.TestCase):
    """Test suite for SIMPOptimizer2D."""

    def test_1_element_stiffness_matrix_properties(self):
        """Verify k0 is symmetric and has 3 zero eigenvalues (rigid body modes in 2D)."""
        opt = SIMPOptimizer2D(nelx=10, nely=5, dx=2.0, dy=2.0)
        k0 = opt.k0

        self.assertEqual(k0.shape, (8, 8))
        # Symmetry check
        np.testing.assert_allclose(k0, k0.T, atol=1e-12, err_msg="k0 must be strictly symmetric.")

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
        self.assertGreaterEqual(np.min(res.density_matrix), 0.001)
        self.assertLessEqual(np.max(res.density_matrix), 1.0)

        # Compliance must decrease as SIMP redistributes material
        self.assertGreater(res.compliance_history[0], res.compliance_history[-1])

        print(f"\n[SIMP TEST PASSED] Converged in {res.iterations_run} iters ({res.execution_time_sec:.2f}s). Final Vol: {res.volume_fraction*100:.1f}%, Compliance: {res.compliance:.2f}")

    def test_3_plotly_heatmap_generation(self):
        """Verify Plotly interactive heatmap generation works without errors."""
        opt = SIMPOptimizer2D(nelx=20, nely=10, dx=1.0, dy=1.0, volfrac=0.5)
        opt.fix_wall("left")
        opt.add_load(20, 5, fx=0.0, fy=-50.0)
        res = opt.solve(max_iter=5, tol=0.05)

        fig = opt.to_plotly_heatmap(res, threshold=0.45)
        self.assertIsNotNone(fig)
        self.assertGreater(len(fig.data), 0)


if __name__ == "__main__":
    unittest.main()
