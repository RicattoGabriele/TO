"""
Unit test suite for Ground Structure Generative Physics Engine.
Verifies the exact benchmark from the hand-drawn sketch:
- Grid 11x5 nodes (x=0..10, y=0..4)
- Fixed support on (0,2), (0,3), (0,4)
- Point load on (10,0) with Fy = -1000 N
- Strict physical equilibrium, stress criteria (sigma = F/A), and topology checks.
"""

import math
import os
import sys
import unittest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from physics_engine.ground_structure import GroundStructure2D, GroundStructureResult


class TestCaseStudyGroundStructure(unittest.TestCase):
    """Test suite for the exact 11x5 hand-drawn cantilever case study."""

    def setUp(self):
        # 11 nodes along x (0..10), 5 nodes along y (0..4), dx=dy=10 mm
        self.dx = 10.0
        self.dy = 10.0
        self.gs = GroundStructure2D(nx=11, ny=5, dx=self.dx, dy=self.dy)
        self.gs.generate_candidate_members()

        # Fixed supports: (0,2), (0,3), (0,4)
        self.gs.set_support(0, 2, fix_x=True, fix_y=True)
        self.gs.set_support(0, 3, fix_x=True, fix_y=True)
        self.gs.set_support(0, 4, fix_x=True, fix_y=True)

        # Applied load at (10, 0): Fy = -1000 N
        self.load_f = 1000.0
        self.gs.add_load(10, 0, fx=0.0, fy=-self.load_f)

        # Material: Aluminium 6061-T6
        self.sigma_allowable = 240.0  # MPa
        self.density = 2.70  # g/cm^3

    def test_1_solver_convergence_and_member_generation(self):
        """Verify candidate members are generated correctly and LP converges."""
        self.assertGreater(len(self.gs.members), 100, "Should generate candidate ground structure members.")

        res = self.gs.solve(
            sigma_t=self.sigma_allowable,
            sigma_c=self.sigma_allowable,
            density_g_cm3=self.density,
        )

        self.assertTrue(res.success, f"Ground structure solver failed: {res.status_message}")
        self.assertGreater(len(res.active_members), 0, "Active members must be non-empty.")
        self.assertGreater(res.total_mass_kg, 0.0, "Total mass must be positive.")
        self.assertGreater(res.total_volume_mm3, 0.0, "Total volume must be positive.")

    def test_2_nodal_and_global_equilibrium_laws(self):
        """Verify exact equilibrium at every single node and overall structure."""
        res = self.gs.solve(
            sigma_t=self.sigma_allowable,
            sigma_c=self.sigma_allowable,
            density_g_cm3=self.density,
        )

        # 1. Nodal residual must be virtually zero (LP exactness)
        self.assertLess(
            res.max_nodal_residual_N,
            1e-5,
            f"Nodal equilibrium residual too large: {res.max_nodal_residual_N} N",
        )

        # 2. Global force balances: sum(Fx) = 0, sum(Fy) = 0
        self.assertLess(
            res.global_fx_residual,
            1e-5,
            f"Global Fx residual non-zero: {res.global_fx_residual} N",
        )
        self.assertLess(
            res.global_fy_residual,
            1e-5,
            f"Global Fy residual non-zero: {res.global_fy_residual} N",
        )

        # 3. Global moment balance: sum(M) = 0
        self.assertLess(
            res.global_moment_residual_Nmm,
            1e-3,
            f"Global moment residual non-zero: {res.global_moment_residual_Nmm} N*mm",
        )

        # 4. Vertical reactions must sum to +1000 N
        total_ry = sum(ry for (rx, ry) in res.reactions.values())
        self.assertAlmostEqual(total_ry, self.load_f, places=3)

    def test_3_constitutive_and_sizing_law(self):
        """Verify sigma = |F| / A on every active member."""
        res = self.gs.solve(
            sigma_t=self.sigma_allowable,
            sigma_c=self.sigma_allowable,
            density_g_cm3=self.density,
        )

        for m in res.active_members:
            # Area must equal |force| / sigma_max
            expected_area = abs(m.force) / self.sigma_allowable
            self.assertAlmostEqual(m.area, expected_area, places=5)

            # Radius must be sqrt(area / pi)
            expected_radius = math.sqrt(expected_area / math.pi)
            self.assertAlmostEqual(m.radius, expected_radius, places=5)

            # Stress must never exceed allowable
            self.assertLessEqual(
                m.stress,
                self.sigma_allowable + 1e-4,
                f"Member {m.id} exceeds allowable stress: {m.stress} > {self.sigma_allowable}",
            )

    def test_4_physical_load_path_and_inactive_domain(self):
        """
        Verify the physical intuition of the problem:
        - Inactive nodes: (0,0) and (0,1) must have ZERO material (not part of load path).
        - Load node (10,0) must be active.
        - Support nodes: top nodes in tension, bottom in compression.
        """
        res = self.gs.solve(
            sigma_t=self.sigma_allowable,
            sigma_c=self.sigma_allowable,
            density_g_cm3=self.density,
        )

        id_0_0 = self.gs.get_node_id(0, 0)
        id_0_1 = self.gs.get_node_id(0, 1)
        id_10_0 = self.gs.get_node_id(10, 0)

        # Node (10,0) MUST be active (load application point)
        self.assertIn(id_10_0, res.active_node_ids)

        # Nodes (0,0) and (0,1) MUST NOT be active
        self.assertNotIn(
            id_0_0,
            res.active_node_ids,
            "Node (0,0) has no reason to carry load and should have 0 material.",
        )
        self.assertNotIn(
            id_0_1,
            res.active_node_ids,
            "Node (0,1) has no reason to carry load and should have 0 material.",
        )

        # Moment arm check:
        # Load at x=100mm, y=0mm with Fy=-1000N => Clockwise moment of 1000 * 100 = 100000 N*mm
        # Top support (0, 4) at y=40mm must have Rx < 0 (pulling wall, tension reaction)
        # Bottom support (0, 2) at y=20mm must have Rx > 0 (pushing wall, compression reaction)
        id_0_4 = self.gs.get_node_id(0, 4)
        id_0_2 = self.gs.get_node_id(0, 2)

        rx_top = res.reactions[id_0_4][0]
        rx_bottom = res.reactions[id_0_2][0]

        self.assertLess(rx_top, 0.0, "Top support should be under horizontal tension (Rx < 0).")
        self.assertGreater(rx_bottom, 0.0, "Bottom support should be under horizontal compression (Rx > 0).")

        print("\n" + "=" * 70)
        print("CASE STUDY VERIFICATION REPORT (11x5 Hand-Drawn Sketch)")
        print("=" * 70)
        print(f"Total candidate members evaluated: {len(res.all_members)}")
        print(f"Active members carrying load:       {len(res.active_members)}")
        print(f"Active nodes participating:         {len(res.active_node_ids)} / {len(res.nodes)}")
        print(f"Total mass (Al 6061-T6):            {res.total_mass_kg * 1000:.2f} g")
        print(f"Total solid volume:                 {res.total_volume_mm3:.1f} mm³")
        print(f"Max member axial force:             {res.max_axial_force_N:.2f} N")
        print(f"Max member stress:                  {res.max_stress_MPa:.2f} MPa (Limit: {self.sigma_allowable} MPa)")
        print(f"Max nodal equilibrium residual:     {res.max_nodal_residual_N:.2e} N")
        print(f"Global Fx residual:                 {res.global_fx_residual:.2e} N")
        print(f"Global Fy residual:                 {res.global_fy_residual:.2e} N")
        print(f"Global Moment residual:             {res.global_moment_residual_Nmm:.2e} N*mm")
        print("Support Reactions:")
        for nid in [self.gs.get_node_id(0, 4), self.gs.get_node_id(0, 3), self.gs.get_node_id(0, 2)]:
            node = self.gs.nodes[nid]
            rx, ry = res.reactions[nid]
            print(f"  Node ({node.ix}, {node.iy}) at ({node.x:.0f}, {node.y:.0f}) mm -> Rx = {rx:9.2f} N, Ry = {ry:9.2f} N")
        print("=" * 70)


if __name__ == "__main__":
    unittest.main()
