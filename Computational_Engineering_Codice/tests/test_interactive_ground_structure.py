"""
Integration test for the interactive Generative Ground Structure & PicoGK pipeline.
Verifies:
1. Dynamic support and load boundary conditions (cantilever vs two-support bridge).
2. Serialization to GroundStructureTrussParams model.
3. Standalone C# source synthesis.
4. Real-time 3D solid mesh generation with PicoGK .NET engine.
5. Valid manifold mesh outputs (.stl and .obj).
"""

import os
import sys
import tempfile
import unittest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from physics_engine.ground_structure import GroundStructure2D
from PicoGK_Dashboard.core.models import ComponentType, GroundStructureTrussParams
from PicoGK_Dashboard.core.template_engine import generate_csharp_source
from PicoGK_Dashboard.core.execution_runner import run_generation
from PicoGK_Dashboard.tests.mesh_test_utils import validate_obj_file, validate_stl_file


class TestInteractiveGroundStructurePicoGK(unittest.TestCase):
    """End-to-end test for dynamic Ground Structure solver and 3D PicoGK synthesis."""

    def setUp(self):
        self.temp_dir_obj = tempfile.TemporaryDirectory()
        self.temp_dir = self.temp_dir_obj.name

    def tearDown(self):
        try:
            self.temp_dir_obj.cleanup()
        except Exception:
            pass

    def test_1_dynamic_two_support_bridge_case(self):
        """Test a bridge topology: 13x5 grid with supports at both lower corners and central vertical load."""
        gs = GroundStructure2D(nx=13, ny=5, dx=10.0, dy=10.0)
        gs.generate_candidate_members()

        # Left bottom pin, right bottom pin
        gs.set_support(0, 0, fix_x=True, fix_y=True)
        gs.set_support(12, 0, fix_x=True, fix_y=True)

        # Central vertical load downward
        gs.add_load(6, 0, fx=0.0, fy=-2000.0)

        res = gs.solve(sigma_t=240.0, sigma_c=240.0, density_g_cm3=2.70)
        self.assertTrue(res.success)
        self.assertGreater(len(res.active_members), 0)

        # Vertical equilibrium: each pin should carry exactly 1000 N
        r0 = res.reactions[gs.get_node_id(0, 0)]
        r12 = res.reactions[gs.get_node_id(12, 0)]
        self.assertAlmostEqual(r0[1], 1000.0, places=1)
        self.assertAlmostEqual(r12[1], 1000.0, places=1)

        # Residual equilibrium
        self.assertLess(res.max_nodal_residual_N, 1e-5)
        self.assertLess(res.global_fy_residual, 1e-5)

    def test_2_end_to_end_picogk_3d_mesh_generation(self):
        """Solve 11x5 cantilever, serialize to GroundStructureTrussParams, and generate 3D STL via PicoGK."""
        gs = GroundStructure2D(nx=11, ny=5, dx=10.0, dy=10.0)
        gs.generate_candidate_members()
        gs.set_supports_list([(0, 2), (0, 3), (0, 4)])
        gs.add_load(10, 0, fx=0.0, fy=-1000.0)

        res = gs.solve(sigma_t=240.0, sigma_c=240.0, density_g_cm3=2.70)
        self.assertTrue(res.success)

        # Convert to Pydantic model with arbitrary 3D depth (e.g. 10mm)
        truss_params = gs.to_truss_params(res, voxel_size_mm=0.6, depth_z_mm=12.0)
        self.assertEqual(truss_params.component_type, ComponentType.GROUND_STRUCTURE)
        self.assertEqual(len(truss_params.beams), len(res.active_members))

        # Check standalone C# source code synthesis
        cs_file = os.path.join(self.temp_dir, "GeneratedTruss.cs")
        csharp_code = generate_csharp_source(truss_params, output_path=cs_file)
        self.assertTrue(os.path.exists(cs_file))
        self.assertIn("Lattice(lib)", csharp_code)
        self.assertIn("AddBeam", csharp_code)

        # Run .NET execution
        gen_res = run_generation(truss_params, output_dir=self.temp_dir, timeout_sec=60)
        self.assertTrue(gen_res.success, f"PicoGK generation failed: {gen_res.error_message}\nStderr: {gen_res.stderr}")
        self.assertIsNotNone(gen_res.stl_path)
        self.assertTrue(os.path.exists(gen_res.stl_path))
        self.assertIsNotNone(gen_res.obj_path)
        self.assertTrue(os.path.exists(gen_res.obj_path))

        self.assertGreater(gen_res.triangle_count, 1000)
        self.assertGreater(gen_res.vertex_count, 500)

        # Byte-level format verification
        is_stl_valid, stl_reason, stl_triangles = validate_stl_file(gen_res.stl_path)
        self.assertTrue(is_stl_valid, f"Invalid STL: {stl_reason}")
        self.assertGreater(stl_triangles, 0)

        is_obj_valid, obj_reason, obj_v, obj_f = validate_obj_file(gen_res.obj_path)
        self.assertTrue(is_obj_valid, f"Invalid OBJ: {obj_reason}")
        self.assertGreater(obj_v, 0)
        self.assertGreater(obj_f, 0)

        print(f"\n[E2E TEST PASSED] 3D Solid Mesh: {gen_res.triangle_count:,} triangles, {gen_res.vertex_count:,} vertices in {gen_res.execution_time_sec:.2f}s")


if __name__ == "__main__":
    unittest.main()
