import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Optional

# Ensure PicoGK_Dashboard and project root are in sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
for p in [str(PROJECT_ROOT), str(SCRIPT_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import streamlit as st
import numpy as np

from physics_engine.simp_engine_2d import SIMPOptimizer2D

# -----------------------------------------------------------------------------
# Streamlit Page Setup
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Topology optimization: SIMP and buckling analysis",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Courier+Prime:ital,wght@0,400;0,700;1,400;1,700&display=swap');
    
    * {
        font-family: 'Courier Prime', monospace !important;
    }
    
    /* Hide the top-right passing icons loading widget */
    [data-testid="stStatusWidget"] {
        display: none !important;
    }
    
    /* Hide the sidebar completely */
    [data-testid="stSidebar"] {
        display: none;
    }
    [data-testid="collapsedControl"] {
        display: none;
    }

    .main-header {
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        border-bottom: 1px solid #ddd;
        padding-bottom: 10px;
    }
    .sub-header {
        font-size: 1.0rem;
        color: #666;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Main Header
# -----------------------------------------------------------------------------
st.markdown('<div class="main-header">Topology optimization: SIMP and buckling analysis</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">SIMP continuous domain optimization with linearized buckling analysis</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Parameter Configuration
# -----------------------------------------------------------------------------
col_s1, col_s2, col_s3 = st.columns(3)

with col_s1:
    st.markdown("**1. Domain and material properties**")
    nelx_in = st.number_input("Elements X (nelx)", min_value=10, max_value=240, value=55, step=2)
    nely_in = st.number_input("Elements Y (nely)", min_value=6, max_value=120, value=25, step=2)
    dx_in = st.number_input("Element size dx (mm)", min_value=0.2, max_value=20.0, value=2.0, step=0.5)
    dy_in = st.number_input("Element size dy (mm)", min_value=0.2, max_value=20.0, value=2.0, step=0.5)
    thickness_in = st.number_input("Thickness t (mm)", min_value=0.1, max_value=100.0, value=5.0, step=0.5)
    E0_in = st.number_input("Young's modulus E0 (MPa)", min_value=1.0, max_value=500000.0, value=68900.0, step=1000.0)
    nu_in = st.number_input("Poisson's ratio (ν)", min_value=0.0, max_value=0.49, value=0.33, step=0.01)

with col_s2:
    st.markdown("**2. Optimization parameters**")
    volfrac_in = st.slider("Target volume fraction (Vf)", min_value=0.10, max_value=0.85, value=0.40, step=0.05)
    
    st.markdown("**Pareto trade-off weight (critical)**")
    alpha_in = st.slider("Buckling reinforcement factor (α)", min_value=0.00, max_value=0.90, value=0.40, step=0.05)
    
    rmin_in = st.slider("Filter radius r_min (mm)", min_value=1.5, max_value=15.0, value=5.0, step=0.5)
    rmin_elems = rmin_in / max(dx_in, dy_in)
    if rmin_elems < 2.0:
        st.caption(f"⚠️ r_min = {rmin_elems:.2f} elements: minimal filtering (mesh-scale features possible; use ≈ 2.5–3 elements for mesh independence).")
    else:
        st.caption(f"Filter span: {rmin_elems:.2f} elements")
    penal_in = st.number_input("SIMP penalization (p)", min_value=1.0, max_value=4.0, value=3.0, step=0.5)
    max_iter_in = st.number_input("Max iterations", min_value=10, max_value=300, value=100, step=5)
    
    st.markdown("**Buckling aggregation method**")
    agg_method = st.radio(
        "Aggregation", 
        ["Weighted sum (fundamental mode, nEig=1)", "KS aggregator (multiple modes, nEig>1)"], 
        index=1, 
        label_visibility="collapsed"
    )
    if "KS" in agg_method:
        nEig_in = st.number_input("Eigenvalues (nEig)", min_value=2, max_value=10, value=4, step=1)
    else:
        nEig_in = 1

with col_s3:
    st.markdown("**3. Boundary conditions and loads**")
    bc_type = st.selectbox(
        "Support scheme:",
        options=[
            "Full left wall fixation (x = 0)",
            "Two bottom supports (x=0 and x=L)",
            "Partial left wall fixation",
            "Custom nodes"
        ],
        index=0
    )

    custom_sup_str = ""
    if bc_type == "Custom nodes":
        custom_sup_str = st.text_input(
            f"Support nodes [X_mm, Y_mm] (0..{nelx_in * dx_in}, 0..{nely_in * dy_in})",
            value=f"(0, 0), (0, {nely_in * dy_in})"
        )

    def_load_str = f"({nelx_in * dx_in}, 0, 0, -1000)"
    loads_str = st.text_area(
        f"External forces [X_mm, Y_mm, Fx_N, Fy_N]",
        value=def_load_str,
        height=75
    )

# Parse and validate inputs
import re
sup_nodes = []
if bc_type == "Custom nodes":
    sup_matches = re.findall(r"\(\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\)", custom_sup_str)
    sup_nodes = [(int(float(x) / dx_in), int(float(y) / dy_in)) for x, y in sup_matches]

load_matches = re.findall(r"\(\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)", loads_str)
loads_simp = [(int(float(x) / dx_in), int(float(y) / dy_in), float(fx), float(fy)) for x, y, fx, fy in load_matches]

invalid_loads = [f"({x},{y})" for x, y, _, _ in loads_simp if not (0 <= x <= nelx_in and 0 <= y <= nely_in)]
if invalid_loads:
    st.error(f"Load nodes out of bounds: {', '.join(invalid_loads)}")

num_elements = int(nelx_in) * int(nely_in)
if num_elements > 20000:
    st.warning(f"⚠️ Warning: the domain contains {num_elements} elements. Linearized buckling analysis (LBA) requires solving a generalized eigenvalue problem on large matrices. This analysis will take several minutes, especially when generating the Pareto frontier (which repeats the calculation 11 times).")

# Cache key
simp_key = (
    int(nelx_in), int(nely_in), float(dx_in), float(dy_in),
    float(thickness_in), float(E0_in), float(nu_in), int(max_iter_in),
    float(volfrac_in), float(rmin_in), float(penal_in),
    "buckling_max", float(alpha_in), int(nEig_in),
    bc_type, custom_sup_str, tuple(loads_simp)
)



if st.button("Run topology optimization", type="primary"):
    with st.container():
        try:
            progress_bar = st.progress(0)
            progress_text = st.empty()
            
            def update_progress(it, max_it):
                pct = int((it / max_it) * 100)
                progress_bar.progress(it / max_it)
                progress_text.info(f"🚀 Executing SIMP optimization... Iteration {it}/{max_it} — {pct}% complete...")
            optimizer = SIMPOptimizer2D(
                nelx=int(nelx_in), nely=int(nely_in), dx=float(dx_in), dy=float(dy_in),
                thickness=float(thickness_in), E0=float(E0_in), nu=float(nu_in),
                penal=float(penal_in), rmin=float(rmin_in), volfrac=float(volfrac_in)
            )

            # Apply boundary conditions
            if bc_type == "Full left wall fixation (x = 0)":
                optimizer.fix_wall("left", fix_x=True, fix_y=True)
            elif bc_type == "Two bottom supports (x=0 and x=L)":
                optimizer.fix_node(0, 0, fix_x=True, fix_y=True)
                optimizer.fix_node(int(nelx_in), 0, fix_x=False, fix_y=True)
            elif bc_type == "Partial left wall fixation":
                mid_y = int(nely_in) // 2
                for jy in range(max(0, mid_y - 2), min(int(nely_in) + 1, mid_y + 3)):
                    optimizer.fix_node(0, jy, fix_x=True, fix_y=True)
            elif bc_type == "Custom nodes":
                for sx, sy in sup_nodes:
                    optimizer.fix_node(sx, sy, fix_x=True, fix_y=True)

            for lx, ly, fx, fy in loads_simp:
                optimizer.add_load(lx, ly, fx=fx, fy=fy)

            simp_res = optimizer.solve(max_iter=int(max_iter_in), tol=0.01, mode="buckling_max", alpha_buckling=float(alpha_in), nEig=int(nEig_in), progress_callback=update_progress)
            progress_text.empty()
            progress_bar.empty()

            st.session_state.simp_key = simp_key
            st.session_state.simp_optimizer = optimizer
            st.session_state.simp_result = simp_res
        except Exception as ex:
            st.error(f"Error during SIMP resolution: {str(ex)}")
            st.session_state.simp_result = None

simp_res = st.session_state.get("simp_result", None)
optimizer = st.session_state.get("simp_optimizer", None)


def render_main_visualization(optimizer, simp_res):
    st.markdown("#### Topology metrics")
    st.caption(simp_res.status_message)
    sm1, sm2, sm3, sm4, sm5 = st.columns(5)
    sm1.metric("Compliance (C)", f"{simp_res.compliance:.2f} N·mm")
    if simp_res.blf_history:
        sm2.metric("Buckling (BLF)", f"{simp_res.blf_history[-1]:.3f}")
    else:
        sm2.metric("Buckling (BLF)", "N/A")
    sm3.metric("Volume fraction", f"{simp_res.volume_fraction*100:.1f}%")
    sm4.metric("Iterations", f"{simp_res.iterations_run} / {int(max_iter_in)}")
    sm5.metric("Time", f"{simp_res.execution_time_sec:.1f} s")

    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([1, 1, 1])
    with col_ctrl1:
        view_style_choice = st.selectbox(
            "View style:",
            options=["silhouette_bw", "stress_masked"],
            format_func=lambda x: "Solid silhouette (B&W)" if x == "silhouette_bw" else "Stress heatmap"
        )
    with col_ctrl2:
        dens_threshold = st.slider("Density threshold", min_value=0.10, max_value=0.85, value=0.40, step=0.02)
    with col_ctrl3:
        st.markdown("<br>", unsafe_allow_html=True)
        smooth_viz = st.checkbox("Smooth visualization", value=False)

    fig_plot = optimizer.to_scientific_figures(
        res=simp_res,
        threshold=dens_threshold,
        view_style=view_style_choice,
        colormap="Jet_r",
        smooth=smooth_viz
    )
    st.plotly_chart(fig_plot, use_container_width=True)

if simp_res is not None and simp_res.success and optimizer is not None:
    st.markdown("---")
    render_main_visualization(optimizer, simp_res)

    default_dens_threshold = 0.40

    # -------------------------------------------------------------------------
    # Pareto Sweep Section
    # -------------------------------------------------------------------------
    st.markdown("---")
    
    @st.fragment
    def render_pareto_section():
        if st.button("Generate Pareto alpha sweep"):
            alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            
            st.markdown("#### Pareto frontier sweep")
            st.markdown(
                "This procedure generates the Pareto frontier a posteriori. "
                "The **α** parameter mathematically balances bending stiffness (compliance) and structural stability (buckling).<br>"
                "- **α near 0**: designed exclusively for maximum stiffness (thinner members, higher buckling risk).<br>"
                "- **α near 1**: designed to delay buckling as much as possible (thicker, aligned members, at the expense of global compliance).",
                unsafe_allow_html=True
            )
            
            progress_text = st.empty()
            progress_bar = st.progress(0)
            
            # Create placeholders progressively
            cols = st.columns(3)
            placeholders = [cols[i % 3].empty() for i in range(len(alphas))]
            
            for i, a in enumerate(alphas):
                pct = int(((i+1)/len(alphas))*100)
                progress_text.info(f"⏳ Computing alpha = {a:.1f} ({i+1}/{len(alphas)}) — {pct}%...")
                
                # Create a clone of the optimizer for the run
                opt_clone = SIMPOptimizer2D(
                    nelx=int(nelx_in), nely=int(nely_in), dx=float(dx_in), dy=float(dy_in),
                    thickness=float(thickness_in), E0=float(E0_in), nu=float(nu_in),
                    penal=float(penal_in), rmin=float(rmin_in), volfrac=float(volfrac_in)
                )
                # Re-apply BCs
                if bc_type == "Full left wall fixation (x = 0)":
                    opt_clone.fix_wall("left", fix_x=True, fix_y=True)
                elif bc_type == "Two bottom supports (x=0 and x=L)":
                    opt_clone.fix_node(0, 0, fix_x=True, fix_y=True)
                    opt_clone.fix_node(int(nelx_in), 0, fix_x=False, fix_y=True)
                elif bc_type == "Partial left wall fixation":
                    mid_y = int(nely_in) // 2
                    for jy in range(max(0, mid_y - 2), min(int(nely_in) + 1, mid_y + 3)):
                        opt_clone.fix_node(0, jy, fix_x=True, fix_y=True)
                elif bc_type == "Custom nodes":
                    for sx, sy in sup_nodes:
                        opt_clone.fix_node(sx, sy, fix_x=True, fix_y=True)

                for lx, ly, fx, fy in loads_simp:
                    opt_clone.add_load(lx, ly, fx=fx, fy=fy)

                res_alpha = opt_clone.solve(max_iter=int(max_iter_in), tol=0.01, mode="buckling_max", alpha_buckling=a, nEig=int(nEig_in))
                
                if res_alpha.success:
                    with placeholders[i].container():
                        st.markdown(f"**α = {a:.1f}**")
                        st.caption(f"C: {res_alpha.compliance:.1f} N·mm | BLF: {res_alpha.blf_history[-1] if res_alpha.blf_history else 'N/A':.3f}")
                        f_plot = opt_clone.to_scientific_figures(res=res_alpha, threshold=0.40, view_style="silhouette_bw", smooth=True)
                        f_plot.update_layout(height=250, margin=dict(l=10, r=10, t=10, b=10))
                        st.plotly_chart(f_plot, use_container_width=True)
                
                progress_bar.progress((i + 1) / len(alphas))
                
            progress_text.empty()
            
    render_pareto_section()
