"""
Top-Level FEA Driver for 40 MWth Marine MSR
=============================================

Executes the complete FEA analysis pipeline:
    1. Build mesh for specified geometry level
    2. Run coupled multiphysics analysis
    3. Generate visualization figures
    4. Print summary results
    5. Save results to output directory

Usage:
    # From project root:
    python -m fea.run_fea --level 1 --physics all

    # Or programmatically:
    from fea.run_fea import run_fea_analysis
    results = run_fea_analysis(level=1, output_dir='results/fea')
"""

import numpy as np
import time
import os
import sys
import json
import argparse

# Ensure project root is on path
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


def run_fea_analysis(level=1, physics=None, total_power=40e6,
                     output_dir='results/fea'):
    """Run complete FEA analysis.

    Parameters
    ----------
    level : int
        Geometry level: 1 (R-Z full core), 2 (hex cell), 3 (rosette).
    physics : list of str or None
        Physics modules to run: ['neutronics', 'thermal', 'structural'].
        Default None -> all physics for the level.
    total_power : float
        Reactor thermal power [W]. Default 40 MW.
    output_dir : str
        Output directory for results and figures.

    Returns
    -------
    results : dict
        Dictionary with all results, compatible with main.py format.
        Keys: 'level', 'mesh_info', 'neutronics', 'thermal',
              'structural', 'timing'.
    """
    from fea.solvers.coupling import CoupledAnalysis
    from fea.postprocessing.visualization import generate_fea_figures

    os.makedirs(output_dir, exist_ok=True)

    if physics is None:
        if level == 1:
            physics = ['neutronics', 'thermal', 'structural']
        else:
            physics = ['thermal', 'structural']

    print("=" * 70)
    print(f"40 MWth Marine MSR - FEA Analysis (Level {level})")
    print("=" * 70)
    print(f"Physics: {', '.join(physics)}")
    print(f"Total power: {total_power/1e6:.1f} MW")
    print(f"Output: {output_dir}/")
    print()

    # --- Run coupled analysis ---
    t_start = time.time()

    coupling = CoupledAnalysis(level=level)
    coupled_result = coupling.run(total_power=total_power)

    t_solve = time.time() - t_start
    print(f"\n[Timing] Total solve time: {t_solve:.1f} s")

    # --- Generate figures ---
    print("\n[Post-processing] Generating figures...")
    t_viz_start = time.time()
    generate_fea_figures(coupled_result, output_dir=output_dir, level=level)
    t_viz = time.time() - t_viz_start
    print(f"[Timing] Visualization time: {t_viz:.1f} s")

    # --- Print summary ---
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    mesh = coupled_result.mesh
    print(f"\nMesh:")
    print(f"  Nodes:    {mesh.n_nodes}")
    print(f"  Elements: {mesh.n_elements}")
    print(f"  Type:     {mesh.element_type}")
    print(f"  Coord:    {mesh.coord_system}")

    results = {
        'level': level,
        'mesh_info': {
            'n_nodes': mesh.n_nodes,
            'n_elements': mesh.n_elements,
            'element_type': mesh.element_type,
            'coord_system': mesh.coord_system,
        },
        'timing': {
            'solve_seconds': t_solve,
            'viz_seconds': t_viz,
        },
    }

    if coupled_result.neutronics is not None:
        neut = coupled_result.neutronics
        print(f"\nNeutronics:")
        print(f"  keff:             {neut.keff:.5f}")
        print(f"  Converged:        {neut.converged}")
        print(f"  Iterations:       {neut.iterations}")
        print(f"  Peak flux:        {np.max(neut.flux):.4f} (normalized)")
        print(f"  Peak power den.:  {np.max(neut.power_density):.3e} W/m3")
        print(f"  Avg power den.:   {np.mean(neut.power_density[neut.power_density > 0]):.3e} W/m3")

        results['neutronics'] = {
            'keff': float(neut.keff),
            'converged': neut.converged,
            'iterations': neut.iterations,
            'peak_flux': float(np.max(neut.flux)),
            'peak_power_density_W_m3': float(np.max(neut.power_density)),
        }

    thermal = coupled_result.thermal
    print(f"\nThermal:")
    print(f"  T_min:         {np.min(thermal.temperature):.1f} K "
          f"({np.min(thermal.temperature) - 273.15:.1f} C)")
    print(f"  T_max:         {np.max(thermal.temperature):.1f} K "
          f"({np.max(thermal.temperature) - 273.15:.1f} C)")
    print(f"  T_avg (core):  {_avg_core_temp(mesh, thermal.temperature):.1f} K")
    print(f"  Converged:     {thermal.converged}")
    print(f"  Iterations:    {thermal.iterations}")
    print(f"  Max heat flux: {np.max(np.linalg.norm(thermal.heat_flux, axis=1)):.3e} W/m2")

    results['thermal'] = {
        'T_min_K': float(np.min(thermal.temperature)),
        'T_max_K': float(np.max(thermal.temperature)),
        'T_avg_core_K': float(_avg_core_temp(mesh, thermal.temperature)),
        'converged': thermal.converged,
        'iterations': thermal.iterations,
        'max_heat_flux_W_m2': float(np.max(np.linalg.norm(thermal.heat_flux, axis=1))),
    }

    struct = coupled_result.structural
    disp_mag = np.sqrt(struct.displacement[:, 0]**2 + struct.displacement[:, 1]**2)
    print(f"\nStructural:")
    print(f"  Max displacement:     {np.max(disp_mag):.4e} m "
          f"({np.max(disp_mag)*1e6:.1f} um)")
    print(f"  Max von Mises:        {np.max(struct.von_mises):.3e} Pa "
          f"({np.max(struct.von_mises)/1e6:.1f} MPa)")
    print(f"  Max stress intensity: {np.max(struct.stress_intensity):.3e} Pa "
          f"({np.max(struct.stress_intensity)/1e6:.1f} MPa)")

    # Vessel stress check (zone 3 for Level 1)
    if level == 1:
        vessel_mask = struct.mesh.material_ids == 3
        if np.any(vessel_mask):
            vessel_vm = struct.von_mises[vessel_mask]
            print(f"  Vessel max VM:        {np.max(vessel_vm):.3e} Pa "
                  f"({np.max(vessel_vm)/1e6:.1f} MPa)")
            print(f"  Vessel allowable:     55 MPa (ASME)")
            if np.max(vessel_vm) / 1e6 < 55:
                print(f"  --> PASS (margin = {(55 - np.max(vessel_vm)/1e6):.1f} MPa)")
            else:
                print(f"  --> EXCEEDS allowable by {(np.max(vessel_vm)/1e6 - 55):.1f} MPa")

    results['structural'] = {
        'max_displacement_m': float(np.max(disp_mag)),
        'max_von_mises_Pa': float(np.max(struct.von_mises)),
        'max_stress_intensity_Pa': float(np.max(struct.stress_intensity)),
    }

    # --- Save results JSON ---
    results_file = os.path.join(output_dir, 'fea_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_file}")

    # --- List generated figures ---
    print(f"\nGenerated figures in {output_dir}/:")
    for fname in sorted(os.listdir(output_dir)):
        if fname.endswith('.png'):
            fpath = os.path.join(output_dir, fname)
            size_kb = os.path.getsize(fpath) / 1024
            print(f"  {fname} ({size_kb:.0f} KB)")

    print("\n" + "=" * 70)
    print("FEA analysis complete.")
    print("=" * 70)

    return results


def _avg_core_temp(mesh, temperature):
    """Compute average temperature in core region (zone 0).

    Parameters
    ----------
    mesh : Mesh
    temperature : ndarray, shape (N_nodes,)

    Returns
    -------
    T_avg : float
    """
    core_elements = np.where(mesh.material_ids == 0)[0]
    if len(core_elements) == 0:
        return np.mean(temperature)

    # Average over core element centroids
    T_sum = 0.0
    for e in core_elements:
        conn = mesh.elements[e]
        T_sum += np.mean(temperature[conn[:3]])  # Corner nodes

    return T_sum / len(core_elements)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='40 MWth Marine MSR FEA Analysis',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        '--level', type=int, default=1, choices=[1, 2, 3],
        help='Geometry level:\n'
             '  1 = R-Z axisymmetric full core (default)\n'
             '  2 = Single hexagonal unit cell\n'
             '  3 = 7-channel rosette cluster'
    )
    parser.add_argument(
        '--physics', type=str, default='all',
        help='Physics to run (comma-separated):\n'
             '  all = all physics (default)\n'
             '  neutronics,thermal,structural'
    )
    parser.add_argument(
        '--power', type=float, default=40e6,
        help='Total thermal power in watts (default: 40e6)'
    )
    parser.add_argument(
        '--output', type=str, default='results/fea',
        help='Output directory (default: results/fea)'
    )
    parser.add_argument(
        '--benchmark', action='store_true',
        help='Run analytical benchmarks instead of coupled analysis'
    )

    args = parser.parse_args()

    if args.benchmark:
        from fea.validation.analytical_benchmarks import run_all_benchmarks
        run_all_benchmarks()
        return

    if args.physics == 'all':
        physics = None
    else:
        physics = [p.strip() for p in args.physics.split(',')]

    run_fea_analysis(
        level=args.level,
        physics=physics,
        total_power=args.power,
        output_dir=args.output,
    )


if __name__ == '__main__':
    main()
