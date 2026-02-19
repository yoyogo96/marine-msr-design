# Monte Carlo Neutronics Analysis Results
## 40 MWth Marine Molten Salt Reactor - Basic Design Phase

### Simulation Overview

| Parameter | Value |
|-----------|-------|
| Code | Custom 2-group Monte Carlo neutron transport |
| Energy groups | 2 (Fast: >0.625 eV, Thermal: <0.625 eV) |
| Geometry | 3D hexagonal lattice, cylindrical core + reflector |
| Solver | Power iteration (k-eigenvalue) |
| Tallies | Track-length flux estimator, cylindrical mesh |
| Source convergence | Shannon entropy monitoring |

### Core Geometry

| Parameter | Value |
|-----------|-------|
| Core diameter | 124.5 cm |
| Core height | 149.4 cm |
| Fuel channels | 547 (hexagonal lattice) |
| Channel diameter | 2.5 cm |
| Hex pitch | 5.0 cm (flat-to-flat) |
| Fuel salt fraction | 23% |
| Graphite fraction | 77% |
| Radial reflector | 15.0 cm graphite |
| Axial reflector | 15.0 cm graphite (top + bottom) |

### Cross-Section Data (Calibrated)

**Fuel salt (FLiBe + 5 mol% UF4, 12% enrichment, 650°C)**

| Group | Σ_t (cm⁻¹) | Σ_a (cm⁻¹) | Σ_f (cm⁻¹) | νΣ_f (cm⁻¹) |
|-------|-----------|-----------|-----------|------------|
| Fast (>0.625 eV) | 0.226 | 0.010 | 0.002 | 0.005 |
| Thermal (<0.625 eV) | 0.357 | 0.052 | 0.040 | 0.098 |

**Graphite moderator (IG-110, 650°C)**

| Group | Σ_t (cm⁻¹) | Σ_a (cm⁻¹) | Σ_s total (cm⁻¹) |
|-------|-----------|-----------|------------|
| Fast | 0.386 | 0.0001 | 0.386 |
| Thermal | 0.503 | 0.00023 | 0.503 |

Cross-sections calibrated against microscopic calculations using ENDF/B-VIII.0 evaluated data. Temperature/enrichment/density scaling included.

### Key Results

#### 1. Eigenvalue at Design Enrichment (12%)

| Parameter | MC Value | Conceptual Design | Difference |
|-----------|---------|-------------------|------------|
| k_eff | 1.510 ± 0.027 | 1.146 (4-factor) | +36,400 pcm |
| k_eff (no axial refl) | 1.495 ± 0.016 | - | - |
| Leakage fraction | 13.4% | 30.2% | MC has reflector |
| Non-leakage probability | 0.866 | 0.698 | Reflector savings |

**Note**: The large k_eff at 12% indicates massive excess reactivity (~51,000 pcm). This is because 5 mol% UF4 at 12% enrichment provides far more fissile material than needed for criticality. The difference from conceptual design is primarily due to:
1. Radial + axial graphite reflectors (not in diffusion model)
2. Heterogeneous lattice effects (better moderation geometry)
3. 2-group energy resolution vs 1-group diffusion

#### 2. Critical Enrichment Search (Bisection Method)

| Iteration | Enrichment | k_eff ± σ | Status |
|-----------|-----------|-----------|--------|
| 1 | 7.50% | 1.327 ± 0.012 | Supercritical |
| 2 | 5.25% | 1.158 ± 0.013 | Supercritical |
| 3 | 4.12% | 1.049 ± 0.007 | Slightly supercritical |
| 4 | 3.56% | 0.974 ± 0.008 | Subcritical |
| 5 | 3.84% | 1.034 ± 0.010 | Supercritical |
| **6** | **3.70%** | **1.012 ± 0.010** | **Converged** |

**MC Critical enrichment: 3.7 ± 0.2%** (without axial reflector)
**Estimated with axial reflector: ~3.5%**

Conceptual design (1-group diffusion): 7.35%

**Design implication**: The reactor can achieve criticality at LEU enrichment well below the 5% natural boundary. At 12% design enrichment, ~47,000 pcm of excess reactivity is available for:
- Burnup compensation
- Xenon/samarium fission product poisoning
- Temperature defect
- Control margin

#### 3. Temperature Coefficient of Reactivity

| Parameter | Value |
|-----------|-------|
| T_low | 600°C (873.15 K) |
| T_high | 700°C (973.15 K) |
| k_eff (600°C) | 1.530 ± 0.008 |
| k_eff (700°C) | 1.489 ± 0.008 |
| **dk/dT** | **-40.7 ± 11.8 pcm/K** |
| Significance | 3.5σ (statistically significant) |

**The temperature coefficient is STRONGLY NEGATIVE**, confirming inherent safety.

Physical mechanisms:
1. **Fuel salt density decrease** (-0.488 kg/m³ per °C) → reduced macroscopic XS → more leakage
2. **Doppler broadening** of U-238 resonances → increased parasitic absorption
3. **Thermal upscatter increase** → harder spectrum → less thermal fission

Compared to conceptual design's fuel-only α = -8.3 pcm/K, the MC value (-41 pcm/K) includes ALL temperature effects simultaneously and is larger in magnitude.

#### 4. Power Distribution

At 12% enrichment with reflector:
| Parameter | MC Value | Conceptual Design |
|-----------|---------|-------------------|
| Axial peaking factor | 1.70 | 1.54 |
| Radial peaking factor | 2.30 | 1.59 |
| Total peaking factor | 3.92 | 2.45 |

The MC predicts higher peaking factors due to:
- Heterogeneous fuel channel effects (thermal flux peaking at channel surface)
- Reflector-induced flux peaking near core periphery
- 2-group energy resolution capturing spectral effects

### Simulation Statistics

| Run | Particles/batch | Batches (inactive+active) | Histories | Wall time |
|-----|----------------|--------------------------|-----------|-----------|
| Quick eigenvalue (12%) | 500 | 8+12 | 10,000 | 28 s |
| Enrichment search (per point) | 800 | 10+20 | 24,000 | 130 s |
| Temperature coefficient (per point) | 800 | 10+20 | 24,000 | 72 s |

### Comparison: MC vs Conceptual Design

| Parameter | Conceptual (Diffusion) | MC Transport | MC More Accurate Because |
|-----------|----------------------|--------------|--------------------------|
| k_eff method | 1-group diffusion + 4-factor | 2-group Monte Carlo | Transport > diffusion for heterogeneous |
| Geometry | Bare cylinder, homogenized | Hex lattice + reflectors | Explicit heterogeneity |
| Energy groups | 1 | 2 | Better spectral resolution |
| Leakage model | Analytical B² | Particle tracking | Exact geometry boundaries |
| Reflector | None | 15 cm radial + axial | Major physics addition |
| Temperature coeff | Perturbation theory | Differential MC | All effects included |

### Limitations and Uncertainties

1. **2-group approximation**: Only fast/thermal groups. No resonance-region detail. U-238 resonance self-shielding is approximate.
2. **Cross-section source**: Hand-calibrated from microscopic data, not from formal NJOY group condensation with S(α,β) thermal scattering.
3. **Analog transport**: Higher statistical variance than implicit-capture methods. Production runs need >5000 particles/batch.
4. **No burnup**: Clean-core, beginning-of-life only. No fission product buildup.
5. **No delayed neutrons in transport**: Appropriate for eigenvalue calculations.
6. **Statistical uncertainty**: Reduced-statistics runs (800 particles) give ~1% uncertainty on k_eff. Full production runs recommended for final design.

### Recommendations for Next Design Phase

1. **Reduce design enrichment**: Critical enrichment ~3.5-3.7% suggests 5-7% enrichment provides adequate excess reactivity with margin.
2. **Control system design**: At 12% enrichment, ~50,000 pcm excess reactivity requires substantial control worth (rods + soluble poison).
3. **Higher-fidelity validation**: Compare with continuous-energy MC (Serpent-2/OpenMC) for cross-section validation.
4. **Burnup analysis**: Implement depletion coupling to assess cycle length at reduced enrichment.
5. **Hot channel analysis**: Use MC power distribution for detailed thermal-hydraulic hot channel factors.
