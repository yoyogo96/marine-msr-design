# 40 MWth Marine Molten Salt Reactor (MSR) - Design Analysis Package

[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status: Active Development](https://img.shields.io/badge/Status-Active%20Development-green.svg)]()

---

## English / 한국어

## Overview | 개요

This package contains a complete **nuclear reactor design analysis** for a 40 MWth marine molten salt reactor (MSR) suitable for integration into a 6,000 TEU container ship power plant. The design features a **graphite-moderated, FLiBe-fueled core** with **inherently safe negative temperature feedback** and advanced Monte Carlo neutron transport modeling.

본 패키지는 **6,000 TEU 컨테이너선**의 동력원으로 적합한 **40 MWth 해양 용융염 원자로(MSR) 완전 설계 분석**을 포함합니다. 설계는 **그래파이트 감속 및 FLiBe 연료염**을 사용하며 **고유 안전성(negative feedback)**과 **고급 몬테카를로 중성자 수송 해석**을 특징으로 합니다.

### Key Features | 주요 특징

- **Conceptual & Basic Design Phases**: 8 analysis modules covering all disciplines (~23,000 lines Python)
- **Advanced Monte Carlo Neutronics**: Custom 2-group neutron transport code (~7,000 lines Python)
- **Comprehensive Report**: 135-page Korean PDF design documentation
- **Integrated Analysis**: Neutronics, thermal-hydraulics, structural, shielding, safety transient, and drain tank analysis

---

## Key Results | 주요 결과

### Nuclear Performance

| Parameter | Value | Unit | Basis |
|-----------|-------|------|-------|
| **k_eff (Design Enrichment)** | 1.510 ± 0.027 | - | MC with 12% U-235, + reflector |
| **MC Critical Enrichment** | 3.70 ± 0.2 | % | LEU, via bisection search |
| **Temperature Coefficient** | -40.7 ± 11.8 | pcm/K | Strongly negative (inherently safe) |
| **Leakage Fraction (MC)** | 13.4 | % | 2-group MC with radial/axial reflectors |
| **Excess Reactivity (12%)** | ~51,000 | pcm | Available for burnup, Xe/Sm poison, control |

### Thermal-Hydraulic & Structural

| Parameter | Value | Unit |
|-----------|-------|------|
| **Thermal Power** | 40 | MWth |
| **Electrical Power** | 16 | MWe (40% efficiency, sCO₂ Brayton) |
| **Core Diameter** | 124.5 | cm |
| **Core Height** | 149.4 | cm |
| **Fuel Channels** | 547 | (hexagonal lattice) |
| **Inlet / Outlet Temperature** | 600 / 700 | °C |
| **Core Power Density** | 22.0 | MW/m³ |
| **Fuel Salt Inventory** | 2,600 | kg (core + loop + HX) |
| **Total U Mass** | 446 | kg |
| **U-235 Mass** | 53.5 | kg |

### Safety Characteristics

| Parameter | Value | Status |
|-----------|-------|--------|
| **Drain Tank k_eff** | 0.294 | ✓ Subcritical |
| **Temperature Feedback** | Negative | ✓ Inherently safe |
| **ULOF Peak Temp** | 659°C | ✓ Below limits |
| **UTOP Peak Temp** | 681°C | ✓ Below limits |
| **All Thermal Limits** | Met | ✓ Yes |

---

## Monte Carlo Neutronics Results | 몬테카를로 중성자 해석 결과

The **basic design phase** includes a custom **2-group Monte Carlo neutron transport code** for advanced criticality analysis. This represents a major advancement from the 1-group diffusion approach used in conceptual design.

**기본 설계 단계**에는 **커스텀 2군 몬테카를로 중성자 수송 코드**가 포함되어 있으며, 이는 개념설계의 1군 확산 모델에서 큰 진전을 보여줍니다.

### Energy Structure
- **Group 1 (Fast)**: E > 0.625 eV (slowing-down spectrum)
- **Group 2 (Thermal)**: E < 0.625 eV (Maxwellian thermal spectrum)

### Geometry
- **Core**: 124.5 cm dia × 149.4 cm height, 547 fuel channels in hexagonal lattice
- **Reflectors**: 15 cm graphite radial + 15 cm axial (top & bottom)
- **Cross-sections**: ENDF/B-VIII.0 calibrated, temperature and density scaling

### Critical Enrichment Search
Via bisection method: **MC predicts 3.70% ± 0.2% critical enrichment (LEU)**
- Conceptual design (1-group diffusion): 7.35%
- **Reason for difference**: Reflector savings + heterogeneous lattice effects + 2-group resolution

### Temperature Coefficient
**dk/dT = -40.7 ± 11.8 pcm/K** (3.5σ significance)
- Mechanisms: FLiBe density feedback + Doppler in U-238 resonances
- Result: **Strongly negative feedback for inherent safety**

---

## Project Structure | 프로젝트 구조

```
├── config.py                          # Design parameters (40 MWth basis)
├── main.py                            # Integrated analysis runner [1-8/8]
├── requirements.txt                   # Python dependencies
│
├── neutronics/                        # Conceptual design neutronics (1-group diffusion)
│   ├── cross_sections.py              # Homogenized cross-section calculation
│   ├── criticality.py                 # Criticality analysis & 4-factor method
│   ├── diffusion.py                   # 1-group diffusion eigenvalue solver
│   ├── core_geometry.py               # Core geometric design
│   ├── reactivity.py                  # Reactivity coefficients
│   └── burnup.py                      # Burnup analysis
│
├── mc_neutronics/                     # Basic design: 2-group Monte Carlo transport
│   ├── constants.py                   # Physical constants & energy group structure
│   ├── materials.py                   # 2-group cross-sections (ENDF calibrated)
│   ├── geometry.py                    # 3D hexagonal lattice + reflectors
│   ├── particle.py                    # Neutron particle transport kernel
│   ├── tallies.py                     # Track-length flux estimator
│   ├── eigenvalue.py                  # Power iteration k-eigenvalue solver
│   ├── analysis.py                    # Post-processing: flux spectrum, power, 4-factor
│   ├── reactivity.py                  # Temperature & enrichment coefficients
│   └── run_mc.py                      # CLI: Monte Carlo execution
│
├── thermal_hydraulics/                # Thermal-hydraulic analysis
│   ├── salt_properties.py             # FLiBe thermophysical correlations
│   ├── channel_analysis.py            # Single-channel heat transfer
│   ├── temperature.py                 # Core temperature distribution
│   ├── coolant_loop.py                # Primary loop hydraulics
│   └── loop_model.py                  # Coupled loop dynamics
│
├── heat_exchanger/                    # Intermediate heat exchanger design
│   ├── design.py                      # Shell-and-tube HX sizing
│   └── performance.py                 # Thermal-hydraulic performance
│
├── structural/                        # Structural & mechanical design
│   ├── vessel_design.py               # Reactor vessel (ASME pressure vessel code)
│   ├── thermal_stress.py              # Thermal stress analysis
│   └── seismic_marine.py              # Seismic + ship motion loads
│
├── shielding/                         # Neutron & gamma shielding
│   ├── source_term.py                 # Fission source calculation
│   ├── attenuation.py                 # Neutron & gamma attenuation
│   └── dose_rate.py                   # Dose rate mapping
│
├── safety/                            # Safety and transient analysis
│   ├── point_kinetics.py              # Point kinetics solver
│   ├── transients.py                  # Accident analysis (ULOF, UTOP, SBO)
│   └── drain_tank.py                  # Emergency drain tank sizing
│
├── report/                            # Design report generation
│   ├── generate_pdf.py                # PDF generator
│   ├── design_report.md               # Markdown template
│   ├── chapters/                      # Report chapters (ch01-ch09)
│   └── 40MWth_Marine_MSR_Design_Report.pdf
│
├── results/                           # Analysis output & summary
│   ├── summary.md                     # Complete results summary
│   ├── mc_neutronics_summary.md       # MC analysis results
│   └── summary.txt
│
└── utils/                             # Utility functions
    ├── plotting.py                    # Visualization
    └── tables.py                      # Formatted output
```

---

## Installation | 설치 방법

### Requirements
- **Python 3.9+**
- **pip** or **conda** package manager

### Setup

```bash
# Clone or download the repository
cd /path/to/msr

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies
- **numpy** >= 1.21  (numerical computation)
- **scipy** >= 1.7   (scientific algorithms)
- **matplotlib** >= 3.5  (plotting)

---

## Usage | 사용 방법

### Running Integrated Analysis (Conceptual Design)

```bash
# Execute all 8 analysis modules
python main.py
```

This runs:
1. **Foundation**: Core configuration & derived parameters
2. **Neutronics**: Cross-sections, criticality, 4-factor, burnup
3. **Thermal-Hydraulics**: Channel analysis, temperature distribution
4. **Heat Exchanger**: Shell-and-tube HX design
5. **Structural**: Vessel design, thermal stress, ship loads
6. **Shielding**: Source term, attenuation, dose rates
7. **Safety**: Transient analysis (ULOF, UTOP, SBO)
8. **Drain Tank**: Emergency drain tank subcriticality

**Output**: Comprehensive results summary to console and `results/summary.md`

### Running Monte Carlo Neutronics (Basic Design)

```bash
# Production quality (full statistics, ~30-60 min)
python3 mc_neutronics/run_mc.py

# Quick run (reduced statistics, ~1-2 min)
python3 mc_neutronics/run_mc.py --quick

# Reactivity coefficients only
python3 mc_neutronics/run_mc.py --coefficients
```

**Output**:
- `results/mc_neutronics_summary.md` - Full MC analysis report
- Console output with eigenvalue, critical enrichment, temperature coefficient

### Configuration

Edit `config.py` to modify design parameters:
- Thermal power, enrichment, salt composition
- Operating temperatures, pressures
- Geometric targets (H/D ratio, channel pitch)
- Material properties

Example:
```python
THERMAL_POWER = 40e6        # W (40 MWth)
ELECTRICAL_POWER = 16e6     # W (16 MWe)
THERMAL_EFFICIENCY = 0.40   # 40% sCO2 Brayton
U235_ENRICHMENT = 0.12      # 12% HALEU (iterated to criticality)
```

---

## Design Modules | 설계 모듈

### 1. Neutronics (Conceptual Design)

**File**: `neutronics/`

Implements **1-group diffusion theory** analysis:
- Homogenized cross-section calculation (volume-weighted)
- 4-factor neutron balance (f, ε, p, η)
- Cylindrical diffusion eigenvalue solver
- Critical enrichment search (bisection)
- Burnup analysis via 1-group depletion
- Temperature & void reactivity coefficients

**Key equations**:
- Diffusion equation: -∇·(D∇φ) + Σ_a φ = (νΣ_f / k_eff) φ
- 4-factor: k_eff = f · ε · p · η · (1 - M²B²)
- Critical enrichment: iterative search until k_eff = 1.0

### 2. Monte Carlo Neutronics (Basic Design)

**File**: `mc_neutronics/`

Custom **2-group Monte Carlo neutron transport** code:
- **Energy structure**: Fast (>0.625 eV) + Thermal (<0.625 eV)
- **Geometry**: 3D hexagonal lattice, cylindrical core + reflectors
- **Transport**: Analog random walk, explicit collision handling
- **Tallies**: Track-length flux estimator with cylindrical mesh
- **Eigenvalue**: Power iteration (k-eigenvalue method)
- **Cross-sections**: ENDF/B-VIII.0 calibrated, temperature scaling
- **Reaction rates**: Fission, absorption, scattering by group
- **Reactivity**: Enrichment sensitivity, temperature coefficient
- **Convergence**: Shannon entropy source monitoring

**Advantages over diffusion**:
- Captures heterogeneous lattice effects
- Includes 2-group energy coupling
- Radial/axial reflector savings
- Direct k_eff uncertainty quantification

### 3. Thermal-Hydraulics

**File**: `thermal_hydraulics/`

Analyzes core thermal performance:
- FLiBe salt properties (density, viscosity, specific heat, conductivity)
- Single-channel heat transfer with subchannel mixing
- Pressure drop and pumping power
- Core temperature distribution (nominal + hot channel)
- Naturalcirculation fraction
- Reynolds number, Prandtl number, heat transfer coefficient

**Key results**:
- Core inlet: 600°C, outlet: 700°C
- Mass flow rate: 167.6 kg/s
- Reynolds number: 2,242 (transitional/turbulent)
- Peak salt temp (hot channel): 726.5°C
- All thermal limits met

### 4. Heat Exchanger Design

**File**: `heat_exchanger/`

Designs intermediate loop shell-and-tube HX:
- Duty: 40 MWth heat rejection to secondary loop
- LMTD counter-flow calculation
- Overall heat transfer coefficient (U)
- Tube-side and shell-side pressure drops
- Tube count, length, and shell diameter
- Mass and material requirements
- Effectiveness (thermal performance)

### 5. Structural Analysis

**File**: `structural/`

ASME-compliant vessel design:
- Pressure vessel code stress calculations
- Thermal stress from ΔT radial gradients
- Ship motion loads (roll, pitch, heave)
- Creep rupture limits (Hastelloy-N @ 700°C)
- Safety factors (pressure, stress, creep)
- Total reactor assembly mass

### 6. Shielding

**File**: `shielding/`

Neutron and gamma dose rate analysis:
- Fission source term (neutrons + delayed gammas)
- Attenuation through structural materials
- Dose rate at specified distances
- Occupational vs. public exposure limits

### 7. Safety Transients

**File**: `safety/`

Analyzes design basis accidents:
- **ULOF** (Unprotected Loss of Flow): Coasting reactor
- **UTOP** (Unprotected Over-Power): Rod withdrawal
- **SBO** (Station Blackout): Loss of forced circulation

Point kinetics solver with delayed neutron groups validates:
- Peak fuel temperature margins
- Shutdown margin adequacy
- Subcritical drain tank state

### 8. Drain Tank

**File**: `safety/drain_tank.py`

Emergency cooling subsystem:
- Passive drain to subcritical tank
- Volume sizing for safety margin
- k_eff verification (< 0.95)
- Decay heat dissipation time
- Material compatibility

---

## Key Design Parameters | 주요 설계 변수

### Core Geometry
- **Diameter**: 124.5 cm (cylindrical)
- **Height**: 149.4 cm (H/D = 1.2, slightly elongated)
- **Volume**: 1.82 m³
- **Fuel channels**: 547 (hexagonal lattice, 5 cm pitch)
- **Channel diameter**: 2.5 cm
- **Moderator volume fraction**: 77% graphite
- **Fuel salt volume fraction**: 23%

### Fuel Salt (FLiBe + UF₄)
- **Composition**: 64.5 mol% LiF + 30.5 mol% BeF₂ + 5 mol% UF₄
- **Enrichment**: 12% U-235 (HALEU)
- **Density**: ~2,360 kg/m³ @ 650°C
- **Viscosity**: ~7.0 mPa·s @ 650°C
- **Specific heat**: 2,386 J/kg·K
- **Thermal conductivity**: 1.1 W/m·K

### Graphite Moderator (IG-110)
- **Density**: 1,780 kg/m³
- **Total mass in core**: 2,503 kg
- **Thermal conductivity**: 120 W/m·K (unirradiated)
- **Thermal expansion**: 4.5×10⁻⁶ K⁻¹

### Operating Conditions
- **Thermal power**: 40 MWth
- **Core inlet temperature**: 600°C (873 K)
- **Core outlet temperature**: 700°C (973 K)
- **Core average temperature**: 650°C (923 K)
- **Operating pressure**: ~0.2 MPa (near atmospheric)
- **Mass flow rate**: 167.6 kg/s

### Reactor Vessel (Hastelloy-N)
- **Inner diameter**: 1.677 m
- **Wall thickness**: 20 mm (design pressure 0.2 MPa)
- **Vessel height**: 4.14 m (core + plena)
- **Design pressure**: 0.2 MPa (conservative)
- **Design temperature**: 700°C (service limit)

---

## Verification & Validation | 검증

### Neutronics
- ✓ 4-factor neutron balance (k_eff computation)
- ✓ Diffusion eigenvalue solver (cylinder, boundary conditions)
- ✓ Critical enrichment search convergence
- ✓ MC vs. diffusion comparison (large k_eff difference explained by reflector savings)
- ✓ Temperature coefficient sign (negative, inherently safe)

### Thermal-Hydraulics
- ✓ Salt properties correlations (ORNL/TM-2006/12)
- ✓ Convective heat transfer (Nusselt correlation)
- ✓ Pressure drop (Darcy-Weisbach friction factor)
- ✓ Energy balance (mass flow rate from power/ΔT)
- ✓ All thermal limits met

### Structural
- ✓ Pressure vessel code (ASME VIII-1)
- ✓ Safety factors on stress, creep, pressure
- ✓ Hastelloy-N creep rupture data

---

## Limitations & Design Notes | 제한사항

### Nuclear Analysis
1. **Cross-sections**: Hand-calibrated against ENDF/B-VIII.0 microscopic data; not formally homogenized via lattice codes (SERPENT, WIMS)
2. **Energy groups**: 1-group diffusion for conceptual design; 2-group MC for basic design. Full multi-group codes (MCNP, Serpent) recommended for detailed design
3. **Burnup**: Simplified 1-group depletion. No xenon/samarium fission product coupling in diffusion solver (accounted separately)
4. **Control rods**: Not explicitly modeled; shutdown margin estimated from excess reactivity

### Thermal-Hydraulic
1. **Subchannel effects**: Single-channel model with hot-channel factors; detailed CFD recommended for detailed design
2. **Salt properties**: Correlations valid 600–700°C; extended range not validated
3. **Radiation heat transfer**: Not included (low-temperature salt, relatively small effect)

### Structural
1. **Ship motion**: Simplified DNV GL load cases; formal ship-reactor coupled dynamics study recommended
2. **Fatigue**: Thermal cycling not detailed
3. **Corrosion allowance**: Hastelloy-N material selection chosen based on MSRE experience; design life 20 years assumed

### Shielding
1. **Dose mapping**: Simplified point-kernel attenuation; detailed 3D transport (MCNP) recommended for occupational shielding design
2. **Afterheat source**: Conservative 1-hour decay assumed; more detailed decay heat models available

### Safety
1. **Transient analysis**: Point kinetics (lumped model); spatial kinetics (MOX, PARCS equivalent) recommended for licensing
2. **Passive cooling**: Drain tank sized for natural convection; active decay heat removal systems not modeled

---

## Running Example: Step-by-Step | 실행 예제

### Example 1: Conceptual Design Analysis

```bash
# 1. Activate virtual environment
source venv/bin/activate

# 2. Run full integrated analysis
python main.py

# 3. View results
cat results/summary.md
less results/summary.txt
```

**Expected output**: 8 analysis modules executed sequentially, with detailed results for each discipline.

### Example 2: Monte Carlo Criticality Study

```bash
# 1. Quick run to verify installation (~1-2 min)
python3 mc_neutronics/run_mc.py --quick

# 2. Production run for publication (~45 min)
python3 mc_neutronics/run_mc.py

# 3. Extract critical enrichment
grep "Critical enrichment" results/mc_neutronics_summary.md
```

**Expected result**: k_eff = 1.51 ± 0.03 @ 12% enrichment; critical enrichment ≈ 3.7% LEU

### Example 3: Parametric Study (Edit config.py)

```python
# Modify enrichment for sensitivity study
U235_ENRICHMENT = 0.10  # 10% instead of 12%

# Run analysis
python main.py

# Compare results before/after
diff results/summary_old.md results/summary.md
```

---

## References | 참고문헌

### Fundamental MSR Design
1. **ORNL-4541**: Molten-Salt Reactor Program, Quarterly Progress Reports (LeBaron et al.)
2. **ORNL/TM-2006/12**: FLiBe Salt Thermophysical Properties (Williams et al.)
3. **ORNL/TM-2004/197**: Molten-Salt Reactor Technology Transfer, Final Report (Engel et al.)

### Neutronics
1. **ENDF/B-VIII.0**: Evaluated Nuclear Structure and Decay Data (U.S. Nuclear Data Center)
2. **Serpent 2.1.32**: 3D Continuous-Energy Monte Carlo Reactor Physics Burnup Calculation Code (Leppänen et al.)
3. **Fast-thermal coupling**: Honeck & Fineman, "Spectral Shift Model for Heterogeneous Reactors", Nuclear Science & Engineering (1967)

### Thermal-Hydraulics
1. **Shah, MM**: A General Correlation for Heat Transfer to Fluids at Forced Convection, HTFS 3/1 (1978)
2. **ASME Steam Tables**: Comprehensive thermodynamic property data (valid for salt properties calibration)

### Structural
1. **ASME Boiler & Pressure Vessel Code**, Section VIII (Pressure Vessels)
2. **Hastelloy-N Technical Data**: INCO Alloys International, Corrosion Behavior in Molten Salts

### Safety
1. **ANS 53.1**: Standard for Safety Design Criteria for Integral Pressurized Water Reactors (PWRs)
2. **10 CFR 50, Appendix A**: General Design Criteria (U.S. NRC)

### Ship Integration
1. **DNV GL Rules for Classification: Ships** – Part 4, Chapter 4 (Power plants for ships)
2. **IMO SOLAS**: International Convention for Safety of Life at Sea

### Design Report
- **40MWth_Marine_MSR_Design_Report.pdf**: Comprehensive 135-page Korean design documentation (in `report/` directory)

---

## Contributing | 기여

This is a **design analysis package** for educational and research purposes. Contributions welcome:

- Bug reports: Open an issue with reproducibility steps
- Physics improvements: Submit PR with theoretical justification
- Validation studies: Compare against published benchmarks (MSRE, MSBR data)

---

## License | 라이선스

This project is licensed under the **MIT License** – see LICENSE file for details.

---

## Contact & Citation | 연락처 및 인용

**Citation** (if used in academic work):
```bibtex
@software{msr_marine_2025,
  title={40 MWth Marine Molten Salt Reactor Design Analysis Package},
  author={Design Team},
  year={2025},
  url={https://github.com/...}
}
```

**Questions?** Refer to the design report (`report/40MWth_Marine_MSR_Design_Report.pdf`) or the module docstrings.

---

## Acknowledgments | 감사의 말

This design builds upon decades of MSR research at Oak Ridge National Laboratory (ORNL), including the MSRE (Molten-Salt Reactor Experiment) and MSBR (Molten-Salt Breeder Reactor) programs. Material property data, cross-sections, and design methodologies are adapted from ORNL publications and validated against MSRE operational experience.

본 설계는 미국 오크리지 국립연구소(ORNL)의 MSRE(용융염 원자로 실험)와 MSBR(용융염 증식로) 프로그램을 포함한 수십 년의 MSR 연구에 기초하고 있습니다.

---

**Last Updated**: February 2025
**Status**: Conceptual & Basic Design Complete, Detailed Design Phase Pending
