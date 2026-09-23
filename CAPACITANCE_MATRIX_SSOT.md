# Capacitance Matrix Characterization SSOT

**Status:** Implemented and verified  
**Last updated:** 2026-07-16  
**Repository:** `/home/canswang/lut_char`  
**Branch at last verification:** `main`  
**Schema version:** `1`

This document is the single source of truth for the signed capacitance-matrix
characterization work in `lut_char`. The README remains the user-facing quick
start, and `CAPACITANCE_MATRIX_PLAN.md` is retained only as historical planning
context. If implementation behavior, schema, model normalization, or validation
status changes, update this document in the same change.

## 1. Objective

Extend the LUT characterization pipeline with an opt-in, information-preserving
terminal-capacitance profile suitable for later symbolic small-signal analysis.
The profile must:

- preserve the independent terminal-charge derivatives of a four-terminal MOS;
- include intrinsic and model-reported extrinsic capacitances;
- support the built-in BSIM4 and IHP PSP devices;
- provide an external configuration path for INFO-enabled BSIM-CMG models;
- preserve the existing six-capacitance LUT behavior by default;
- keep legacy and new outputs physically and operationally separate.

The LUT pipeline does not enforce a gm/ID methodology. Its outputs can be used
with gm/ID or another design method after the solution/design space is resolved.

## 2. Scope and Non-Goals

### Implemented

- Signed total-capacitance Matrix9 characterization.
- BSIM4, PSP 103, and BSIM-CMG normalization adapters.
- JSON-defined external devices and simulator-output aliases.
- A fail-fast simulator/model capability probe.
- Profile-aware MAT merging and compressed NetCDF export.
- Unit, real-simulator, legacy-regression, and exporter verification.

### Not implemented

- The flattened-netlist symbolic transfer-function analyzer.
- Direct integration with SLiCAP or Lcapy.
- A reduction from Matrix9 to SLiCAP's simplified reciprocal MOS model.
- A concrete BSIM-CMG PDK/device configuration.
- `GBD`, `GBS`, or other junction-conductance characterization.
- Non-quasi-static, frequency-dependent, distributed, or post-layout parasitics.

## 3. Why Matrix9 Is the Canonical Data

For terminal order `G,D,S,B`, define the incremental charge matrix as:

```text
Cij = dQi/dVj
ii  = s * sum_j(Cij * vj)
```

A four-terminal device begins with 16 derivatives. Charge conservation makes
each column sum to zero, and invariance to a common terminal-voltage shift makes
each row sum to zero. One of those eight equations is redundant, leaving:

```text
16 - 7 = 9 independent capacitance entries
```

The stored Matrix9 is the `G,D,S` block:

```text
CGG  CGD  CGS
CDG  CDD  CDS
CSG  CSD  CSS
```

The bulk row and column are reconstructed exactly:

```text
CiB = -(CiG + CiD + CiS)
CBj = -(CGj + CDj + CSj)
CBB = sum(Cij), i,j in {G,D,S}
```

Off-diagonal terms are signed. In a biased compact MOS model, `Cij` and `Cji`
are different terminal-charge derivatives and are not assumed reciprocal.

SLiCAP's built-in MOS instead uses five ordinary branch capacitors (`cgs`,
`cgb`, `cdg`, `cdb`, and `csb`). That is a useful reciprocal approximation for
compact symbolic expressions, but it cannot preserve a general Matrix9. The LUT
therefore stores Matrix9 first; a topology-aware SLiCAP reduction can be applied
later, after terminal ties and the required analysis accuracy are known.

## 4. Canonical Output Contract

### Tensor fields

Every Matrix9 field has shape:

```text
(nL, nVGS, nVDS, nVSB)
```

Required electrical fields are:

```text
ID VT GM GMB GDS
CGG CGD CGS CDG CDD CDS CSG CSD CSS
```

`STH`, `SFL`, and `VDSAT` remain optional because availability is model- and
analysis-dependent.

### Metadata

Matrix9 MAT structs and NetCDF datasets carry:

| Key | Value/meaning |
| --- | --- |
| `CAPACITANCE_PROFILE` | `matrix9` |
| `CAPACITANCE_SCHEMA_VERSION` | `1` |
| `CAPACITANCE_CONVENTION` | `Cij=dQi/dVj` |
| `CAPACITANCE_TERMINALS` | `G,D,S,B` |
| `CAPACITANCE_COMPONENTS` | `total` |
| `MODEL_FAMILY` | `bsim4`, `psp`, or `bsimcmg` |
| `BULK_TERMINAL_ALIAS` | Native bulk name mapped to canonical `B` |

### Profile isolation

- Legacy mode remains the default.
- Matrix9 is enabled only with `--cap-matrix`.
- Matrix9 files include `_cm9` before any partial-L suffix.
- Legacy and Matrix9 files cannot be merged together.

Example names:

```text
nfet_01v8_TT_Tp27_vsb3.mat
nfet_01v8_TT_Tp27_vsb3_cm9.mat
nfet_01v8_TT_Tp27_vsb3_cm9_L150to1000nm.mat
```

## 5. Model Normalization

Normalization is implemented in `capacitance.py`. All adapters produce total
signed `Cij=dQi/dVj` values in canonical terminal order.

### BSIM4

Required intrinsic native fields:

```text
cgg cgd cgs cdg cdd cds csg csd css
```

Required extrinsic fields:

```text
cgdo cgso cgbo capbd capbs
```

ngspice BSIM4 off-diagonal outputs are already signed charge derivatives. The
adapter keeps their signs and stamps reciprocal overlap/junction branches into
the reduced matrix:

- `cgdo`: G-D branch;
- `cgso`: G-S branch;
- `cgbo`: G-B branch;
- `capbd`: D-B branch;
- `capbs`: S-B branch.

### PSP 103 / IHP

Required intrinsic native fields:

```text
cgg cgd cgs cdg cdd cds csg csd css
```

Required extrinsic fields:

```text
cgdol cgsol lp_cgbov cjd cjs
```

PSP reports positive diagonal terms and `-dQi/dVj` for off-diagonal terms. The
adapter negates only the intrinsic off-diagonals, then stamps:

- `cgdol`: G-D branch;
- `cgsol`: G-S branch;
- `lp_cgbov`: G-B branch;
- `cjd`: D-B branch;
- `cjs`: S-B branch.

### BSIM-CMG

Required native total fields:

```text
cgg cgd cgs cdg cdd cds csg csd css
```

BSIM-CMG INFO outputs already include the model's intrinsic and extrinsic
components. Like PSP, diagonals are reported directly and off-diagonals are
reported as `-dQi/dVj`; the adapter negates only the off-diagonal terms. No
additional branch capacitances are added.

Requirements and naming rules:

- the Verilog-A model must be compiled with INFO outputs enabled;
- the native electrostatic terminal `E` is represented canonically as `B`;
- `bulk_terminal_alias: "E"` records that mapping in output metadata;
- `output_aliases` maps canonical lowercase names to simulator-specific names.

## 6. Runner and Configuration Behavior

### CLI

```bash
# Built-in BSIM4 or PSP device
python run_lut_char_all.py \
    --device sky130:nfet_01v8 \
    --cap-matrix --test-run

# Externally defined device, including BSIM-CMG
python run_lut_char_all.py \
    --device-config devices.json \
    --device mypdk:nmos \
    --cap-matrix --test-run
```

`--device-config` is repeatable. Duplicate registry keys are rejected, including
attempts to replace a built-in device.

### JSON device contract

Required fields are:

```text
key device pdk fet_type model_lib lib_corner_map
l_vec vgs_max vds_max save_pfx instance_template
```

Important optional fields include:

```text
model_family model_include model_setup_lines spiceinit_src
analysis w_um nfing vsb_vec has_explicit_u
id_col gmb_col output_aliases simulation_env bulk_terminal_alias
```

Supported `instance_template` placeholders are:

```text
{D} {G} {S} {B} {DEVICE} {LX} {W_UM} {NFING}
```

Supported `model_setup_lines` placeholders are:

```text
{MODEL_LIB} {MODEL_INCLUDE} {CORNER} {LIB_CORNER} {DEVICE}
```

Paths and environment values expand `~` and environment variables. JSON devices
can use arbitrary supported supply maxima; existing hand-tuned grid definitions
remain unchanged for built-in supply values. Test and smoke modes use `TT` when
available and otherwise use the first configured corner.

### Capability probe

Before any Matrix9 sweep, `probe_capacitance_outputs()` runs a one-point OP
simulation using the same model, instance, output aliases, environment, and
corner setup as production. It aborts before parallel jobs if any required
capacitance or core operating-point output is unavailable.

## 7. Data Flow

```text
Built-in/JSON device config
          |
          v
Matrix9 capability probe
          |
          v
ngspice netlist generation and PVT sweep
          |
          v
native OP fields -> model adapter -> canonical Matrix9
          |
          v
per-PVT MAT + schema metadata
          |
          +--> merge_mats.py (optional partial-L merge)
          |
          v
merge_to_nc.py -> compressed per-device NetCDF
```

## 8. File Responsibilities

| File | Responsibility |
| --- | --- |
| `capacitance.py` | Schema constants, model adapters, branch stamping, Matrix9/Matrix16 conversion, validation |
| `run_lut_char_all.py` | CLI, JSON registry, model/instance rendering, capability probe, simulation, parsing, MAT writing |
| `merge_mats.py` | Profile-safe partial-L MAT merging |
| `merge_to_nc.py` | Profile-safe PVT aggregation and compressed NetCDF export |
| `tests/test_capacitance.py` | Sign, branch-stamp, reconstruction, and missing-field tests |
| `tests/test_runner_matrix.py` | CLI helpers, JSON adapter, netlist, parser, metadata, and corner-fallback tests |
| `tests/test_merge_profiles.py` | Profile isolation, partial merge, singleton-axis, and NetCDF tests |
| `README.md` | User-facing commands and JSON example |
| `CAPACITANCE_MATRIX_PLAN.md` | Historical plan; not normative |

## 9. Merge and Export Invariants

- The required tensor set is selected from `CAPACITANCE_PROFILE`.
- All partial/PVT inputs must have the same profile, optional tensor set, axis
  coordinates, schema metadata, model family, and bulk alias.
- Partial L coordinates are sorted and must be strictly increasing without overlap.
- MATLAB-squeezed singleton axes are restored by checking total tensor size
  against `(nL,nVGS,nVDS,nVSB)`, not by guessing which dimension was removed.
- Matrix9 and legacy filename groups remain separate.
- NetCDF data variables use zlib compression (`complevel=4`, shuffle enabled).

## 10. Verification Record

Verification completed on 2026-07-15/16:

### Automated tests

```text
pytest -q: 12 passed
pyflakes: clean
python compilation/compileall: clean
git diff --check: clean
```

Coverage includes:

- BSIM4 signed intrinsic terms and all five extrinsic branch stamps;
- PSP off-diagonal normalization and all five extrinsic branch stamps;
- BSIM-CMG off-diagonal normalization from INFO conventions;
- Matrix16 bulk reconstruction and conservation;
- missing native-output rejection;
- JSON registration and generic netlist generation;
- Matrix9 MAT fields and metadata;
- legacy/Matrix9 merge rejection;
- MATLAB singleton-axis restoration;
- NetCDF dataset construction and metadata preservation.

### Real ngspice runs

| Run | Result |
| --- | --- |
| Sky130 BSIM4 Matrix9 test | 364 operating points; all four existing sweep validations passed |
| IHP PSP Matrix9 test | 316 operating points; all four existing sweep validations passed |
| Sky130 legacy smoke test | 273 operating points; original six-capacitance path completed successfully |

The real BSIM4 and PSP Matrix9 outputs contained finite values. Reconstructed
row/column conservation residuals were approximately `1e-29`.

### Real export checks

- BSIM4 Matrix9 MAT -> NetCDF: passed.
- PSP Matrix9 MAT -> NetCDF: passed.
- Singleton `VSB=1` dimensions were restored correctly.
- Matrix9 metadata was preserved.
- `CDG` and other data variables reported zlib compression enabled.

### Remaining validation gap

BSIM-CMG normalization is unit-tested and checked against the local BSIM-CMG
111 Verilog-A INFO definitions, but no end-to-end BSIM-CMG simulator run has
been performed because the repository does not contain a concrete INFO-enabled
model/device configuration.

## 11. Known Limitations and Risks

1. Matrix9 is a quasi-static operating-point Jacobian. It does not represent
   NQS behavior or arbitrary frequency-dependent device dynamics.
2. “Total” means intrinsic plus extrinsic components exposed by the compact
   model adapter. It does not include extracted interconnect or layout parasitics.
3. Model output names and INFO availability remain simulator/build dependent;
   the capability probe detects, but cannot repair, missing outputs.
4. Reducing Matrix9 to five SLiCAP branch capacitors is approximate unless the
   relevant cross derivatives are reciprocal and omitted terms are negligible.
5. The symbolic flattened-netlist and performance-database pipeline remains a
   separate future implementation.

## 12. Next Work

1. Add a site-specific JSON configuration for an available INFO-enabled
   BSIM-CMG model and run the same real test/export sequence.
2. Implement the symbolic small-signal consumer that reads the flattened
   netlist and stamps the full signed Matrix9 at the selected LUT operating point.
3. Reconstruct the bulk row/column in that consumer and validate its MNA matrix
   against an ngspice AC reference.
4. Add a topology-aware optional reduction for SLiCAP after terminal ties are
   known; do not replace Matrix9 in the characterization database.
5. Compute transfer functions first, then derive gain, poles/zeros, UGF/UBW,
   phase margin, and other performance metrics into the performance database.

## 13. Change-Control Rules

- Do not change the meaning or sign of an existing field without incrementing
  `CAPACITANCE_SCHEMA_VERSION`.
- Do not remove `_cm9` isolation while legacy output remains supported.
- Every new model adapter must define native required fields, sign convention,
  extrinsic composition, capability-probe behavior, and focused tests.
- Every merge/export change must test profile mismatch and singleton axes.
- Update this SSOT, the README user-facing section, and tests together whenever
  the external contract changes.

## 14. Repository Handoff State

At the time this SSOT was created, the feature was implemented in the local
`main` worktree but had not been committed or pushed. The worktree also contained
pre-existing interpolation/PCHIP benchmark artifacts that are outside the scope
of this capacitance-matrix implementation and were not reverted.
