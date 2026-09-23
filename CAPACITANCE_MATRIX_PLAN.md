# Capacitance Matrix Characterization Plan

> **Historical document.** Implementation status, contracts, validation evidence,
> and next work are maintained in `CAPACITANCE_MATRIX_SSOT.md`.

## Goal

Add an opt-in characterization profile that captures enough signed terminal
capacitance data for symbolic small-signal analysis without changing existing
LUT generation or legacy file consumers.

## Decisions

- Store Matrix9: the G/D/S rows and columns of `Cij = dQi/dVj`.
- Reconstruct the bulk row and column from conservation; do not store redundant terms.
- Store total intrinsic plus extrinsic capacitance.
- Keep legacy mode as the default and isolate Matrix9 outputs with `_cm9`.
- Support BSIM4 and PSP built-ins plus BSIM-CMG through JSON device definitions.
- Characterize capacitance only; do not add `GBD` or `GBS` in this profile.
- Probe simulator/model capabilities before starting production jobs.

## Implemented Functions

### Canonical schema and adapters (`capacitance.py`)

- `normalize_model_family()` maps model-family aliases.
- `required_native_fields()` defines the simulator outputs each adapter needs.
- `native_to_matrix9()` normalizes BSIM4, PSP, and BSIM-CMG data, including
  each model family's off-diagonal sign convention.
- `matrix9_to_fields()` and `fields_to_matrix9()` convert between tensors and LUT fields.
- `matrix9_to_matrix16()` reconstructs the G/D/S/B matrix.
- `validate_matrix9()` rejects invalid shapes and non-finite data and verifies conservation.

### Characterization runner (`run_lut_char_all.py`)

- `load_device_configs()` validates and registers repeatable JSON device files.
- `_instance_line()` and `_model_block()` render built-in or JSON-defined netlists.
- `_matrix_save_params()` and `_build_save_lines()` select profile-specific OP outputs.
- `probe_capacitance_outputs()` fails fast when required outputs are unavailable.
- `parse_and_save()` writes Matrix9 tensors and schema metadata when opted in.
- `run_pvt()` propagates profile identity through jobs and `_cm9` filenames.

### Merge and export

- `merge_mats.merge_parts()` validates profile/schema identity before L-axis merging.
- `merge_to_nc.build_dataset()` keeps Matrix9 variables and metadata separate from legacy data.
- NetCDF variables use compression to limit the storage increase from six to nine capacitance tensors.

## Validation

- Unit tests cover model sign normalization, extrinsic stamps, bulk reconstruction,
  missing fields, JSON adapters, parser metadata, profile-safe merging, and NetCDF assembly.
- A real `--cap-matrix --test-run` should be run for one BSIM4 and one PSP device.
- A BSIM-CMG test run requires a site-specific JSON config and a model compiled with
  `INFO` outputs; no concrete PDK config is bundled.
