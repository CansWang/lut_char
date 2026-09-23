"""Canonical quasi-static MOS terminal-capacitance helpers.

Matrix9 stores the G/D/S rows and columns of the total terminal-charge
Jacobian.  The bulk row and column are reconstructed from charge conservation
and invariance to a common-mode terminal-voltage shift.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


LEGACY_CAP_KEYS = ("CGG", "CGB", "CGD", "CGS", "CDD", "CSS")
MATRIX9_KEYS = (
    "CGG", "CGD", "CGS",
    "CDG", "CDD", "CDS",
    "CSG", "CSD", "CSS",
)
MATRIX16_KEYS = (
    "CGG", "CGD", "CGS", "CGB",
    "CDG", "CDD", "CDS", "CDB",
    "CSG", "CSD", "CSS", "CSB",
    "CBG", "CBD", "CBS", "CBB",
)

CAP_PROFILE_LEGACY = "legacy"
CAP_PROFILE_MATRIX9 = "matrix9"
CAP_SCHEMA_VERSION = 1

_REDUCED_TERMINALS = {"g": 0, "d": 1, "s": 2}

_MATRIX9_NATIVE_FIELDS = {
    "bsim4": (
        "cgg", "cgd", "cgs", "cdg", "cdd", "cds", "csg", "csd", "css",
        "cgdo", "cgso", "cgbo", "capbd", "capbs",
    ),
    "psp": (
        "cgg", "cgd", "cgs", "cdg", "cdd", "cds", "csg", "csd", "css",
        "cgdol", "cgsol", "lp_cgbov", "cjd", "cjs",
    ),
    "bsimcmg": (
        "cgg", "cgd", "cgs", "cdg", "cdd", "cds", "csg", "csd", "css",
    ),
}


class CapacitanceError(ValueError):
    """Raised when native capacitance data cannot form a valid Matrix9."""


def normalize_model_family(model_family: str) -> str:
    """Return the canonical model-family identifier."""
    normalized = model_family.lower().replace("-", "").replace("_", "")
    aliases = {
        "bsim4": "bsim4",
        "psp": "psp",
        "psp103": "psp",
        "bsimcmg": "bsimcmg",
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise CapacitanceError(f"Unsupported capacitance model family: {model_family}") from exc


def required_native_fields(model_family: str) -> tuple[str, ...]:
    """Return native simulator outputs required by the Matrix9 adapter."""
    return _MATRIX9_NATIVE_FIELDS[normalize_model_family(model_family)]


def capacitance_metadata(model_family: str, bulk_alias: str = "B") -> dict[str, object]:
    """Metadata persisted with Matrix9 MAT and NetCDF outputs."""
    return {
        "CAPACITANCE_PROFILE": CAP_PROFILE_MATRIX9,
        "CAPACITANCE_SCHEMA_VERSION": CAP_SCHEMA_VERSION,
        "CAPACITANCE_CONVENTION": "Cij=dQi/dVj",
        "CAPACITANCE_TERMINALS": "G,D,S,B",
        "CAPACITANCE_COMPONENTS": "total",
        "MODEL_FAMILY": normalize_model_family(model_family),
        "BULK_TERMINAL_ALIAS": bulk_alias.upper(),
    }


def _native_array(native: Mapping[str, object], name: str) -> np.ndarray:
    try:
        return np.asarray(native[name], dtype=float)
    except KeyError as exc:
        raise CapacitanceError(f"Missing native capacitance field: {name}") from exc


def _empty_matrix(native: Mapping[str, object]) -> np.ndarray:
    shape = _native_array(native, "cgg").shape
    return np.zeros(shape + (3, 3), dtype=float)


def _stamp_branch(matrix: np.ndarray, a: str, b: str, value: object) -> None:
    """Add a reciprocal branch capacitance to a reduced G/D/S matrix."""
    cap = np.asarray(value, dtype=float)
    ai = _REDUCED_TERMINALS.get(a.lower())
    bi = _REDUCED_TERMINALS.get(b.lower())
    if ai is not None:
        matrix[..., ai, ai] += cap
    if bi is not None:
        matrix[..., bi, bi] += cap
    if ai is not None and bi is not None:
        matrix[..., ai, bi] -= cap
        matrix[..., bi, ai] -= cap


def native_to_matrix9(model_family: str, native: Mapping[str, object]) -> np.ndarray:
    """Normalize model-native capacitances into total signed Matrix9 values."""
    family = normalize_model_family(model_family)
    missing = [name for name in required_native_fields(family) if name not in native]
    if missing:
        raise CapacitanceError("Missing native capacitance fields: " + ", ".join(missing))

    matrix = _empty_matrix(native)
    native_names = (
        ("cgg", "cgd", "cgs"),
        ("cdg", "cdd", "cds"),
        ("csg", "csd", "css"),
    )

    if family == "bsim4":
        for row, names in enumerate(native_names):
            for col, name in enumerate(names):
                matrix[..., row, col] = _native_array(native, name)
    else:
        # PSP and BSIM-CMG report positive diagonal capacitances and
        # -dQi/dVj for off-diagonal transcapacitances.
        for row, names in enumerate(native_names):
            for col, name in enumerate(names):
                sign = 1.0 if row == col else -1.0
                matrix[..., row, col] = sign * _native_array(native, name)

    if family == "bsim4":
        _stamp_branch(matrix, "g", "d", _native_array(native, "cgdo"))
        _stamp_branch(matrix, "g", "s", _native_array(native, "cgso"))
        _stamp_branch(matrix, "g", "b", _native_array(native, "cgbo"))
        _stamp_branch(matrix, "d", "b", _native_array(native, "capbd"))
        _stamp_branch(matrix, "s", "b", _native_array(native, "capbs"))
    elif family == "psp":
        _stamp_branch(matrix, "g", "d", _native_array(native, "cgdol"))
        _stamp_branch(matrix, "g", "s", _native_array(native, "cgsol"))
        _stamp_branch(matrix, "g", "b", _native_array(native, "lp_cgbov"))
        _stamp_branch(matrix, "d", "b", _native_array(native, "cjd"))
        _stamp_branch(matrix, "s", "b", _native_array(native, "cjs"))

    validate_matrix9(matrix)
    return matrix


def matrix9_to_fields(matrix: object) -> dict[str, np.ndarray]:
    """Return canonical Matrix9 field arrays from a trailing 3x3 matrix."""
    arr = np.asarray(matrix, dtype=float)
    if arr.shape[-2:] != (3, 3):
        raise CapacitanceError(f"Expected trailing Matrix9 shape (3, 3), got {arr.shape}")
    return {
        MATRIX9_KEYS[row * 3 + col]: arr[..., row, col]
        for row in range(3)
        for col in range(3)
    }


def fields_to_matrix9(fields: Mapping[str, object]) -> np.ndarray:
    """Build a trailing 3x3 matrix from canonical Matrix9 fields."""
    missing = [name for name in MATRIX9_KEYS if name not in fields]
    if missing:
        raise CapacitanceError("Missing Matrix9 fields: " + ", ".join(missing))
    shape = np.asarray(fields[MATRIX9_KEYS[0]]).shape
    matrix = np.empty(shape + (3, 3), dtype=float)
    for row in range(3):
        for col in range(3):
            matrix[..., row, col] = np.asarray(fields[MATRIX9_KEYS[row * 3 + col]], dtype=float)
    validate_matrix9(matrix)
    return matrix


def matrix9_to_matrix16(matrix: object) -> np.ndarray:
    """Reconstruct the full G/D/S/B capacitance matrix from Matrix9."""
    reduced = np.asarray(matrix, dtype=float)
    if reduced.shape[-2:] != (3, 3):
        raise CapacitanceError(f"Expected trailing Matrix9 shape (3, 3), got {reduced.shape}")

    full = np.zeros(reduced.shape[:-2] + (4, 4), dtype=float)
    full[..., :3, :3] = reduced
    full[..., :3, 3] = -np.sum(reduced, axis=-1)
    full[..., 3, :3] = -np.sum(reduced, axis=-2)
    full[..., 3, 3] = np.sum(reduced, axis=(-2, -1))
    return full


def validate_matrix9(matrix: object) -> None:
    """Validate Matrix9 shape, finite values, and reconstructed conservation."""
    arr = np.asarray(matrix, dtype=float)
    if arr.shape[-2:] != (3, 3):
        raise CapacitanceError(f"Expected trailing Matrix9 shape (3, 3), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise CapacitanceError("Matrix9 contains non-finite capacitance values")

    full = matrix9_to_matrix16_unchecked(arr)
    scale = max(float(np.max(np.abs(full))), 1e-30)
    residual = max(
        float(np.max(np.abs(np.sum(full, axis=-1)))),
        float(np.max(np.abs(np.sum(full, axis=-2)))),
    )
    if residual / scale > 1e-10:
        raise CapacitanceError(
            f"Reconstructed capacitance matrix violates conservation: {residual / scale:.3e}"
        )


def matrix9_to_matrix16_unchecked(reduced: np.ndarray) -> np.ndarray:
    """Internal reconstruction used while validating Matrix9."""
    full = np.zeros(reduced.shape[:-2] + (4, 4), dtype=float)
    full[..., :3, :3] = reduced
    full[..., :3, 3] = -np.sum(reduced, axis=-1)
    full[..., 3, :3] = -np.sum(reduced, axis=-2)
    full[..., 3, 3] = np.sum(reduced, axis=(-2, -1))
    return full
