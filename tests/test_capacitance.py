import numpy as np
import pytest

from capacitance import (
    CapacitanceError,
    MATRIX9_KEYS,
    matrix9_to_fields,
    matrix9_to_matrix16,
    native_to_matrix9,
)


def _native_matrix(values):
    names = (
        ("cgg", "cgd", "cgs"),
        ("cdg", "cdd", "cds"),
        ("csg", "csd", "css"),
    )
    return {names[row][col]: np.array([values[row, col]])
            for row in range(3) for col in range(3)}


def test_bsim4_normalizes_signed_intrinsic_and_extrinsic_branches():
    intrinsic = np.array([[10.0, -2.0, -3.0], [-4.0, 20.0, -5.0], [-6.0, -7.0, 30.0]])
    native = _native_matrix(intrinsic)
    native.update(cgdo=[1.0], cgso=[2.0], cgbo=[3.0], capbd=[4.0], capbs=[5.0])

    result = native_to_matrix9("bsim4", native)[0]
    expected = intrinsic.copy()
    expected += np.array([[6.0, -1.0, -2.0], [-1.0, 5.0, 0.0], [-2.0, 0.0, 7.0]])
    np.testing.assert_allclose(result, expected)


def test_psp_flips_only_intrinsic_off_diagonal_signs():
    reported = np.array([[10.0, 2.0, 3.0], [4.0, 20.0, 5.0], [6.0, 7.0, 30.0]])
    native = _native_matrix(reported)
    native.update(cgdol=[1.0], cgsol=[2.0], lp_cgbov=[3.0], cjd=[4.0], cjs=[5.0])

    result = native_to_matrix9("psp103", native)[0]
    intrinsic = reported.copy()
    intrinsic[~np.eye(3, dtype=bool)] *= -1
    expected = intrinsic + np.array([[6.0, -1.0, -2.0], [-1.0, 5.0, 0.0], [-2.0, 0.0, 7.0]])
    np.testing.assert_allclose(result, expected)


def test_bsim_cmg_normalizes_total_outputs_and_reconstructs_bulk():
    reported = np.array([[9.0, 2.0, 3.0], [1.0, 8.0, 4.0], [5.0, 6.0, 7.0]])
    expected = reported.copy()
    expected[~np.eye(3, dtype=bool)] *= -1
    reduced = native_to_matrix9("bsim-cmg", _native_matrix(reported))
    np.testing.assert_allclose(reduced[0], expected)

    full = matrix9_to_matrix16(reduced)
    np.testing.assert_allclose(full.sum(axis=-1), 0.0)
    np.testing.assert_allclose(full.sum(axis=-2), 0.0)
    assert tuple(matrix9_to_fields(reduced)) == MATRIX9_KEYS


def test_missing_native_field_is_rejected():
    native = _native_matrix(np.eye(3))
    native.pop("cdg")
    with pytest.raises(CapacitanceError, match="Missing native capacitance fields: cdg"):
        native_to_matrix9("bsimcmg", native)
