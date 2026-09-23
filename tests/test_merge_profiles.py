import numpy as np
import pytest
from scipy.io import savemat

from capacitance import LEGACY_CAP_KEYS, MATRIX9_KEYS, capacitance_metadata
from merge_mats import merge_parts
from merge_to_nc import build_dataset, collect_files


CORE = ("ID", "VT", "GM", "GMB", "GDS")


def _write_mat(path, l_values, caps, metadata=None, corner="TT", temp=27,
               vsb_values=(0.0, 0.5)):
    shape = (len(l_values), 2, 2, len(vsb_values))
    data = {
        "L": np.asarray(l_values),
        "VGS": np.array([0.0, 1.0]),
        "VDS": np.array([0.0, 1.0]),
        "VSB": np.array(vsb_values),
        "W": 1.0,
        "NFING": 1.0,
        "CORNER": corner,
        "TEMP": float(temp),
    }
    for index, key in enumerate(CORE + tuple(caps)):
        data[key] = np.full(shape, index + 1.0)
    if metadata:
        data.update(metadata)
    sign = "p" if temp >= 0 else "m"
    savemat(path, {f"demo_{corner}_T{sign}{abs(temp)}": data})


def test_merge_parts_preserves_matrix_profile(tmp_path):
    metadata = capacitance_metadata("bsim4")
    first = tmp_path / "demo_TT_Tp27_vsb2_cm9_L10to10nm.mat"
    second = tmp_path / "demo_TT_Tp27_vsb2_cm9_L20to20nm.mat"
    output = tmp_path / "merged.mat"
    _write_mat(first, [0.01], MATRIX9_KEYS, metadata)
    _write_mat(second, [0.02], MATRIX9_KEYS, metadata)

    _, merged = merge_parts([second, first], output)
    np.testing.assert_allclose(merged["L"], [0.01, 0.02])
    assert merged["CAPACITANCE_PROFILE"] == "matrix9"
    assert merged["CDG"].shape == (2, 2, 2, 2)


def test_merge_parts_rejects_mixed_profiles(tmp_path):
    matrix = tmp_path / "matrix.mat"
    legacy = tmp_path / "legacy.mat"
    _write_mat(matrix, [0.01], MATRIX9_KEYS, capacitance_metadata("bsim4"))
    _write_mat(legacy, [0.02], LEGACY_CAP_KEYS)
    with pytest.raises(ValueError, match="profile|tensor keys"):
        merge_parts([matrix, legacy], tmp_path / "bad.mat")


def test_cm9_files_group_separately_and_build_dataset(tmp_path):
    metadata = capacitance_metadata("bsimcmg", bulk_alias="E")
    path_27 = tmp_path / "demo_TT_Tp27_vsb2_cm9.mat"
    path_hot = tmp_path / "demo_TT_Tp125_vsb2_cm9.mat"
    legacy = tmp_path / "demo_TT_Tp27_vsb2.mat"
    _write_mat(path_27, [0.01, 0.02], MATRIX9_KEYS, metadata, temp=27)
    _write_mat(path_hot, [0.01, 0.02], MATRIX9_KEYS, metadata, temp=125)
    _write_mat(legacy, [0.01, 0.02], LEGACY_CAP_KEYS, temp=27)

    groups = collect_files(tmp_path)
    assert set(groups) == {"demo_vsb2", "demo_vsb2_cm9"}
    ds = build_dataset(groups["demo_vsb2_cm9"], "demo_vsb2_cm9")
    assert set(MATRIX9_KEYS).issubset(ds.data_vars)
    assert "CGB" not in ds
    assert ds.attrs["CAPACITANCE_PROFILE"] == "matrix9"
    assert ds.attrs["BULK_TERMINAL_ALIAS"] == "E"


def test_build_dataset_restores_matlab_squeezed_singleton_axes(tmp_path):
    path = tmp_path / "demo_TT_Tp27_vsb1_cm9.mat"
    _write_mat(path, [0.01, 0.02], MATRIX9_KEYS, capacitance_metadata("bsim4"),
               vsb_values=(0.0,))
    groups = collect_files(tmp_path)
    ds = build_dataset(groups["demo_vsb1_cm9"], "demo_vsb1_cm9")
    assert ds["CDG"].shape == (1, 1, 2, 2, 2, 1)
