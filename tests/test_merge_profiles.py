import numpy as np
import pytest
from scipy.io import savemat

from capacitance import LEGACY_CAP_KEYS, MATRIX9_KEYS, capacitance_metadata
from merge_mats import merge_parts
from merge_to_nc import build_dataset, collect_files, export_group_streaming


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


def test_streaming_export_keeps_custom_corner_and_spectre_fields(tmp_path):
    extra = capacitance_metadata("bsimcmg", bulk_alias="E")
    extra.update({"VDSSAT": np.full((2, 2, 2, 2), 0.12),
                  "CJDT": np.full((2, 2, 2, 2), 1e-16),
                  "CJST": np.full((2, 2, 2, 2), 2e-16),
                  "CAPACITANCE_JUNCTION_MODE": "native_total",
                  "NOISE_FREQ_HZ": 1.0, "SIMULATOR": "spectre"})
    for temp in (27, 125):
        _write_mat(tmp_path / f"demo_NOM_Tp{temp}_vsb2_cm9.mat",
                   [0.016, 0.02], MATRIX9_KEYS, extra, corner="NOM", temp=temp)
    group = collect_files(tmp_path)["demo_vsb2_cm9"]
    out = tmp_path / "demo.nc"
    export_group_streaming(group, "demo", out, ["NOM"], [27, 125])
    import xarray as xr
    with xr.open_dataset(out) as ds:
        assert ds.corner.values.tolist() == ["NOM"]
        assert ds.VDSSAT.shape == (1, 2, 2, 2, 2, 2)
        assert float(ds.CJDT.isel(corner=0, temp=0, L=0, VGS=0, VDS=0, VSB=0)) == 1e-16
    with pytest.raises(ValueError, match="temperature set"):
        export_group_streaming(group, "demo", tmp_path / "bad.nc", ["NOM"], [27, 85, 125])
    assert not (tmp_path / "bad.nc").exists()
