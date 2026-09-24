from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.io import loadmat

from run_lut_char_all import DevCfg
import spectre_backend as spectre


CAPS = spectre.MATRIX9_KEYS
DC = spectre.CORE + CAPS


def cfg(**changes):
    native_names = {"ID": "ids", "VT": "vth", "GMB": "gmbs"}
    values = dict(
        key="demo:nch_lvt", device="nch_lvt_mac", pdk="demo", fet_type="nfet",
        model_lib="/models/top.scs", lib_corner_map={"TT": "top_tt"},
        model_setup_lines=['include "{MODEL_LIB}" section={LIB_CORNER}'],
        instance_template="m0 ({D} {G} {S} {B}) {DEVICE} l={LX}u nfin={NFING} nf={NF}",
        save_pfx="", l_vec=[0.016], vgs_max=1.0, vds_max=1.0,
        vdd=1.0, nfing=4, nf=4, simulator="spectre", caps_model="bsimcmg",
        spectre_dc_signals={name: f"m0:{native_names.get(name, name.lower())}"
                            for name in DC},
        spectre_parasitic_signals={"CJDT": "m0:cjdt", "CJST": "m0:cjst"},
        spectre_sat_signals={"VDSSAT": "m0:vdssat"},
        spectre_noise_signals={"STH": "m0:therm_sid", "SFL": "m0:flicker"},
    )
    values.update(changes)
    return DevCfg(**values)


def _psf(lines):
    return "HEADER\nPSFversion 1.00\nTYPE\nTRACE\nVALUE\n" + "\n".join(lines) + "\nEND\n"


def test_psfascii_reads_scalar_and_struct_noise(tmp_path):
    path = tmp_path / "noise.noise"
    path.write_text('HEADER\nTYPE\n"mosNoise" STRUCT(\n'
                    '"therm_sid" FLOAT DOUBLE PROP(\n"units" "A2/Hz"\n)\n'
                    '"flicker" FLOAT DOUBLE\n)\nTRACE\n"m0" "mosNoise"\n'
                    'VALUE\n"m0" (\n1e-24\n2e-25\n)\n'
                    '"m0" (\n3e-24\n4e-25\n)\nEND\n')
    data = spectre.read_psfascii(path)
    np.testing.assert_allclose(data["m0:therm_sid"], [1e-24, 3e-24])
    np.testing.assert_allclose(data["m0:flicker"], [2e-25, 4e-25])


def test_spectre_job_makes_complete_matrix_lut(tmp_path, monkeypatch):
    conf = cfg()
    monkeypatch.setattr(spectre.shutil, "which", lambda executable: "/bin/spectre")
    calls = []

    def fake_run(argv, cwd, **kwargs):
        calls.append(argv)
        netlist = Path(argv[1]).read_text()
        assert "sweepvds sweep param=ds" in netlist
        assert "sweepvgs_noise noise freq=1" in netlist
        assert "section=top_tt" in netlist
        assert "save m0 d g b" in netlist
        raw = Path(cwd) / "techsweep.raw"
        raw.mkdir()
        dc = []
        for vds in (0.0, 0.5):
            for vgs in (0.0, 0.5):
                dc += [f'"g" {vgs}', f'"d" {vds}', '"b" 0']
                for name in DC:
                    val = (vgs + vds + 1) if name == "ID" else (1 if name in ("CGG", "CDD", "CSS") else 0.1)
                    dc.append(f'"{conf.spectre_dc_signals[name]}" {val}')
                dc += ['"m0:cjdt" 0.02', '"m0:cjst" 0.03', '"m0:vdssat" 0.15']
        (raw / "sweepvds_sweepvgs-sweep").write_text(_psf(dc))
        noise = []
        for _ in range(4):
            noise += ['"m0:therm_sid" 1e-24', '"m0:flicker" 2e-25']
        (raw / "sweepvds_noise_sweepvgs_noise-sweep").write_text(_psf(noise))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(spectre.subprocess, "run", fake_run)
    path = tmp_path / "out.mat"
    spectre.run_spectre_job(conf, "TT", 27, [0.016], [0.0],
                           np.array([0.0, 0.5]), np.array([0.0, 0.5]),
                           tmp_path / "sim", path)
    data = next(v for k, v in loadmat(path, simplify_cells=True).items()
                if not k.startswith("_"))
    assert data["ID"].shape == (2, 2)
    assert data["ID"][1, 0] == 1.5
    assert data["CGD"][0, 0] == -0.1
    assert data["CJDT"][0, 0] == 0.02
    assert data["VDSSAT"][0, 0] == 0.15
    assert data["SFL"][0, 0] == 2e-25
    assert data["CAPACITANCE_JUNCTION_MODE"] == "native_total"
    assert len(calls) == 1
    spectre.run_spectre_job(conf, "TT", 27, [0.016], [0.0],
                           np.array([0.0, 0.5]), np.array([0.0, 0.5]),
                           tmp_path / "sim", path)
    assert len(calls) == 1
    with pytest.raises(spectre.SpectreDataError, match="different grid"):
        spectre.run_spectre_job(cfg(spectre_cap_junction_mode="add_junction"),
                               "TT", 27, [0.016], [0.0],
                               np.array([0.0, 0.5]), np.array([0.0, 0.5]),
                               tmp_path / "sim", path)


def test_pmos_signs_and_explicit_junction_stamp():
    conf = cfg(fet_type="pfet", spectre_cap_junction_mode="add_junction")
    traces = {spec: np.array([0.0]) for spec in conf.spectre_dc_signals.values()}
    traces.update({"m0:ids": np.array([-2e-6]), "m0:vth": np.array([-0.3]),
                   "m0:cdd": np.array([1.0]), "m0:css": np.array([2.0]),
                   "m0:cgd": np.array([0.1]), "m0:cjdt": np.array([0.2]),
                   "m0:cjst": np.array([0.3]), "m0:vdssat": np.array([-0.15])})
    data = spectre._normalize_dc(conf, traces, 1, 1, 1)
    assert data["ID"][0, 0] == 2e-6
    assert data["VT"][0, 0] == 0.3
    assert data["CGD"][0, 0] == -0.1
    assert data["CDD"][0, 0] == 1.2
    assert data["CSS"][0, 0] == 2.3
    assert data["VDSSAT"][0, 0] == 0.15


def test_spectre_missing_required_signal_fails_without_mat(tmp_path, monkeypatch):
    monkeypatch.setattr(spectre.shutil, "which", lambda executable: "/bin/spectre")
    bad = cfg(spectre_noise_signals={"STH": "m0:therm_sid"})
    path = tmp_path / "out.mat"
    with pytest.raises(spectre.SpectreDataError, match="STH and SFL"):
        spectre.run_spectre_job(bad, "TT", 27, [0.016], [0.0],
                               [0.0], [0.0], tmp_path / "sim", path)
    assert not path.exists()
