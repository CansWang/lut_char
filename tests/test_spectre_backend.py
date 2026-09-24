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
        spectre_parasitic_signals={
            "CJDT": "m0:cjdt", "CJST": "m0:cjst",
            "CGDEXT": "m0:cgdext", "CGSEXT": "m0:cgsext", "CGBOV": "m0:cgbov",
        },
        spectre_sat_signals={"VDSAT": "m0:vdsat", "VDSSAT": "m0:vdssat"},
        spectre_noise_signals={"STH": "m0:therm_sid", "SFL": "m0:flicker"},
    )
    values.update(changes)
    return DevCfg(**values)


def _psf(lines):
    return "HEADER\nPSFversion 1.00\nTYPE\nTRACE\nVALUE\n" + "\n".join(lines) + "\nEND\n"


def _noise_psf(vgs_values):
    fields = ("flicker", "shot_igs", "shot_igd", "shot_igb", "therm_rs",
              "therm_rd", "therm_rg", "therm_sid", "total")
    lines = ["HEADER", '"PSFversion" "1.00"', "TYPE",
             '"sweep" FLOAT DOUBLE', '"nch_lvt_mac.10" STRUCT(']
    for field in fields:
        lines += [f'"{field}" FLOAT DOUBLE PROP(', '"units" "A^2/Hz"', ')']
    lines += [') PROP(', '"key" "inst"', '"master" "bsimcmg"', ')',
              "SWEEP", '"gs" "sweep"', "TRACE", '"m0" "nch_lvt_mac.10"',
              "VALUE"]
    for vgs in vgs_values:
        lines += [f'"gs" {vgs}', '"m0" (',
                  "2e-25", "0", "0", "0", "0", "0", "0", "1e-24", "1.2e-24", ")"]
    lines += ["END", ""]
    return "\n".join(lines)


def _analysis_record(name, analysis, filename, parent, ds):
    return "\n".join([
        f'"{name}" "analysisInst" (', f'"{analysis}"', f'"{filename}"',
        '"PSF"', f'"{parent}"', '("gs")', f'"{analysis} leaf"', ') PROP(',
        '"data_type" "scalar"', '"sweep_tree_type" "leafNode"', f'"ds" {ds}', ')',
    ])


def _write_family_raw(raw, conf, vds_values=(0.0, 0.5), vgs_values=(0.0, 0.5),
                      omit_dc_index=None):
    raw.mkdir(parents=True, exist_ok=True)
    records = []
    # Reverse record order to ensure the log reader orders leaves by ds.
    for index in reversed(range(len(vds_values))):
        vds = vds_values[index]
        dc_name = f"sweepvds-{index:03d}_sweepvgs.dc"
        noise_name = f"sweepvds_noise-{index:03d}_sweepvgs_noise.noise"
        records.append(_analysis_record(
            f"sweepvds-{index:03d}_sweepvgs-dc", "dc", dc_name,
            "sweepvds_sweepvgs-sweep", vds))
        records.append(_analysis_record(
            f"sweepvds_noise-{index:03d}_sweepvgs_noise-noise", "noise", noise_name,
            "sweepvds_noise_sweepvgs_noise-sweep", vds))
        if index != omit_dc_index:
            dc = []
            for vgs in vgs_values:
                dc += [f'"gs" {vgs}', f'"g" {vgs}', f'"d" {vds}', '"b" 0']
                for name in DC:
                    val = ((vgs + vds + 1) if name == "ID"
                           else (1 if name in ("CGG", "CDD", "CSS") else 0.1))
                    dc.append(f'"{conf.spectre_dc_signals[name]}" {val}')
                for name, signal in conf.spectre_parasitic_signals.items():
                    dc.append(f'"{signal}" {0.02 if name == "CJDT" else 0.03}')
                for name, signal in conf.spectre_sat_signals.items():
                    dc.append(f'"{signal}" {0.12 if name == "VDSAT" else 0.15}')
            (raw / dc_name).write_text(_psf(dc))
        (raw / noise_name).write_text(_noise_psf(vgs_values))
    (raw / "logFile").write_text(
        'HEADER\n"PSFversion" "1.00"\nTYPE\nVALUE\n' + "\n\n".join(records) + "\nEND\n")


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


def test_psfascii_ignores_struct_type_properties_from_spectre_25(tmp_path):
    path = tmp_path / "noise.noise"
    path.write_text(_noise_psf([0.3, 0.305]))
    data = spectre.read_psfascii(path)
    assert data["gs"].size == 2
    assert data["m0:flicker"].size == 2
    assert data["m0:therm_sid"].size == 2
    assert "m0:key" not in data
    assert "m0:master" not in data


def test_psf_family_uses_log_order_and_rejects_missing_leaf(tmp_path):
    conf = cfg()
    raw = tmp_path / "good.raw"
    _write_family_raw(raw, conf)
    data = spectre.read_psf_family(
        raw, "sweepvds_sweepvgs-sweep", [0.0, 0.5], [0.0, 0.5])
    np.testing.assert_allclose(data["d"], [0.0, 0.0, 0.5, 0.5])
    np.testing.assert_allclose(data["gs"], [0.0, 0.5, 0.0, 0.5])
    with pytest.raises(spectre.SpectreDataError, match="ds grid differs"):
        spectre.read_psf_family(
            raw, "sweepvds_sweepvgs-sweep", [0.0, 0.25], [0.0, 0.5])
    with pytest.raises(spectre.SpectreDataError, match="gs grid differs"):
        spectre.read_psf_family(
            raw, "sweepvds_sweepvgs-sweep", [0.0, 0.5], [0.0, 0.25])

    broken = tmp_path / "broken.raw"
    _write_family_raw(broken, conf, omit_dc_index=1)
    with pytest.raises(spectre.SpectreDataError, match="missing PSF leaf"):
        spectre.read_psf_family(
            broken, "sweepvds_sweepvgs-sweep", [0.0, 0.5], [0.0, 0.5])


def test_spectre_netlist_rejects_singleton_nested_sweep(tmp_path):
    with pytest.raises(ValueError, match="at least two VGS and two VDS"):
        spectre.make_netlist(cfg(), "TT", 27, 0.016, 0.0,
                             np.array([0.3]), np.array([0.0, 0.5]),
                             tmp_path / "raw")


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
        _write_family_raw(raw, conf)
        (Path(cwd) / "techsweep.log").write_text("spectre completes with 0 errors\n")
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
    assert data["CGDEXT"][0, 0] == 0.03
    assert data["VDSAT"][0, 0] == 0.12
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


def test_spectre_job_reuses_matching_successful_raw_without_marker(tmp_path, monkeypatch):
    conf = cfg()
    monkeypatch.setattr(spectre.shutil, "which", lambda executable: "/bin/spectre")
    monkeypatch.setattr(
        spectre.subprocess, "run",
        lambda *args, **kwargs: pytest.fail("existing successful raw data was rerun"),
    )
    sim_dir = tmp_path / "sim"
    mat_path = tmp_path / "recovered.mat"
    chunk = sim_dir / "recovered_L000_B000"
    raw = chunk / "techsweep.raw"
    chunk.mkdir(parents=True)
    vgs = np.array([0.0, 0.5])
    vds = np.array([0.0, 0.5])
    (chunk / "techsweep.scs").write_text(
        spectre.make_netlist(conf, "TT", 27, 0.016, 0.0, vgs, vds, raw))
    (chunk / "techsweep.log").write_text("spectre completes with 0 errors\n")
    _write_family_raw(raw, conf)

    spectre.run_spectre_job(conf, "TT", 27, [0.016], [0.0], vgs, vds,
                            sim_dir, mat_path)
    assert mat_path.exists()
    assert (chunk / "complete.json").exists()


def test_parser_failure_keeps_simulation_marker_for_retry(tmp_path, monkeypatch):
    conf = cfg()
    monkeypatch.setattr(spectre.shutil, "which", lambda executable: "/bin/spectre")
    calls = []

    def fake_run(argv, cwd, **kwargs):
        calls.append(argv)
        _write_family_raw(Path(cwd) / "techsweep.raw", conf, omit_dc_index=1)
        (Path(cwd) / "techsweep.log").write_text("spectre completes with 0 errors\n")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(spectre.subprocess, "run", fake_run)
    sim_dir = tmp_path / "sim"
    mat_path = tmp_path / "retry.mat"
    vgs = np.array([0.0, 0.5])
    vds = np.array([0.0, 0.5])
    with pytest.raises(spectre.SpectreDataError, match="missing PSF leaf"):
        spectre.run_spectre_job(conf, "TT", 27, [0.016], [0.0], vgs, vds,
                                sim_dir, mat_path)
    chunk = sim_dir / "retry_L000_B000"
    assert (chunk / "complete.json").exists()

    _write_family_raw(chunk / "techsweep.raw", conf)
    spectre.run_spectre_job(conf, "TT", 27, [0.016], [0.0], vgs, vds,
                            sim_dir, mat_path)
    assert len(calls) == 1
    assert mat_path.exists()


def test_pmos_signs_and_explicit_junction_stamp():
    conf = cfg(fet_type="pfet", spectre_cap_junction_mode="add_junction")
    traces = {spec: np.array([0.0]) for spec in conf.spectre_dc_signals.values()}
    traces.update({spec: np.array([0.0])
                   for spec in conf.spectre_parasitic_signals.values()})
    traces.update({spec: np.array([0.0])
                   for spec in conf.spectre_sat_signals.values()})
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
