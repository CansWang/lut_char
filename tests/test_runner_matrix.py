import json

import pytest
from scipy.io import loadmat

import run_lut_char_all as runner
from capacitance import MATRIX9_KEYS
from run_lut_char_all import (
    DEVICES,
    DevCfg,
    _build_save_lines,
    generate_netlist,
    load_device_configs,
    parse_and_save,
)


def _cmg_cfg(**overrides):
    values = dict(
        key="demo:nmos",
        device="nmos_demo",
        pdk="demo",
        fet_type="nfet",
        model_lib="/models/demo.lib",
        lib_corner_map={"TT": "tt"},
        l_vec=[0.02],
        vgs_max=1.0,
        vds_max=1.0,
        vsb_vec=[0.0],
        analysis="op",
        save_pfx="@m.xm1.m0",
        caps_model="bsimcmg",
        gmb_col="gmbs",
        id_col="ids",
        has_explicit_u=True,
        instance_template="XM1 {D} {G} {S} {B} {DEVICE} L={LX} W={W_UM}u NFIN={NFING}",
        model_setup_lines=[".lib {MODEL_LIB} {LIB_CORNER}"],
    )
    values.update(overrides)
    return DevCfg(**values)


def test_legacy_save_list_is_unchanged_and_matrix_list_adds_cross_terms():
    cfg = DEVICES["sky130:nfet_01v8"]
    legacy = _build_save_lines(cfg)
    matrix = _build_save_lines(cfg, cap_matrix=True)
    assert f"{cfg.save_pfx}[cdg]" not in legacy
    assert f"{cfg.save_pfx}[cdg]" in matrix
    assert f"{cfg.save_pfx}[csd]" in matrix
    assert f"{cfg.save_pfx}[cgbo]" in matrix


def test_json_device_config_and_generic_netlist(tmp_path):
    record = {
        "devices": [{
            "key": "demo:nmos",
            "device": "nmos_demo",
            "pdk": "demo",
            "fet_type": "nfet",
            "model_family": "bsim-cmg",
            "model_lib": "$HOME/models/demo.lib",
            "lib_corner_map": {"TT": "tt"},
            "l_vec": [0.02],
            "vgs_max": 1.0,
            "vds_max": 1.0,
            "save_pfx": "@m.xm1.m0",
            "has_explicit_u": True,
            "instance_template": "XM1 {D} {G} {S} {B} {DEVICE} L={LX} NFIN={NFING}",
            "model_setup_lines": [".lib {MODEL_LIB} {LIB_CORNER}"],
            "output_aliases": {"gmb": "gmbs", "id": "ids"}
        }]
    }
    config_path = tmp_path / "devices.json"
    config_path.write_text(json.dumps(record))
    registry = load_device_configs([config_path], registry={})
    cfg = registry["demo:nmos"]
    assert cfg.caps_model == "bsimcmg"

    netlist = tmp_path / "probe.spice"
    generate_netlist(cfg, "TT", 27, [0.02], [0.0], str(tmp_path / "out.txt"),
                     str(netlist), vgs_override=[0.5], cap_matrix=True)
    text = netlist.read_text()
    assert "XM1 d g 0 b nmos_demo L={lx} NFIN=1" in text
    assert f".lib {cfg.model_lib} tt" in text
    assert "@m.xm1.m0[cdg]" in text


def test_matrix_parser_writes_nine_fields_and_schema_metadata(tmp_path):
    cfg = _cmg_cfg()
    columns = [
        "ids", "vth", "gm", "gmbs", "gds",
        "cgg", "cgd", "cgs", "cdg", "cdd", "cds", "csg", "csd", "css",
    ]
    header = " ".join(f"{cfg.save_pfx}[{name}]" for name in columns)
    values = " ".join(str(index + 1.0) for index in range(len(columns)))
    txt_path = tmp_path / "raw.txt"
    txt_path.write_text(header + "\n" + values + "\n")
    mat_path = tmp_path / "out.mat"

    data, dims = parse_and_save(
        cfg, str(txt_path), "TT", 27, [0.02], [0.5], [0.0], str(mat_path),
        vds_vec=[0.5], cap_matrix=True,
    )
    assert dims == [1, 1, 1, 1]
    assert all(name in data for name in MATRIX9_KEYS)
    assert "CGB" not in data
    assert data["CGG"].item() == 6.0
    assert data["CGD"].item() == -7.0
    assert data["CDG"].item() == -9.0
    assert data["CAPACITANCE_PROFILE"] == "matrix9"
    assert data["CAPACITANCE_CONVENTION"] == "Cij=dQi/dVj"

    inner = next(value for key, value in loadmat(mat_path, simplify_cells=True).items()
                 if not key.startswith("_"))
    assert inner["CAPACITANCE_PROFILE"] == "matrix9"


def test_smoke_uses_first_available_corner_when_tt_is_absent(tmp_path, monkeypatch):
    cfg = _cmg_cfg(lib_corner_map={"TYP": "typ"})
    jobs = []
    def record(job):
        jobs.append(job)
        return (job[1], job[2], "mock.mat")
    monkeypatch.setattr(runner, "_run_one_pvt", record)
    runner.run_pvt(
        cfg, ["TYP"], [27], smoke_mode=True, max_workers=1,
        base_sim_dir=tmp_path / "sim", base_out_dir=tmp_path / "out",
    )
    assert jobs[0][1] == "TYP"


def test_spectre_smoke_uses_two_adjacent_uniform_vgs_points():
    cfg = _cmg_cfg(simulator="spectre")
    grid = runner.build_uniform_vgs(cfg.vgs_max, 0.005)
    _, smoke_vgs, _ = runner._smoke_grids(cfg, grid)
    assert len(smoke_vgs) == 2
    assert smoke_vgs[1] - smoke_vgs[0] == pytest.approx(0.005)
