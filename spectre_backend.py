"""Spectre-only BSIM-CMG sweep backend.

Each Spectre invocation covers one geometry/body-bias slice.  The PSF ASCII
reader accepts scalar traces and the STRUCT records used for noise sources;
unknown or incomplete output is an error, never a partially populated LUT.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import os
import re
import shutil
import subprocess

import numpy as np
from scipy.io import loadmat, savemat

from capacitance import (
    MATRIX9_KEYS, capacitance_metadata, matrix9_to_fields,
    native_to_matrix9, validate_matrix9,
)


CORE = ("ID", "VT", "GM", "GMB", "GDS")
PARASITIC_REQUIRED = ("CJDT", "CJST")
NOISE_REQUIRED = ("STH", "SFL")
_SCALAR = re.compile(r'^"([^"]+)"\s+([^\s()]+)\s*$')
_STRUCT_BEGIN = re.compile(r'^"([^"]+)"\s*\(\s*$')
_TYPE_BEGIN = re.compile(r'^"([^"]+)"\s+STRUCT\s*\(\s*$')
_TRACE = re.compile(r'^"([^"]+)"\s+"?([^"\s()]+)"?')
_FIELD = re.compile(r'^"([^"]+)"\s+')


class SpectreDataError(ValueError):
    pass


def read_psfascii(path: Path) -> dict[str, np.ndarray]:
    """Read numeric VALUE traces, including one-level structured device noise.

    PSF ASCII is version-dependent; remote acceptance must exercise the actual
    Spectre version before a production sweep.  Unsupported records fail closed.
    """
    types: dict[str, list[str]] = {}
    traces: dict[str, str] = {}
    values: dict[str, list[float]] = defaultdict(list)
    section = ""
    current_type = None
    type_depth = 0
    current_struct = None
    struct_values: list[float] = []
    with path.open(encoding="utf-8", errors="replace") as stream:
        for raw_line in stream:
            line = raw_line.strip()
            if not line:
                continue
            if line in {"HEADER", "TYPE", "SWEEP", "TRACE", "VALUE", "END"}:
                section = line
                continue
            if section == "TYPE":
                if current_type is not None:
                    if type_depth == 1:
                        match = _FIELD.match(line)
                        if match:
                            types[current_type].append(match.group(1))
                    type_depth += line.count("(") - line.count(")")
                    if type_depth == 0:
                        current_type = None
                    continue
                match = _TYPE_BEGIN.match(line)
                if match:
                    current_type = match.group(1)
                    types[current_type] = []
                    type_depth = 1
            elif section == "TRACE":
                match = _TRACE.match(line)
                if match:
                    traces[match.group(1)] = match.group(2)
            elif section == "VALUE":
                if current_struct is not None:
                    if line == ")":
                        type_name = traces.get(current_struct)
                        fields = types.get(type_name or "", [])
                        if len(fields) != len(struct_values):
                            raise SpectreDataError(
                                f"{path}: STRUCT {current_struct} has {len(struct_values)} "
                                f"values, expected {len(fields)}")
                        for field, value in zip(fields, struct_values):
                            values[f"{current_struct}:{field}"].append(value)
                        current_struct = None
                        struct_values = []
                    else:
                        try:
                            struct_values.append(float(line))
                        except ValueError as exc:
                            raise SpectreDataError(f"{path}: invalid STRUCT value {line!r}") from exc
                    continue
                match = _STRUCT_BEGIN.match(line)
                if match:
                    current_struct = match.group(1)
                    continue
                match = _SCALAR.match(line)
                if match:
                    try:
                        values[match.group(1)].append(float(match.group(2)))
                    except ValueError:
                        # Non-numeric metadata in VALUE is not a sweep trace.
                        pass
    if current_struct is not None:
        raise SpectreDataError(f"{path}: unterminated STRUCT {current_struct}")
    if not values:
        raise SpectreDataError(f"{path}: no numeric VALUE traces")
    return {name: np.asarray(samples, dtype=float) for name, samples in values.items()}


def _signal(traces: dict[str, np.ndarray], spec, count: int, label: str) -> np.ndarray:
    names = [spec] if isinstance(spec, str) else list(spec)
    for name in names:
        if name in traces:
            arr = traces[name]
            if arr.size != count:
                raise SpectreDataError(f"{label}: {name} has {arr.size} points, expected {count}")
            if not np.all(np.isfinite(arr)):
                raise SpectreDataError(f"{label}: {name} contains nonfinite values")
            return arr
    raise SpectreDataError(f"{label}: none of {names!r} found; available: {list(traces)}")


def _dataset(raw_dir: Path, outer: str, inner: str) -> Path:
    target = f"{outer}_{inner}-sweep"
    matches = [p for p in raw_dir.rglob(target) if p.is_file()]
    if len(matches) != 1:
        raise SpectreDataError(f"{raw_dir}: expected one {target}, found {len(matches)}")
    return matches[0]


def _model_lines(cfg, corner: str) -> list[str]:
    values = {
        "MODEL_LIB": cfg.model_lib,
        "MODEL_INCLUDE": cfg.model_include or "",
        "CORNER": corner,
        "LIB_CORNER": cfg.lib_corner_map[corner],
        "DEVICE": cfg.device,
    }
    return [line.format_map(values) for line in cfg.model_setup_lines]


def _sweep(name: str, analysis: str, start: float, stop: float, step: float) -> str:
    return f"{name} {analysis} param=gs start={start:.12g} stop={stop:.12g} step={step:.12g}"


def make_netlist(cfg, corner: str, temp: int, length: float, vsb: float,
                 vgs: np.ndarray, vds: np.ndarray, raw_dir: Path) -> str:
    if len(vgs) < 2 or len(vds) < 2:
        raise ValueError(
            "Spectre nested DC/noise sweeps require at least two VGS and two VDS points")
    if len(vgs) > 1 and not np.allclose(np.diff(vgs), np.diff(vgs)[0], atol=1e-9):
        raise ValueError("Spectre requires a uniform VGS grid")
    if len(vds) > 1 and not np.allclose(np.diff(vds), np.diff(vds)[0], atol=1e-9):
        raise ValueError("Spectre requires a uniform VDS grid")
    vgs_step = float(vgs[1] - vgs[0]) if len(vgs) > 1 else 1.0
    vds_step = float(vds[1] - vds[0]) if len(vds) > 1 else 1.0
    polarity = -1 if cfg.fet_type == "pfet" else 1
    instance = cfg.instance_template.format_map({
        "D": "d", "G": "g", "S": "0", "B": "b", "DEVICE": cfg.device,
        "LX": length, "W_UM": cfg.w_um, "NFING": cfg.nfing, "NF": cfg.nf,
    })
    lines = [
        "simulator lang=spectre",
        *_model_lines(cfg, corner),
        f"parameters gs=0 ds=0 sb={vsb:.12g}",
        "vnoi (vx 0) vsource dc=0",
        f"vd (d vx) vsource dc={'-' if polarity < 0 else ''}ds",
        f"vg (g 0) vsource dc={'-' if polarity < 0 else ''}gs",
        f"vb (b 0) vsource dc={'-' if polarity > 0 else ''}sb",
        instance,
        f"save {cfg.spectre_instance} d g b",
        f'runOptions options temp={temp} rawfmt=psfascii rawfile="{raw_dir}"',
        f"sweepvds sweep param=ds start={vds[0]:.12g} stop={vds[-1]:.12g} step={vds_step:.12g} {{",
        "  " + _sweep("sweepvgs", "dc", float(vgs[0]), float(vgs[-1]), vgs_step),
        "}",
        f"sweepvds_noise sweep param=ds start={vds[0]:.12g} stop={vds[-1]:.12g} step={vds_step:.12g} {{",
        "  sweepvgs_noise noise freq=1 oprobe=vnoi param=gs "
        f"start={vgs[0]:.12g} stop={vgs[-1]:.12g} step={vgs_step:.12g}",
        "}",
        "",
    ]
    return "\n".join(lines)


def _plane(arr: np.ndarray, n_vgs: int, n_vds: int) -> np.ndarray:
    return arr.reshape(n_vds, n_vgs).T


def _validate_bias_traces(cfg, traces, vgs_vec, vds_vec, vsb):
    """Verify the flattened PSF order before assigning values to LUT axes."""
    count = len(vgs_vec) * len(vds_vec)
    sign = -1 if cfg.fet_type == "pfet" else 1
    expected = {
        "g": np.tile(vgs_vec, len(vds_vec)) * sign,
        "d": np.repeat(vds_vec, len(vgs_vec)) * sign,
        "b": np.full(count, -vsb * sign),
    }
    for name, reference in expected.items():
        measured = _signal(traces, (name, "/" + name), count, f"bias {name}")
        if not np.allclose(measured, reference, rtol=1e-5, atol=1e-6):
            raise SpectreDataError(f"{cfg.key}: PSF {name} bias order differs from requested grid")


def _normalize_dc(cfg, traces: dict[str, np.ndarray], count: int,
                  n_vgs: int, n_vds: int) -> dict[str, np.ndarray]:
    expected = CORE + MATRIX9_KEYS
    missing = [name for name in expected if name not in cfg.spectre_dc_signals]
    if missing:
        raise SpectreDataError(f"{cfg.key}: missing Spectre DC signal mappings: {missing}")
    data = {
        name: _plane(_signal(traces, spec, count, name), n_vgs, n_vds)
        for name, spec in cfg.spectre_dc_signals.items()
    }
    for name, spec in cfg.spectre_parasitic_signals.items():
        data[name] = _plane(_signal(traces, spec, count, name), n_vgs, n_vds)
    for name, spec in cfg.spectre_sat_signals.items():
        data[name] = _plane(_signal(traces, spec, count, name), n_vgs, n_vds)
    if cfg.fet_type == "pfet":
        for name in ("ID", "VT", "IGD", "IGS"):
            if name in data:
                data[name] = -data[name]
        for name in ("VDSAT", "VDSSAT"):
            if name in data:
                data[name] = np.abs(data[name])
    native = {name.lower(): data[name] for name in MATRIX9_KEYS}
    matrix = native_to_matrix9("bsimcmg", native)
    if cfg.spectre_cap_junction_mode == "add_junction":
        matrix[..., 1, 1] += data["CJDT"]
        matrix[..., 2, 2] += data["CJST"]
        validate_matrix9(matrix)
    data.update(matrix9_to_fields(matrix))
    return data


def _existing_valid(path: Path, axes: dict[str, np.ndarray], required: set[str],
                    config_hash: str) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    raw = loadmat(path, simplify_cells=True)
    records = [v for k, v in raw.items() if not k.startswith("_")]
    if len(records) != 1:
        return False
    data = records[0]
    if not required.issubset(data) or not axes.keys() <= data.keys():
        return False
    if str(data.get("JOB_CONFIG_SHA256", "")) != config_hash:
        return False
    for name, axis in axes.items():
        if not np.array_equal(np.atleast_1d(data[name]), axis):
            return False
    shape = tuple(len(axes[k]) for k in ("L", "VGS", "VDS", "VSB"))
    return all(np.asarray(data[name]).size == int(np.prod(shape))
               and np.all(np.isfinite(data[name])) for name in required)


def run_spectre_job(cfg, corner: str, temp: int, l_vec, vsb_vec, vgs_vec,
                    vds_vec, sim_dir: Path, mat_path: Path) -> str:
    if shutil.which("spectre") is None:
        raise RuntimeError("spectre is not on PATH; load the site Spectre module")
    if not cfg.spectre_parasitic_signals or any(
        k not in cfg.spectre_parasitic_signals for k in PARASITIC_REQUIRED
    ):
        raise SpectreDataError(f"{cfg.key}: CJDT and CJST signal mappings are required")
    if not set(NOISE_REQUIRED).issubset(cfg.spectre_noise_signals):
        raise SpectreDataError(f"{cfg.key}: STH and SFL noise mappings are required")
    if "VDSSAT" not in cfg.spectre_sat_signals:
        raise SpectreDataError(f"{cfg.key}: native VDSSAT mapping is required")
    if cfg.spectre_cap_junction_mode not in {"native_total", "add_junction"}:
        raise SpectreDataError(f"{cfg.key}: invalid junction mode")
    missing_dc = set(CORE + MATRIX9_KEYS) - set(cfg.spectre_dc_signals)
    if missing_dc:
        raise SpectreDataError(f"{cfg.key}: missing DC signal mappings: {sorted(missing_dc)}")

    axes = {"L": np.asarray(l_vec), "VGS": np.asarray(vgs_vec),
            "VDS": np.asarray(vds_vec), "VSB": np.abs(np.asarray(vsb_vec))}
    fingerprint = json.dumps({
        "config": asdict(cfg), "corner": corner, "temp": temp,
        "axes": {name: values.tolist() for name, values in axes.items()},
    }, sort_keys=True, default=str)
    config_hash = hashlib.sha256(fingerprint.encode()).hexdigest()
    required = set(CORE + MATRIX9_KEYS + NOISE_REQUIRED + PARASITIC_REQUIRED + ("VDSSAT",))
    if _existing_valid(mat_path, axes, required, config_hash):
        print(f"  [skip] verified Spectre LUT exists: {mat_path}")
        return str(mat_path)
    if mat_path.exists():
        raise SpectreDataError(f"{mat_path}: existing LUT is incomplete or has a different grid")

    n_vgs, n_vds = len(vgs_vec), len(vds_vec)
    count = n_vgs * n_vds
    shape = (len(l_vec), n_vgs, n_vds, len(vsb_vec))
    work_dir = sim_dir / (mat_path.stem + ".work")
    work_dir.mkdir(parents=True, exist_ok=True)
    keys = set(CORE + MATRIX9_KEYS + NOISE_REQUIRED)
    keys.update(cfg.spectre_dc_signals)
    keys.update(cfg.spectre_parasitic_signals)
    keys.update(cfg.spectre_sat_signals)
    arrays = {
        key: np.lib.format.open_memmap(work_dir / f"{key}.npy", mode="w+", dtype="float64", shape=shape)
        for key in sorted(keys)
    }
    succeeded = False
    try:
        for li, length in enumerate(l_vec):
            for bi, vsb in enumerate(vsb_vec):
                chunk_dir = sim_dir / f"{mat_path.stem}_L{li:03d}_B{bi:03d}"
                chunk_dir.mkdir(parents=True, exist_ok=True)
                raw_dir = chunk_dir / "techsweep.raw"
                netlist = chunk_dir / "techsweep.scs"
                content = make_netlist(cfg, corner, temp, length, abs(vsb),
                                       np.asarray(vgs_vec), np.asarray(vds_vec), raw_dir)
                netlist.write_text(content)
                digest = hashlib.sha256(content.encode()).hexdigest()
                marker = chunk_dir / "complete.json"
                cached = False
                if marker.exists():
                    try:
                        cached = json.loads(marker.read_text()).get("netlist_sha256") == digest
                    except (OSError, ValueError):
                        pass
                log_path = chunk_dir / "techsweep.log"
                if not cached:
                    marker.unlink(missing_ok=True)
                    if raw_dir.exists():
                        shutil.rmtree(raw_dir)
                    with log_path.open("w") as log:
                        result = subprocess.run(["spectre", str(netlist)], cwd=chunk_dir,
                                                stdout=log, stderr=subprocess.STDOUT,
                                                env={**os.environ, **cfg.simulation_env},
                                                timeout=cfg.spectre_timeout_s, check=False)
                    if result.returncode:
                        log_tail = "\n".join(
                            log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-30:])
                        raise RuntimeError(
                            f"Spectre failed ({result.returncode}); see {log_path}\n"
                            f"Last 30 log lines:\n{log_tail}")
                dc = read_psfascii(_dataset(raw_dir, "sweepvds", "sweepvgs"))
                noise = read_psfascii(_dataset(raw_dir, "sweepvds_noise", "sweepvgs_noise"))
                _validate_bias_traces(cfg, dc, np.asarray(vgs_vec), np.asarray(vds_vec), abs(vsb))
                data = _normalize_dc(cfg, dc, count, n_vgs, n_vds)
                if "freq" in noise and (noise["freq"].size != count or
                                         not np.allclose(noise["freq"], 1.0, atol=1e-9)):
                    raise SpectreDataError(f"{cfg.key}: noise output is not one 1 Hz point per bias")
                for name, spec in cfg.spectre_noise_signals.items():
                    data[name] = _plane(_signal(noise, spec, count, name), n_vgs, n_vds)
                    if np.any(data[name] < 0):
                        raise SpectreDataError(f"{name} contains negative noise PSD")
                for key in arrays:
                    if key not in data:
                        raise SpectreDataError(f"{cfg.key}: missing output tensor {key}")
                    arrays[key][li, :, :, bi] = data[key]
                marker.write_text(json.dumps({"netlist_sha256": digest}) + "\n")

        sign = "p" if temp >= 0 else "m"
        mat_key = f"{cfg.device.replace('-', '_')}_{corner}_T{sign}{abs(temp)}"
        record = {
            "INFO": f"{cfg.pdk.upper()} PDK — {cfg.device}",
            "CORNER": corner, "TEMP": float(temp),
            **axes, "W": float(cfg.w_um), "NFING": float(cfg.nfing),
            "NF": float(cfg.nf), "NOISE_FREQ_HZ": 1.0,
            "SIMULATOR": "spectre",
            "JOB_CONFIG_SHA256": config_hash,
            "CAPACITANCE_JUNCTION_MODE": cfg.spectre_cap_junction_mode,
            **capacitance_metadata("bsimcmg", cfg.bulk_terminal_alias),
        }
        record.update(arrays)
        temp_path = mat_path.with_name("." + mat_path.name + ".tmp")
        savemat(temp_path, {mat_key: record}, appendmat=False)
        os.replace(temp_path, mat_path)
        print(f"  [save] {mat_path}")
        succeeded = True
        return str(mat_path)
    finally:
        for value in arrays.values():
            value.flush()
        if succeeded:
            shutil.rmtree(work_dir)
