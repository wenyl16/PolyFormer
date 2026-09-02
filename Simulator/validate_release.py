"""Validate the small, release-critical subset of PolyFormer artifacts.

This command is deliberately read-only.  It checks the data and result files
needed by the paper's public benchmark cases without importing a solver or
walking the complete results tree.

Examples
--------
python Simulator/validate_release.py
python Simulator/validate_release.py --root /path/to/PolyFormer
"""

from __future__ import annotations

import argparse
import csv
import math
import posixpath
import re
import sys
import zipfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence
from xml.etree import ElementTree as ET


PUBLIC_DS_CASES = (
    ("case10ba_ds.mat", 10),
    ("case17me_ds.mat", 17),
    ("case33bw_ds.mat", 33),
    ("case51ga_ds.mat", 51),
    ("case74_ds.mat", 74),
    ("case118zh_ds.mat", 118),
    ("case136ma_ds.mat", 136),
    ("case533mt_hi_ds.mat", 533),
)

PUBLIC_TS_CASES = (
    ("case4gsts.mat", 4),
    ("case118_ts.mat", 118),
    ("case300_ts.mat", 300),
)

DRCC_CASES = (
    (50, 2, 150),
    (150, 3, 300),
    (300, 5, 900),
    (400, 8, 1280),
)

THREE_PHASE_INPUTS = (
    Path("real_dis_data") / "load_file.xls",
    Path("real_dis_data") / "volt_file.xls",
)

TD_RESULT_BOOK = "td_experiment_results.xlsx"


@dataclass(frozen=True)
class CheckResult:
    category: str
    name: str
    status: str
    detail: str


class ReleaseValidator:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.results: list[CheckResult] = []
        self.np: Any | None = None
        self.loadmat: Any | None = None
        self.whosmat: Any | None = None

    def record(self, category: str, name: str, status: str, detail: str) -> None:
        self.results.append(CheckResult(category, name, status, detail))

    def passed(self, category: str, name: str, detail: str) -> None:
        self.record(category, name, "PASS", detail)

    def failed(self, category: str, name: str, detail: str) -> None:
        self.record(category, name, "FAIL", detail)

    def load_scientific_readers(self) -> bool:
        try:
            import numpy as np
            from scipy.io import loadmat, whosmat
        except Exception as exc:  # includes binary ABI/import failures
            self.failed(
                "runtime",
                "NumPy/SciPy readers",
                f"cannot inspect NPZ/MAT schemas: {type(exc).__name__}: {exc}",
            )
            return False

        self.np = np
        self.loadmat = loadmat
        self.whosmat = whosmat
        self.passed(
            "runtime",
            "NumPy/SciPy readers",
            f"NumPy {np.__version__}; MAT reader available",
        )
        return True

    def check_aggregation(self, scientific_available: bool) -> None:
        building_path = self.root / "data" / "aggregator_data" / "ZH_buildings.csv"
        required_columns = ("HBLD", "CBLD", "PRT")
        try:
            with building_path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                columns = reader.fieldnames or []
                missing = [
                    column for column in required_columns if column not in columns
                ]
                if missing:
                    raise ValueError(f"missing columns: {', '.join(missing)}")
                sampled = 0
                for row in reader:
                    values = [float(row[column]) for column in required_columns]
                    if not all(math.isfinite(value) for value in values):
                        raise ValueError(
                            f"non-finite required value in sampled row {sampled + 2}"
                        )
                    sampled += 1
                    if sampled == 400:
                        break
                if sampled < 400:
                    raise ValueError(
                        f"only {sampled} building rows; at least 400 required"
                    )
            self.passed(
                "aggregation",
                "Zurich building table",
                f"required columns present; first {sampled} required records are finite",
            )
        except Exception as exc:
            self.failed(
                "aggregation",
                "Zurich building table",
                _format_error(building_path, exc),
            )

        if not scientific_available:
            self.failed(
                "aggregation",
                "processed profiles",
                "not checked because NumPy/SciPy readers are unavailable",
            )
            self.failed(
                "aggregation",
                "raw profile MAT files",
                "not checked because NumPy/SciPy readers are unavailable",
            )
        else:
            self._check_profiles_npz()
            self._check_raw_profile_mats()

        artifacts = (
            self.root / "results" / "aggregation" / "pretrainnet_weights.pth",
            self.root / "results" / "aggregation" / "pretrainnet_weights.pthdisc",
        )
        histories = (
            self.root / "results" / "aggregation" / "error_history.pkl",
            self.root / "results" / "aggregation" / "discrete" / "error_history.pkl",
        )
        problems: list[str] = []
        for path in artifacts:
            problem = _torch_archive_problem(path)
            if problem:
                problems.append(problem)
        for path in histories:
            problem = _pickle_problem(path)
            if problem:
                problems.append(problem)
        if problems:
            self.failed("aggregation", "published artifacts", "; ".join(problems))
        else:
            self.passed(
                "aggregation",
                "published artifacts",
                "continuous/discrete weights and error histories are present",
            )

    def _check_profiles_npz(self) -> None:
        assert self.np is not None
        path = self.root / "data" / "profiles_data" / "profiles_data.npz"
        expected = {"load_data", "pv_data", "temp_data"}
        try:
            with self.np.load(path, allow_pickle=False) as archive:
                keys = set(archive.files)
                if keys != expected:
                    raise ValueError(
                        f"keys={sorted(keys)!r}; expected exactly {sorted(expected)!r}"
                    )
                shapes: list[str] = []
                for key in sorted(expected):
                    array = archive[key]
                    if array.shape != (2881,):
                        raise ValueError(f"{key} shape={array.shape}; expected (2881,)")
                    if not self.np.issubdtype(array.dtype, self.np.number):
                        raise ValueError(f"{key} dtype={array.dtype}; expected numeric")
                    if not bool(self.np.isfinite(array).all()):
                        raise ValueError(f"{key} contains NaN/Inf")
                    shapes.append(f"{key}=2881")
            self.passed("aggregation", "processed profiles", ", ".join(shapes))
        except Exception as exc:
            self.failed("aggregation", "processed profiles", _format_error(path, exc))

    def _check_raw_profile_mats(self) -> None:
        assert self.whosmat is not None
        specs = (
            ("BL_samples.mat", "BL_samples", (96, 10), "double"),
            ("PV_samples.mat", "PV_samples", (96, 10), "double"),
            ("CH_2021_real.mat", "temperature", (1, 440), "cell"),
        )
        problems: list[str] = []
        details: list[str] = []
        for filename, variable, shape, matlab_class in specs:
            path = self.root / "data" / "profiles_data" / filename
            try:
                entries = {
                    name: (entry_shape, entry_class)
                    for name, entry_shape, entry_class in self.whosmat(path)
                }
                actual = entries.get(variable)
                if actual != (shape, matlab_class):
                    raise ValueError(
                        f"{variable}={actual!r}; expected {(shape, matlab_class)!r}"
                    )
                details.append(f"{filename}:{shape}")
            except Exception as exc:
                problems.append(_format_error(path, exc))
        if problems:
            self.failed("aggregation", "raw profile MAT files", "; ".join(problems))
        else:
            self.passed("aggregation", "raw profile MAT files", ", ".join(details))

    def check_network_data(self, scientific_available: bool) -> None:
        if not scientific_available:
            self.failed(
                "T-D data",
                "8 public distribution cases",
                "not checked because NumPy/SciPy readers are unavailable",
            )
            self.failed(
                "T-D data",
                "3 public transmission cases",
                "not checked because NumPy/SciPy readers are unavailable",
            )
            return

        ds_root = self.root / "data" / "TD_OPF" / "ds_data"
        for filename, nodes in PUBLIC_DS_CASES:
            self._check_matpower_case(ds_root / filename, nodes, "DS")

        ts_root = self.root / "data" / "TD_OPF"
        for filename, nodes in PUBLIC_TS_CASES:
            self._check_matpower_case(ts_root / filename, nodes, "TS")

    def _check_matpower_case(
        self, path: Path, expected_nodes: int, case_type: str
    ) -> None:
        assert self.np is not None and self.loadmat is not None
        name = path.stem
        try:
            data = self.loadmat(path, simplify_cells=True)
            mpc = data.get("mpc")
            if not isinstance(mpc, dict):
                raise ValueError("top-level 'mpc' MATLAB struct is missing")
            missing = [
                key for key in ("baseMVA", "bus", "branch", "gen") if key not in mpc
            ]
            if missing:
                raise ValueError(f"mpc missing fields: {', '.join(missing)}")
            bus = self.np.asarray(mpc["bus"])
            branch = self.np.asarray(mpc["branch"])
            gen = self.np.asarray(mpc["gen"])
            base_mva = float(self.np.asarray(mpc["baseMVA"]).reshape(-1)[0])
            if bus.ndim != 2 or bus.shape[0] != expected_nodes or bus.shape[1] < 13:
                raise ValueError(
                    f"bus shape={bus.shape}; expected ({expected_nodes}, >=13)"
                )
            if branch.ndim != 2 or branch.shape[1] < 13:
                raise ValueError(
                    f"branch shape={branch.shape}; expected 2-D with >=13 columns"
                )
            if gen.ndim not in (1, 2) or gen.size == 0:
                raise ValueError(
                    f"gen shape={gen.shape}; expected a non-empty vector/matrix"
                )
            if not math.isfinite(base_mva) or base_mva <= 0:
                raise ValueError(f"invalid baseMVA={base_mva}")
            if not all(
                bool(self.np.isfinite(array).all()) for array in (bus, branch, gen)
            ):
                raise ValueError("bus/branch/gen contains NaN/Inf")
            self.passed(
                "T-D data",
                f"{case_type} {name}",
                f"bus={bus.shape}, branch={branch.shape}, gen={gen.shape}",
            )
        except Exception as exc:
            self.failed("T-D data", f"{case_type} {name}", _format_error(path, exc))

    def check_three_phase_inputs(self) -> None:
        data_root = self.root / "data" / "real_dis_data"
        load_path = data_root / "load_file.xls"
        voltage_path = data_root / "volt_file.xls"
        try:
            import xlrd
        except Exception as exc:
            self.failed(
                "three-phase data",
                "XLS reader",
                f"cannot import xlrd: {type(exc).__name__}: {exc}",
            )
            return

        ole_magic = bytes.fromhex("D0CF11E0A1B11AE1")
        load_headers = (
            "management_unit",
            "service_center",
            "customer_id",
            "customer_name",
            "timestamp",
            "asset_id",
            "instantaneous_active_power",
            "phase_a_current",
            "phase_b_current",
            "phase_c_current",
            "neutral_current",
            "phase_a_voltage",
            "phase_b_voltage",
            "phase_c_voltage",
            "total_power_factor",
            "cumulative_forward_active_energy",
            "cumulative_reverse_active_energy",
            "quadrant_i_reactive_energy",
            "quadrant_iv_reactive_energy",
            "ct_ratio",
            "pt_ratio",
            "logical_address",
            "is_supplemental_reading",
            "ingestion_timestamp",
        )
        token_patterns = {
            2: re.compile(r"Customer_ID_\d{3}\Z"),
            3: re.compile(r"Customer_\d{3}\Z"),
            5: re.compile(r"Asset_ID_\d{3}\Z"),
            21: re.compile(r"Logical_Address_\d{3}\Z"),
        }
        expected_time = datetime(2024, 9, 30, 14, 0)

        try:
            if not load_path.is_file():
                raise FileNotFoundError("required anonymized load workbook is missing")
            if _read_prefix(load_path, 8) != ole_magic:
                raise ValueError("not an OLE .xls workbook")
            book = xlrd.open_workbook(load_path)
            expected_sheets = [f"Load_Point_{index:02d}" for index in range(1, 29)]
            if book.sheet_names() != expected_sheets:
                raise ValueError("expected Load_Point_01 through Load_Point_28 in order")

            target_rows = 0
            data_rows = 0
            tokens: dict[int, set[str]] = {column: set() for column in token_patterns}
            for sheet in book.sheets():
                if sheet.ncols != len(load_headers):
                    raise ValueError(f"{sheet.name} has {sheet.ncols} columns; expected 24")
                if tuple(sheet.row_values(0)) != load_headers:
                    raise ValueError(f"{sheet.name} has an unexpected English schema")
                sheet_targets = 0
                for row in range(1, sheet.nrows):
                    values = sheet.row_values(row)
                    data_rows += 1
                    _assert_no_han(values, f"{sheet.name} row {row + 1}")
                    if _normalized(values[0]) and values[0] != "Utility_001":
                        raise ValueError(f"unexpected management-unit token in {sheet.name}")
                    if _normalized(values[1]) and values[1] != "Service_Center_001":
                        raise ValueError(f"unexpected service-center token in {sheet.name}")
                    for column, pattern in token_patterns.items():
                        value = _normalized(values[column])
                        if value and not pattern.fullmatch(value):
                            raise ValueError(
                                f"invalid anonymous token in {sheet.name}, column {column + 1}"
                            )
                        if value:
                            tokens[column].add(value)
                    flag = _normalized(values[22])
                    if flag and flag not in {"Yes", "No"}:
                        raise ValueError(f"unexpected supplemental-reading flag in {sheet.name}")
                    timestamp = _xls_datetime(values[4], book.datemode, xlrd)
                    if timestamp == expected_time:
                        required = [values[column] for column in (7, 8, 9, 11, 12, 13, 14)]
                        if not all(_is_finite_number(value) for value in required):
                            raise ValueError(f"non-numeric target-time load data in {sheet.name}")
                        sheet_targets += 1
                if sheet_targets < 1:
                    raise ValueError(f"target time is missing from {sheet.name}")
                target_rows += sheet_targets

            self.passed(
                "three-phase data",
                load_path.name,
                f"28 sheets, {data_rows} rows, {target_rows} target-time rows; "
                f"anonymous IDs={len(tokens[2])}/{len(tokens[3])}/{len(tokens[5])}/{len(tokens[21])}",
            )
        except Exception as exc:
            self.failed("three-phase data", load_path.name, _format_error(load_path, exc))

        voltage_headers = (
            "timestamp",
            "feeder_active_power",
            "feeder_reactive_power",
            "feeder_current",
            "bus_line_voltage_ab",
        )
        try:
            if not voltage_path.is_file():
                raise FileNotFoundError("required anonymized feeder workbook is missing")
            if _read_prefix(voltage_path, 8) != ole_magic:
                raise ValueError("not an OLE .xls workbook")
            book = xlrd.open_workbook(voltage_path)
            if book.sheet_names() != ["Feeder_Measurements"]:
                raise ValueError("expected one worksheet named Feeder_Measurements")
            sheet = book.sheet_by_index(0)
            if tuple(sheet.row_values(0)) != voltage_headers:
                raise ValueError("feeder workbook has an unexpected English schema")
            target_rows = 0
            for row in range(1, sheet.nrows):
                values = sheet.row_values(row)
                _assert_no_han(values, f"Feeder_Measurements row {row + 1}")
                timestamp = _xls_datetime(values[0], book.datemode, xlrd)
                if timestamp == expected_time:
                    if not _is_finite_number(values[4]):
                        raise ValueError("target-time feeder voltage is non-numeric")
                    target_rows += 1
            if target_rows != 1:
                raise ValueError(f"expected one target-time row; found {target_rows}")
            self.passed(
                "three-phase data",
                voltage_path.name,
                f"1 sheet, {sheet.nrows - 1} rows, target-time row present",
            )
        except Exception as exc:
            self.failed(
                "three-phase data", voltage_path.name, _format_error(voltage_path, exc)
            )

    def check_drcc_data(self) -> None:
        drcc_root = self.root / "data" / "DRCC"
        for assets, groups, samples in DRCC_CASES:
            case_name = f"x{assets}g{groups}s{samples}"
            train_path = drcc_root / f"r_samples_{case_name}.csv"
            test_path = drcc_root / f"r_samples_{case_name}_test.csv"
            try:
                train_detail = _validate_drcc_csv(train_path, assets, groups, samples)
                test_detail = _validate_drcc_csv(test_path, assets, groups, 100)
                self.passed(
                    "DRCC data",
                    case_name,
                    f"train {train_detail}; test {test_detail}",
                )
            except Exception as exc:
                self.failed("DRCC data", case_name, str(exc))

    def check_td_summaries(self) -> None:
        summary_paths = (
            self.root / "results" / "ds_proj" / "td_results" / TD_RESULT_BOOK,
            self.root / "results" / "ds_proj_original" / "td_results" / TD_RESULT_BOOK,
        )
        required_dimensions = {
            "ds_case_parameters": (9, 8),
            "pretrainnet": (28, 16),
            "fullnet": (28, 16),
            "Sheet1": (2, 12),
        }
        expected_result_header = (
            "tscasename",
            "num_ds",
            "dscasename",
            "base_obj",
            "apx_ncons",
            "apx_nvars",
            "apx_obj",
            "apx_peak_memory_MB",
            "apx_time",
            "mean_error",
            "max_error",
            "full_ncons",
            "full_nvars",
            "ipopt_obj",
            "ipopt_peak_memory_MB",
            "ipopt_time",
        )
        for path in summary_paths:
            label = path.relative_to(self.root).as_posix()
            try:
                sheets = _read_xlsx_schema(path)
                for sheet_name, minimum in required_dimensions.items():
                    if sheet_name not in sheets:
                        raise ValueError(f"missing sheet {sheet_name!r}")
                    rows, columns, _ = sheets[sheet_name]
                    if rows < minimum[0] or columns < minimum[1]:
                        raise ValueError(
                            f"sheet {sheet_name!r} shape={(rows, columns)}; "
                            f"expected at least {minimum}"
                        )
                for sheet_name in ("pretrainnet", "fullnet"):
                    header = sheets[sheet_name][2]
                    if tuple(header[:16]) != expected_result_header:
                        raise ValueError(
                            f"sheet {sheet_name!r} has an unexpected header"
                        )
                shape_text = ", ".join(
                    f"{name}={sheets[name][0]}x{sheets[name][1]}"
                    for name in required_dimensions
                )
                self.passed("T-D results", label, shape_text)
            except Exception as exc:
                self.failed("T-D results", label, _format_error(path, exc))

    def check_drcc_results(self) -> None:
        result_root = self.root / "results" / "DRCC"
        for assets, groups, samples in DRCC_CASES:
            case_name = f"x{assets}g{groups}s{samples}"
            case_root = result_root / case_name
            problems: list[str] = []
            total_bytes = 0
            for group in range(groups):
                path = case_root / f"g{group}" / "fullnet_weights.pth"
                problem = _torch_archive_problem(path)
                if problem:
                    problems.append(problem)
                else:
                    total_bytes += path.stat().st_size
            result_pickle = case_root / "test_result.pkl"
            pickle_problem = _pickle_problem(result_pickle)
            if pickle_problem:
                problems.append(pickle_problem)
            if problems:
                self.failed("DRCC results", case_name, "; ".join(problems))
            else:
                self.passed(
                    "DRCC results",
                    case_name,
                    f"{groups}/{groups} group weights ({_human_bytes(total_bytes)}); test_result.pkl present",
                )

    def run(self) -> int:
        if not self.root.is_dir():
            self.failed(
                "release", "repository root", f"directory does not exist: {self.root}"
            )
            self.print_report()
            return 1

        scientific_available = self.load_scientific_readers()
        self.check_aggregation(scientific_available)
        self.check_network_data(scientific_available)
        self.check_three_phase_inputs()
        self.check_drcc_data()
        self.check_td_summaries()
        self.check_drcc_results()
        self.print_report()

        failures = sum(result.status == "FAIL" for result in self.results)
        return int(bool(failures))

    def print_report(self) -> None:
        print("PolyFormer release validation")
        print(f"Root: {self.root}")
        current_category: str | None = None
        for result in self.results:
            if result.category != current_category:
                current_category = result.category
                print(f"\n{current_category}:")
            print(f"  [{result.status}] {result.name}: {result.detail}")

        counts = Counter(result.status for result in self.results)
        print("\nSummary:")
        print(
            "  "
            + ", ".join(
                f"{status}={counts.get(status, 0)}"
                for status in ("PASS", "FAIL")
            )
        )
        failed = counts.get("FAIL", 0) > 0
        if failed:
            print("  Release validation FAILED.")
        else:
            print("  Release validation PASSED.")


def _format_error(path: Path, exc: Exception) -> str:
    return f"{path}: {type(exc).__name__}: {exc}"


def _normalized(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _is_finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(_normalized(value)))
    except (TypeError, ValueError):
        return False


def _assert_no_han(values: Sequence[Any], context: str) -> None:
    for value in values:
        if isinstance(value, str) and re.search(r"[\u3400-\u9fff]", value):
            raise ValueError(f"Chinese text remains in {context}")


def _xls_datetime(value: Any, datemode: int, xlrd_module: Any) -> datetime | None:
    text = _normalized(value)
    if not text:
        return None
    if isinstance(value, (int, float)):
        return xlrd_module.xldate_as_datetime(value, datemode)
    return datetime.fromisoformat(text)


def _read_prefix(path: Path, size: int) -> bytes:
    with path.open("rb") as handle:
        return handle.read(size)


def _torch_archive_problem(path: Path) -> str | None:
    try:
        size = path.stat().st_size
        if size < 1024:
            return f"{path}: unexpectedly small ({size} bytes)"
        magic = _read_prefix(path, 4)
        if magic.startswith(b"PK\x03\x04"):
            with zipfile.ZipFile(path) as archive:
                names = archive.namelist()
                if not any(
                    name.endswith("/data.pkl") or name == "data.pkl" for name in names
                ):
                    return f"{path}: PyTorch ZIP has no data.pkl"
        elif not magic.startswith(b"\x80"):
            return f"{path}: unrecognized PyTorch archive signature"
    except (OSError, zipfile.BadZipFile) as exc:
        return _format_error(path, exc)
    return None


def _pickle_problem(path: Path) -> str | None:
    try:
        size = path.stat().st_size
        if size < 32:
            return f"{path}: missing or unexpectedly small pickle ({size} bytes)"
        if not _read_prefix(path, 1).startswith(b"\x80"):
            return f"{path}: unrecognized binary pickle signature"
    except OSError as exc:
        return _format_error(path, exc)
    return None


def _validate_drcc_csv(path: Path, assets: int, groups: int, samples: int) -> str:
    expected_header = ["group", "mean_r"] + [
        f"sample_{index}" for index in range(1, samples + 1)
    ]
    group_counts = [0] * groups
    row_count = 0
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader, None)
            if header != expected_header:
                actual_columns = len(header) if header is not None else 0
                raise ValueError(
                    f"header has {actual_columns} columns; expected {len(expected_header)} "
                    "with group/mean_r/sample_1...sample_K"
                )
            for line_number, row in enumerate(reader, start=2):
                row_count += 1
                if len(row) != len(expected_header):
                    raise ValueError(
                        f"line {line_number} has {len(row)} fields; expected {len(expected_header)}"
                    )
                group_value = float(row[0])
                group = int(group_value)
                if group_value != group or not 0 <= group < groups:
                    raise ValueError(f"line {line_number} has invalid group {row[0]!r}")
                mean_return = float(row[1])
                returns = [float(value) for value in row[2:]]
                if not math.isfinite(mean_return) or not all(
                    math.isfinite(value) for value in returns
                ):
                    raise ValueError(f"line {line_number} contains NaN/Inf")
                if any(
                    value < -0.1000000001 or value > 0.1000000001 for value in returns
                ):
                    raise ValueError(
                        f"line {line_number} contains a return outside [-0.1, 0.1]"
                    )
                calculated_mean = math.fsum(returns) / samples
                if not math.isclose(
                    mean_return, calculated_mean, rel_tol=0.0, abs_tol=5e-12
                ):
                    raise ValueError(
                        f"line {line_number} mean_r does not match its sample mean"
                    )
                group_counts[group] += 1
    except Exception as exc:
        raise ValueError(_format_error(path, exc)) from exc

    if row_count != assets:
        raise ValueError(f"{path}: rows={row_count}; expected {assets}")
    expected_per_group = assets // groups
    if assets % groups or group_counts != [expected_per_group] * groups:
        raise ValueError(f"{path}: group counts={group_counts}; expected equal groups")
    return f"{row_count}x{len(expected_header)}, groups={group_counts}"


def _read_xlsx_schema(path: Path) -> dict[str, tuple[int, int, list[str | None]]]:
    workbook_ns = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    document_rel_ns = (
        "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    )
    package_rel_ns = "http://schemas.openxmlformats.org/package/2006/relationships"
    with zipfile.ZipFile(path) as archive:
        corrupt_member = archive.testzip()
        if corrupt_member:
            raise ValueError(f"CRC failure in {corrupt_member}")
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        relationships = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        rel_targets = {
            element.attrib["Id"]: element.attrib["Target"]
            for element in relationships.findall(f"{{{package_rel_ns}}}Relationship")
        }
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            shared_root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in shared_root.findall(f"{{{workbook_ns}}}si"):
                shared_strings.append(
                    "".join(
                        node.text or "" for node in item.iter(f"{{{workbook_ns}}}t")
                    )
                )

        output: dict[str, tuple[int, int, list[str | None]]] = {}
        sheets = workbook.find(f"{{{workbook_ns}}}sheets")
        if sheets is None:
            raise ValueError("workbook has no sheets")
        for sheet in sheets:
            name = sheet.attrib["name"]
            relationship_id = sheet.attrib[f"{{{document_rel_ns}}}id"]
            target = rel_targets[relationship_id]
            if target.startswith("/"):
                member = target.lstrip("/")
            elif target.startswith("xl/"):
                member = target
            else:
                member = posixpath.normpath(posixpath.join("xl", target))
            worksheet = ET.fromstring(archive.read(member))
            dimension = worksheet.find(f"{{{workbook_ns}}}dimension")
            rows, columns = _xlsx_dimension(
                dimension.attrib.get("ref", "A1") if dimension is not None else "A1"
            )
            header = _xlsx_first_row(worksheet, workbook_ns, shared_strings, columns)
            output[name] = (rows, columns, header)
    return output


def _xlsx_dimension(reference: str) -> tuple[int, int]:
    last_cell = reference.split(":")[-1].replace("$", "")
    match = re.fullmatch(r"([A-Za-z]+)(\d+)", last_cell)
    if not match:
        raise ValueError(f"invalid worksheet dimension {reference!r}")
    return int(match.group(2)), _column_number(match.group(1))


def _column_number(letters: str) -> int:
    number = 0
    for character in letters.upper():
        number = number * 26 + ord(character) - ord("A") + 1
    return number


def _xlsx_first_row(
    worksheet: ET.Element,
    namespace: str,
    shared_strings: list[str],
    columns: int,
) -> list[str | None]:
    output: list[str | None] = [None] * columns
    sheet_data = worksheet.find(f"{{{namespace}}}sheetData")
    if sheet_data is None:
        return output
    row = next(
        (candidate for candidate in sheet_data if candidate.attrib.get("r") == "1"),
        None,
    )
    if row is None:
        return output
    for cell in row:
        reference = cell.attrib.get("r", "")
        match = re.match(r"([A-Za-z]+)", reference)
        if not match:
            continue
        column = _column_number(match.group(1)) - 1
        cell_type = cell.attrib.get("t")
        value_node = cell.find(f"{{{namespace}}}v")
        if cell_type == "inlineStr":
            inline = cell.find(f"{{{namespace}}}is")
            value = (
                "".join(node.text or "" for node in inline.iter(f"{{{namespace}}}t"))
                if inline is not None
                else ""
            )
        elif value_node is None:
            value = None
        elif cell_type == "s":
            value = shared_strings[int(value_node.text or "0")]
        else:
            value = value_node.text
        if column >= len(output):
            output.extend([None] * (column + 1 - len(output)))
        output[column] = value
    return output


def _human_bytes(value: int) -> str:
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024 or unit == "GiB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{value} B"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="repository root (default: inferred from this script)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    args = build_parser().parse_args(argv)
    return ReleaseValidator(args.root).run()


if __name__ == "__main__":
    raise SystemExit(main())
