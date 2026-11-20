#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CheckGauss.py

Refactored Gaussian output manager with error-aware rerun strategies.

Core behaviors preserved from the user's current workflow:
- scans Gaussian .out files
- ignores jobs that are still running based on joblog state
- for successful jobs, checks frequencies, tags bad TS/CO jobs as =NoTS,
  and runs Gauss_out2in.py for the next step when appropriate
- for failed jobs, rewrites existing .gjf files using the best geometry it can
  recover from the output while preserving non-coordinate metadata already
  present in the input geometry block
- writes jobs_log*.txt and next_gjfs*.txt

Main redesign:
- classify failures before rewriting inputs
- use error-specific recovery plans instead of treating most failures as
  generic SCF issues
- support ONIOM-style charge/multiplicity lines and geometry metadata
  when rewriting .gjf files
"""

from __future__ import annotations

import argparse
import glob
import math
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union


FREQ_RE = re.compile(r"Frequencies\s+--\s+(-?\d+(?:\.\d+)?)")
CONSTRAINED_NAME_RE = re.compile(r".*_(CO|TS)\.out$", re.IGNORECASE)

JOBLOG_TIMEOUT_RE = re.compile(r"The job has run out of time on:", re.IGNORECASE)
JOBLOG_TIMEOUT_CLEANUP_RE = re.compile(
    r"(?:successfully|succesfully)\s+cleaned\s+up\s+before\s+everything\s+was\s+deleted!?",
    re.IGNORECASE,
)
JOBLOG_STARTED_RE = re.compile(r"^The job started on:\s*", re.IGNORECASE)
JOBLOG_FINISHED_RE = re.compile(r"The job finished on:", re.IGNORECASE)
JOBLOG_NEAR_TIMEOUT_RE = re.compile(
    r"nearly\s+timed\s+out\s+so\s+cleaning\s+things\s+up\s+before\s+deletion",
    re.IGNORECASE,
)

CONV_HDR_RE = re.compile(r"^\s*Item\s+Value\s+Threshold\s+Converged\?\s*$", re.IGNORECASE)
CONV_ROW_RE = re.compile(
    r"^\s*(Maximum Force|RMS\s+Force|Maximum Displacement|RMS\s+Displacement)\s+"
    r"([-\d\.]+(?:[DEde][+\-]?\d+)?)\s+"
    r"([-\d\.]+(?:[DEde][+\-]?\d+)?)\s+"
    r"(YES|NO)\s*$",
    re.IGNORECASE,
)

STD_ORIENT_RE = re.compile(r"^\s*(Standard orientation:|Input orientation:)\s*$", re.IGNORECASE)
DASH_RE = re.compile(r"^\s*-{5,}\s*$")

TS_SP_OUT_RE = re.compile(r".*_TS_sp\.out$", re.IGNORECASE)
TS_OUT_RE = re.compile(r".*_TS\.out$", re.IGNORECASE)
CO_OUT_RE = re.compile(r".*_CO\.out$", re.IGNORECASE)
GENERIC_SP_OUT_RE = re.compile(r".*_sp\.out$", re.IGNORECASE)
SMD_OUT_RE = re.compile(r".*\.smd\.out$", re.IGNORECASE)
ORCA_NORMAL_TERMINATION_RE = re.compile(r"\*{4}\s*ORCA TERMINATED NORMALLY\s*\*{4}", re.IGNORECASE)

INTEGER_LINE_RE = re.compile(r"^\s*[+-]?\d+(?:\s+[+-]?\d+)+\s*$")
GALLOC_RE = re.compile(r"galloc:\s*could\s+not\s+allocate\s+memory", re.IGNORECASE)

OPT_STEP_RE_1 = re.compile(r"^\s*Step number\s+(\d+)\b", re.IGNORECASE)
OPT_STEP_RE_2 = re.compile(r"^\s*Optimization\s+Cycle\s*:\s*(\d+)\b", re.IGNORECASE)
OPT_STEP_RE_3 = re.compile(r"^\s*Cycle\s+(\d+)\s+out\s+of\b", re.IGNORECASE)

MODRED_FREEZE_RE = re.compile(
    r"^\s*(?:"
    r"B\s+-?\d+\s+-?\d+\s+F"
    r"|A\s+-?\d+\s+-?\d+\s+-?\d+\s+F"
    r"|D\s+-?\d+\s+-?\d+\s+-?\d+\s+-?\d+\s+F"
    r")\s*$",
    re.IGNORECASE,
)
MODRED_BOND_FREEZE_CAPTURE_RE = re.compile(r"^\s*B\s+(\d+)\s+(\d+)\s+F\b", re.IGNORECASE)
BOND_PAIR_IN_TITLE_RE = re.compile(r"(\d+)\s*-\s*(\d+)")
NUMERIC_TOKEN_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[DEde][+-]?\d+)?$")

FAIL_SCF_RE = re.compile(
    r"(?:Convergence failure -- run terminated|SCF has not converged|"
    r"Error termination via .*l502(?:\.exe)?|Error termination via .*l508(?:\.exe)?)",
    re.IGNORECASE,
)
FAIL_OPT_MAX_RE = re.compile(
    r"(?:Error termination request processed by link 9999|"
    r"Optimization stopped\.|Number of steps exceeded)",
    re.IGNORECASE,
)
FAIL_INTERNAL_RE = re.compile(
    r"(?:Linear angle in Bend|Linear angle in Tors|FormBX had a problem|"
    r"Error in internal coordinate system|New curvilinear step not converged|"
    r"Error imposing constraints)",
    re.IGNORECASE,
)
FAIL_DISK_IO_RE = re.compile(
    r"(?:Erroneous write|write error|g_write|writwa|No space left on device|"
    r"Bad length for file|FileIO operation on non-existent file|open-new-file)",
    re.IGNORECASE,
)
FAIL_INPUT_RE = re.compile(
    r"(?:End of file in ZSymb|Route card not found|Wanted an integer as input|"
    r"There are no atoms in this input structure|Unrecognized atomic symbol)",
    re.IGNORECASE,
)
FAIL_CHECKPOINT_RE = re.compile(
    r"(?:ntrex1|NtrErr|Error opening checkpoint file|Unable to open checkpoint file|"
    r"Unable to open file.*\.chk|open-new-file)",
    re.IGNORECASE,
)
FAIL_MEMORY_WORDS_RE = re.compile(
    r"(?:words are not enough for|not enough memory available|needs more words of memory)",
    re.IGNORECASE,
)

ELEMENTS = [
    "X",
    "H", "He",
    "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
    "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr",
    "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "In", "Sn", "Sb", "Te", "I", "Xe",
    "Cs", "Ba",
    "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn",
    "Fr", "Ra",
    "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr",
    "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn",
    "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
]


def collapse_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def to_float(value: str) -> float:
    return float(value.replace("D", "E").replace("d", "E"))


def parse_csv_terms(value: str) -> List[str]:
    return [term.strip() for term in (value or "").split(",") if term.strip()]


def parse_stage_terms(values: Sequence[str], flag: str = "--orca-for") -> Set[str]:
    """Shared TS/SP/ALL stage selector for --orca-for and --gauss-template-for.

    `flag` only names the option in the error text; both selectors must accept
    exactly the same vocabulary or the two handoffs stop being comparable.
    """
    selected: Set[str] = set()
    for raw in values:
        for term in parse_csv_terms(raw):
            key = term.strip().lower().replace("-", "_")
            if key in {"all", "both"}:
                selected.update({"TS", "SP"})
            elif key in {"ts", "transitionstate", "transition_state"}:
                selected.add("TS")
            elif key in {"sp", "singlepoint", "single_point", "single-point", "ts_sp"}:
                selected.add("SP")
            else:
                raise ValueError(
                    f"Unknown {flag} stage '{term}'. Use TS, SP, or ALL."
                )
    return selected


def route_term_key(token: str) -> str:
    cleaned = token.strip()
    return cleaned.split("=", 1)[0].strip().lower() if "=" in cleaned else cleaned.lower()


def route_term_conflict_group(keyword: str, token: str) -> Optional[str]:
    key = route_term_key(token)
    keyword_lower = keyword.lower()
    if keyword_lower == "scf" and key in {"xqc", "yqc"}:
        return "scf-qc-rescue"
    if keyword_lower == "opt" and key in {"gdiis", "maxstep"}:
        return "opt-gdiis-maxstep"
    return None


def numbered_path(base_name: str) -> Path:
    path = Path(base_name)
    if not path.exists():
        return path
    stem, suffix = path.stem, path.suffix
    idx = 2
    while True:
        candidate = Path(f"{stem}{idx}{suffix}")
        if not candidate.exists():
            return candidate
        idx += 1


def clamp_int(value: int, minimum: int = 1) -> int:
    return max(minimum, value)


class TeeLogger:
    def __init__(self, path: Path):
        self.path = path
        self.fh = path.open("w", encoding="utf-8")

    def log(self, message: str = "") -> None:
        print(message)
        self.fh.write(message + "\n")
        self.fh.flush()

    def close(self) -> None:
        try:
            self.fh.close()
        except Exception:
            pass


@dataclass
class ConvergenceBlock:
    start_line: int
    opt_step: Optional[int]
    rows: Dict[str, Tuple[float, float, str]]

    @property
    def all_yes(self) -> bool:
        return bool(self.rows) and all(v[2].upper() == "YES" for v in self.rows.values())

    @property
    def sse_exceedance(self) -> float:
        score = 0.0
        for value, threshold, _yn in self.rows.values():
            if threshold == 0:
                return float("inf")
            ratio = abs(value) / abs(threshold)
            score += max(0.0, ratio - 1.0) ** 2
        return score


@dataclass
class ImproveConfig:
    stage1_opt_add: List[str]
    stage1_scf: str
    stage2_opt_add: List[str]
    stage2_scf: str
    use_stage2: bool
    opt_limit_add: List[str]
    internal_opt_add: List[str]
    internal_opt_add_modred: List[str]
    mem_shortage_scale: float


@dataclass
class JoblogStatus:
    running: bool = False
    timed_out: bool = False


@dataclass
class FailureInfo:
    kind: str
    label: str
    detail: str
    auto_retry: bool


@dataclass
class Link0Adjustments:
    mem_scale: Optional[float] = None


@dataclass
class RepairPlan:
    label: str
    edit_mode: str
    use_closest_block: bool = False
    use_last_orientation: bool = True
    route_opt_add: Optional[List[str]] = None
    route_scf: Optional[str] = None
    link0: Optional[Link0Adjustments] = None
    allow_no_geometry: bool = False
    manual_only: bool = False


@dataclass
class GeometryLineTemplate:
    label: str
    prefix_tokens: List[str]
    suffix_tokens: List[str]


@dataclass
class FailureCase:
    out_path: Path
    gjf_path: Optional[Path]
    failure: FailureInfo
    coords: Optional[List[Tuple[int, float, float, float]]]
    geometry_note: str
    suggestion_note: str
    suggested_option_codes: List[str]
    route_text: Optional[str]
    editable: bool = True


@dataclass(frozen=True)
class MenuOption:
    code: str
    label: str
    kind: str
    value: Optional[str] = None
    prompt: Optional[str] = None
    requires_int: bool = False


@dataclass(frozen=True)
class RemovalItem:
    kind: str
    remove_key: str
    display: str
    exact_value: str


@dataclass
class RouteChoice:
    opt_additions: List[str]
    scf_additions: List[str]
    raw_additions: List[str]
    removals: List[RemovalItem]


MENU_OPTIONS: List[MenuOption] = [
    MenuOption("0", "custom route fragment(s)", "custom"),
    MenuOption("1", "opt=loose", "opt", "loose"),
    MenuOption("2", "opt=gdiis", "opt", "gdiis"),
    MenuOption("3", "opt=recalcfc=x", "opt", "recalcfc", "Value for recalcfc", True),
    MenuOption("4", "opt=calcfc", "opt", "calcfc"),
    MenuOption("5", "opt=calcall", "opt", "calcall"),
    MenuOption("6", "opt=cartesian", "opt", "cartesian"),
    MenuOption("7", "scf=xqc", "scf", "xqc"),
    MenuOption("8", "scf=yqc", "scf", "yqc"),
    MenuOption("9", "opt=maxcycle=x", "opt", "maxcycle", "Value for opt maxcycle", True),
]


def collect_out_files(inputs: Sequence[str], recursive: bool) -> List[Path]:
    files: List[Path] = []

    def add(path: Path) -> None:
        if path.is_file() and path.suffix.lower() == ".out" and not SMD_OUT_RE.match(path.name):
            files.append(path.resolve())

    for item in inputs:
        path = Path(item)
        if path.exists() and path.is_dir():
            iterator = path.rglob("*.out") if recursive else path.glob("*.out")
            for sub in iterator:
                add(sub)
            continue

        matches = glob.glob(item, recursive=recursive)
        if matches:
            for match in matches:
                subpath = Path(match)
                if subpath.is_dir():
                    iterator = subpath.rglob("*.out") if recursive else subpath.glob("*.out")
                    for sub in iterator:
                        add(sub)
                else:
                    add(subpath)
        elif path.exists():
            add(path)

    return sorted({entry for entry in files})


def _family_stage_info(out_path: Path) -> Tuple[Tuple[str, str, str], int]:
    name = out_path.name
    stem = out_path.stem
    parent_key = str(out_path.parent.resolve())

    match = re.match(r"^(.*)_TS_sp$", stem, flags=re.IGNORECASE)
    if match and TS_SP_OUT_RE.match(name):
        return (parent_key, "constrained", match.group(1)), 3

    match = re.match(r"^(.*)_TS$", stem, flags=re.IGNORECASE)
    if match and TS_OUT_RE.match(name):
        return (parent_key, "constrained", match.group(1)), 2

    match = re.match(r"^(.*)_CO$", stem, flags=re.IGNORECASE)
    if match and CO_OUT_RE.match(name):
        return (parent_key, "constrained", match.group(1)), 1

    match = re.match(r"^(.*)_sp$", stem, flags=re.IGNORECASE)
    if match and GENERIC_SP_OUT_RE.match(name):
        return (parent_key, "unconstrained", match.group(1)), 1

    return (parent_key, "unconstrained", stem), 0


def prefilter_latest_stage_outs(files: Sequence[Path]) -> List[Path]:
    best: Dict[Tuple[str, str, str], Tuple[int, float, Path]] = {}
    for path in files:
        key, rank = _family_stage_info(path)
        try:
            mtime = path.stat().st_mtime
        except OSError:
            mtime = -1.0
        prev = best.get(key)
        if prev is None or rank > prev[0] or (rank == prev[0] and mtime > prev[1]):
            best[key] = (rank, mtime, path)
    return sorted(value[2] for value in best.values())


def tail_lines(path: Path, n: int) -> List[str]:
    if n <= 0:
        return []
    block = 4096
    data = b""
    with path.open("rb") as fh:
        fh.seek(0, 2)
        pos = fh.tell()
        while pos > 0 and data.count(b"\n") <= n:
            step = min(block, pos)
            pos -= step
            fh.seek(pos, 0)
            data = fh.read(step) + data
    return data.decode("utf-8", errors="replace").splitlines()[-n:]


def has_normal_termination(path: Path, tail_n: int) -> bool:
    try:
        last = tail_lines(path, tail_n)
    except OSError:
        return False
    return any(
        "Normal termination" in line or ORCA_NORMAL_TERMINATION_RE.search(line)
        for line in last
    )


def tail_has_galloc(out_path: Path, n: int = 250) -> bool:
    try:
        text = "\n".join(tail_lines(out_path, n))
    except OSError:
        return False
    return bool(GALLOC_RE.search(text))


def _jobid_from_joblog_name(path: Path) -> Optional[int]:
    parts = path.name.split(".")
    if len(parts) < 3:
        return None
    tail = parts[-1]
    return int(tail) if tail.isdigit() else None


def find_latest_joblog_by_id(out_path: Path, extra_dirs: Sequence[Path]) -> Optional[Path]:
    stem = out_path.stem
    patterns = [f"joblog.{stem}.*"]
    search_dirs = [out_path.parent, out_path.parent / "logs", Path.cwd(), *extra_dirs]
    candidates: List[Path] = []
    for directory in search_dirs:
        if not directory.exists() or not directory.is_dir():
            continue
        for pattern in patterns:
            candidates.extend([entry.resolve() for entry in directory.glob(pattern) if entry.is_file()])
    if not candidates:
        return None

    numeric: List[Tuple[int, Path]] = []
    nonnumeric: List[Path] = []
    for candidate in candidates:
        jobid = _jobid_from_joblog_name(candidate)
        if jobid is None:
            nonnumeric.append(candidate)
        else:
            numeric.append((jobid, candidate))

    if numeric:
        numeric.sort(key=lambda item: item[0], reverse=True)
        return numeric[0][1]

    nonnumeric.sort(key=lambda item: item.stat().st_mtime, reverse=True)
    return nonnumeric[0]


def read_joblog_status(joblog: Path) -> JoblogStatus:
    try:
        text = joblog.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return JoblogStatus()

    if JOBLOG_TIMEOUT_RE.search(text):
        return JoblogStatus(running=False, timed_out=True)
    if JOBLOG_TIMEOUT_CLEANUP_RE.search(text):
        return JoblogStatus(running=False, timed_out=True)
    if JOBLOG_NEAR_TIMEOUT_RE.search(text):
        return JoblogStatus(running=False, timed_out=True)

    has_finished = bool(JOBLOG_FINISHED_RE.search(text))
    lines = [line.rstrip() for line in text.splitlines()]
    nonempty = [line for line in lines if line.strip()]
    if not nonempty:
        return JoblogStatus()
    if has_finished:
        return JoblogStatus(running=False, timed_out=False)
    if JOBLOG_STARTED_RE.match(nonempty[-1]):
        return JoblogStatus(running=True, timed_out=False)
    return JoblogStatus()


def read_tail_text(out_path: Path, n: int = 400) -> str:
    try:
        return "\n".join(tail_lines(out_path, n))
    except OSError:
        return ""


def find_first_frequency(out_path: Path) -> Optional[float]:
    with out_path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            match = FREQ_RE.search(line)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    return None
    return None


def needs_constrained_freq_check(out_path: Path) -> bool:
    return bool(CONSTRAINED_NAME_RE.match(out_path.name))


def opt_step_update(line: str, current: Optional[int]) -> Optional[int]:
    for regex in (OPT_STEP_RE_1, OPT_STEP_RE_2, OPT_STEP_RE_3):
        match = regex.match(line)
        if match:
            try:
                return int(match.group(1))
            except Exception:
                return current
    return current


def parse_convergence_blocks(out_path: Path) -> List[ConvergenceBlock]:
    blocks: List[ConvergenceBlock] = []
    current_step: Optional[int] = None
    with out_path.open("r", encoding="utf-8", errors="replace") as fh:
        iterator = enumerate(fh, start=1)
        for lineno, line in iterator:
            current_step = opt_step_update(line, current_step)
            if not CONV_HDR_RE.match(line):
                continue
            block_step = current_step
            rows: Dict[str, Tuple[float, float, str]] = {}
            for _ in range(12):
                try:
                    _, next_line = next(iterator)
                except StopIteration:
                    break
                current_step = opt_step_update(next_line, current_step)
                match = CONV_ROW_RE.match(next_line)
                if not match:
                    break
                name = " ".join(match.group(1).split())
                rows[name] = (
                    to_float(match.group(2)),
                    to_float(match.group(3)),
                    match.group(4).upper(),
                )
            if rows:
                blocks.append(ConvergenceBlock(start_line=lineno, opt_step=block_step, rows=rows))
    return blocks


def pick_closest_block(blocks: Sequence[ConvergenceBlock]) -> Optional[ConvergenceBlock]:
    if not blocks:
        return None
    return min(blocks, key=lambda block: (block.sse_exceedance, block.start_line))


def parse_orientation_table(lines: Sequence[str], start_idx: int) -> Optional[List[Tuple[int, float, float, float]]]:
    index = start_idx + 1
    dashes = 0
    while index < len(lines) and dashes < 2:
        if DASH_RE.match(lines[index].strip()):
            dashes += 1
        index += 1
    if dashes < 2:
        return None

    coords: List[Tuple[int, float, float, float]] = []
    while index < len(lines):
        stripped = lines[index].strip()
        if DASH_RE.match(stripped):
            break
        parts = stripped.split()
        if len(parts) >= 6 and parts[0].isdigit():
            try:
                atomic_number = int(parts[1])
                x_coord = float(parts[3])
                y_coord = float(parts[4])
                z_coord = float(parts[5])
                coords.append((atomic_number, x_coord, y_coord, z_coord))
            except Exception:
                pass
        index += 1
    return coords if coords else None


def extract_orientation_after_line(out_path: Path, from_line: int) -> Optional[List[Tuple[int, float, float, float]]]:
    lines = out_path.read_text(encoding="utf-8", errors="replace").splitlines(True)
    index = max(0, from_line - 1)
    while index < len(lines):
        if STD_ORIENT_RE.match(lines[index].rstrip("\n")):
            coords = parse_orientation_table(lines, index)
            if coords:
                return coords
        index += 1
    return None


def extract_orientation_before_line(out_path: Path, from_line: int) -> Optional[List[Tuple[int, float, float, float]]]:
    lines = out_path.read_text(encoding="utf-8", errors="replace").splitlines(True)
    last: Optional[List[Tuple[int, float, float, float]]] = None
    stop = min(len(lines), max(0, from_line - 1))
    for index in range(stop):
        if STD_ORIENT_RE.match(lines[index].rstrip("\n")):
            coords = parse_orientation_table(lines, index)
            if coords:
                last = coords
    return last


def extract_last_orientation(out_path: Path) -> Optional[List[Tuple[int, float, float, float]]]:
    lines = out_path.read_text(encoding="utf-8", errors="replace").splitlines(True)
    last: Optional[List[Tuple[int, float, float, float]]] = None
    for idx, line in enumerate(lines):
        if STD_ORIENT_RE.match(line.rstrip("\n")):
            coords = parse_orientation_table(lines, idx)
            if coords:
                last = coords
    return last


def remove_opt_from_route(route: str) -> str:
    route = re.sub(r"(?i)\bopt\s*=\s*\([^)]*\)", " ", route)
    route = re.sub(r"(?i)\bopt\s*=\s*[^\s]+", " ", route)
    route = re.sub(r"(?i)\bopt\b", " ", route)
    return collapse_spaces(route)


def merge_opt_terms(existing: Sequence[str], additions: Sequence[str], keyword: str = "opt") -> List[str]:
    out = [token.strip() for token in existing if token.strip()]

    def rebuild_indices() -> Dict[str, int]:
        return {route_term_key(token): idx for idx, token in enumerate(out)}

    indices = rebuild_indices()
    for addition in additions:
        addition = addition.strip()
        if not addition:
            continue
        add_key = route_term_key(addition)
        add_group = route_term_conflict_group(keyword, addition)
        if add_group is not None:
            out = [
                token
                for token in out
                if not (
                    route_term_conflict_group(keyword, token) == add_group
                    and route_term_key(token) != add_key
                )
            ]
            indices = rebuild_indices()

        if add_key in indices:
            out[indices[add_key]] = addition
        else:
            indices[add_key] = len(out)
            out.append(addition)
    return out


def ensure_opt_terms(route: str, additions: Sequence[str]) -> str:
    return ensure_keyword_terms(route, "opt", additions)


def set_scf(route: str, scf_value: str) -> str:
    scf_value = scf_value.strip()
    if not scf_value:
        return collapse_spaces(route)
    if scf_value.startswith("(") and scf_value.endswith(")"):
        scf_value = scf_value[1:-1]
    return ensure_keyword_terms(route, "scf", split_comma_terms(scf_value))


def route_matches_stage(route: str, opt_add: Sequence[str], scf_value: str) -> bool:
    scf_value = scf_value.strip().lower()
    if scf_value:
        match = re.search(r"(?i)\bscf\s*=\s*(\([^)]*\)|[^\s]+)", route)
        if not match or scf_value not in match.group(1).lower():
            return False

    match = re.search(r"(?i)\bopt\s*=\s*(\([^)]*\)|[^\s]+)", route)
    if not match:
        return False
    inside = match.group(1)
    if inside.startswith("(") and inside.endswith(")"):
        inside = inside[1:-1]
    existing = [term.strip() for term in inside.split(",") if term.strip()]
    keys = {
        term.split("=", 1)[0].strip().lower() if "=" in term else term.lower()
        for term in existing
    }
    for addition in opt_add:
        addition = addition.strip()
        if not addition:
            continue
        key = addition.split("=", 1)[0].strip().lower() if "=" in addition else addition.lower()
        if key not in keys:
            return False
    return True


def route_has_scf_value(route: str, scf_value: str) -> bool:
    scf_value = scf_value.strip().lower()
    if not scf_value:
        return False
    match = re.search(r"(?i)\bscf\s*=\s*(\([^)]*\)|[^\s]+)", route)
    if not match:
        return False
    value = match.group(1).strip().lower()
    if value.startswith("(") and value.endswith(")"):
        value = value[1:-1]
    return scf_value in value


def apply_stage_improvement(route: str, cfg: ImproveConfig) -> Tuple[str, str]:
    if cfg.use_stage2 and route_matches_stage(route, cfg.stage1_opt_add, cfg.stage1_scf):
        opt_add, scf_value, label = cfg.stage2_opt_add, cfg.stage2_scf, "stage2"
    else:
        opt_add, scf_value, label = cfg.stage1_opt_add, cfg.stage1_scf, "stage1"
    route = ensure_opt_terms(route, opt_add)
    route = set_scf(route, scf_value)
    return route, label


def apply_scf_only_escalation(route: str, cfg: ImproveConfig) -> Tuple[str, str]:
    if cfg.use_stage2 and route_has_scf_value(route, cfg.stage1_scf):
        return set_scf(route, cfg.stage2_scf), "stage2"
    return set_scf(route, cfg.stage1_scf), "stage1"


def apply_specific_opt_add(route: str, opt_add: Sequence[str], scf_value: str = "") -> str:
    route = ensure_opt_terms(route, opt_add)
    if scf_value.strip():
        route = set_scf(route, scf_value)
    return route


def is_modred_freeze_line(line: str) -> bool:
    return bool(MODRED_FREEZE_RE.match(line.rstrip("\n")))


def strip_freq_only_constraints_after_geometry(
    lines: List[str],
    geom_start_idx: int,
    natoms: int,
) -> Tuple[List[str], int, List[Tuple[int, int]]]:
    if natoms <= 0:
        return lines, 0, []

    geom_blank_idx = geom_start_idx + natoms
    if geom_blank_idx >= len(lines):
        return lines, 0, []

    index = geom_blank_idx
    while index < len(lines) and lines[index].strip() == "":
        index += 1
    if index >= len(lines):
        return lines, 0, []

    block_start = index
    block_end = index
    while block_end < len(lines) and lines[block_end].strip() != "":
        block_end += 1

    block = lines[block_start:block_end]
    if not block or not all(is_modred_freeze_line(line) for line in block):
        return lines, 0, []

    pairs: List[Tuple[int, int]] = []
    seen = set()
    for line in block:
        match = MODRED_BOND_FREEZE_CAPTURE_RE.match(line)
        if match:
            a_idx = int(match.group(1))
            b_idx = int(match.group(2))
            key = (a_idx, b_idx) if a_idx <= b_idx else (b_idx, a_idx)
            if key not in seen:
                seen.add(key)
                pairs.append((a_idx, b_idx))

    removed = len(block)
    del lines[block_start:block_end]

    index = geom_blank_idx
    if index >= len(lines):
        lines.append("\n")
    elif lines[index].strip() != "":
        lines.insert(index, "\n")

    index += 1
    while index < len(lines) and lines[index].strip() == "":
        del lines[index]

    return lines, removed, pairs


def format_bond_pairs_line(pairs: Sequence[Tuple[int, int]]) -> str:
    return ",".join(f"{a_idx}-{b_idx}" for a_idx, b_idx in pairs)


def upsert_constraints_line_into_title(
    lines: List[str],
    route_end_idx: int,
    constraint_line: str,
) -> Tuple[List[str], bool]:
    index = route_end_idx + 1
    while index < len(lines) and lines[index].strip() == "":
        index += 1
    title_start = index
    title_end = title_start
    while title_end < len(lines) and lines[title_end].strip() != "":
        title_end += 1

    for idx in range(title_start, title_end):
        if BOND_PAIR_IN_TITLE_RE.search(lines[idx]):
            lines[idx] = constraint_line + "\n"
            return lines, True

    lines.insert(title_end, constraint_line + "\n")
    return lines, True


def parse_bond_pairs_from_gjf_title(gjf_path: Path) -> List[Tuple[int, int]]:
    try:
        lines = gjf_path.read_text(encoding="utf-8", errors="replace").splitlines(True)
    except OSError:
        return []

    route_start = next((idx for idx, line in enumerate(lines) if line.lstrip().startswith("#")), None)
    if route_start is None:
        return []
    route_end = route_start
    while route_end < len(lines) and lines[route_end].strip() != "":
        route_end += 1

    index = route_end + 1
    while index < len(lines) and lines[index].strip() == "":
        index += 1
    title_start = index
    title_end = title_start
    while title_end < len(lines) and lines[title_end].strip() != "":
        title_end += 1

    pairs: List[Tuple[int, int]] = []
    seen = set()
    for line in lines[title_start:title_end]:
        for match in BOND_PAIR_IN_TITLE_RE.finditer(line):
            a_idx = int(match.group(1))
            b_idx = int(match.group(2))
            key = (a_idx, b_idx) if a_idx <= b_idx else (b_idx, a_idx)
            if key not in seen:
                seen.add(key)
                pairs.append((a_idx, b_idx))
    return pairs


def best_matching_gjf_for_co_out(co_out: Path) -> Optional[Path]:
    candidates: List[Path] = []
    direct = co_out.with_suffix(".gjf")
    if direct.exists():
        candidates.append(direct)

    stripped_stem = co_out.stem.split("=", 1)[0]
    stripped = co_out.with_name(stripped_stem + ".gjf")
    if stripped.exists():
        candidates.append(stripped)

    for candidate in co_out.parent.glob(stripped_stem + "*.gjf"):
        if candidate.is_file():
            candidates.append(candidate)

    if not candidates:
        return None
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0]


def corresponding_co_out_for_ts(ts_out: Path) -> Optional[Path]:
    match = re.match(r"^(.*)_TS$", ts_out.stem, flags=re.IGNORECASE)
    if not match:
        return None
    base = match.group(1)
    exact = ts_out.with_name(base + "_CO" + ts_out.suffix)
    if exact.exists():
        return exact
    candidates = list(ts_out.parent.glob(base + "_CO*.out"))
    if not candidates:
        return None
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0]


def parse_modred_bond_freeze_pairs(co_out: Path) -> List[Tuple[int, int]]:
    pairs: List[Tuple[int, int]] = []
    seen = set()
    in_modred = False
    try:
        with co_out.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if "ModRedundant input section has been read" in line:
                    in_modred = True
                    continue
                if not in_modred:
                    continue
                match = MODRED_BOND_FREEZE_CAPTURE_RE.match(line)
                if match:
                    a_idx = int(match.group(1))
                    b_idx = int(match.group(2))
                    key = (a_idx, b_idx) if a_idx <= b_idx else (b_idx, a_idx)
                    if key not in seen:
                        seen.add(key)
                        pairs.append((a_idx, b_idx))
                    continue
                if line.lstrip().startswith("ITRead=") or line.lstrip().startswith("MicOpt="):
                    break
    except OSError:
        return []

    if pairs:
        return pairs

    try:
        with co_out.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                match = MODRED_BOND_FREEZE_CAPTURE_RE.match(line)
                if not match:
                    continue
                a_idx = int(match.group(1))
                b_idx = int(match.group(2))
                key = (a_idx, b_idx) if a_idx <= b_idx else (b_idx, a_idx)
                if key not in seen:
                    seen.add(key)
                    pairs.append((a_idx, b_idx))
    except OSError:
        pass

    return pairs


def bond_distance_from_coords(coords: Sequence[Tuple[int, float, float, float]], a_idx: int, b_idx: int) -> Optional[float]:
    if a_idx <= 0 or b_idx <= 0 or a_idx > len(coords) or b_idx > len(coords):
        return None
    _za, ax, ay, az = coords[a_idx - 1]
    _zb, bx, by, bz = coords[b_idx - 1]
    dx = ax - bx
    dy = ay - by
    dz = az - bz
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def ts_bond_drift_check_messages(
    ts_out: Path,
    ts_bond_delta: float,
) -> Tuple[bool, List[str]]:
    messages: List[str] = []
    co_out = corresponding_co_out_for_ts(ts_out)
    if co_out is None or not co_out.exists():
        messages.append(
            f"         [WARN] {ts_out.name}: TS bond-check enabled but no matching CO output found; skipping bond check."
        )
        return False, messages

    pairs = parse_modred_bond_freeze_pairs(co_out)
    if not pairs:
        gjf = best_matching_gjf_for_co_out(co_out)
        if gjf is not None and gjf.exists():
            pairs = parse_bond_pairs_from_gjf_title(gjf)
            if pairs:
                messages.append(
                    f"         [TS-CHECK] Using constraint pairs from {gjf.name} Title block: {format_bond_pairs_line(pairs)}"
                )

    if not pairs:
        messages.append(
            f"         [WARN] {ts_out.name}: No constrained bond pairs found in {co_out.name} (or fallback CO.gjf Title); skipping bond check."
        )
        return False, messages

    co_coords = extract_last_orientation(co_out)
    ts_coords = extract_last_orientation(ts_out)
    if co_coords is None or ts_coords is None:
        messages.append(
            f"         [WARN] {ts_out.name}: Could not parse last orientation for CO/TS; skipping bond check."
        )
        return False, messages

    bad = False
    for a_idx, b_idx in pairs:
        dist_co = bond_distance_from_coords(co_coords, a_idx, b_idx)
        dist_ts = bond_distance_from_coords(ts_coords, a_idx, b_idx)
        if dist_co is None or dist_ts is None:
            messages.append(
                f"         [TS-CHECK] atoms {a_idx}-{b_idx}: could not compute distance (index out of range?)"
            )
            continue
        delta = abs(dist_ts - dist_co)
        messages.append(
            f"         [TS-CHECK] {co_out.name} vs {ts_out.name} | atoms {a_idx}-{b_idx}: "
            f"d(CO)={dist_co:.3f} A, d(TS)={dist_ts:.3f} A, |D|={delta:.3f} A (thr={ts_bond_delta:.3f} A)"
        )
        if delta > ts_bond_delta:
            bad = True
    return bad, messages


def ts_bond_drift_exceeds_threshold(ts_out: Path, ts_bond_delta: float, logger: TeeLogger) -> bool:
    bad, messages = ts_bond_drift_check_messages(ts_out, ts_bond_delta)
    for message in messages:
        logger.log(message)
    return bad


def flag_out_as_nots(out_path: Path, logger: TeeLogger, tag: str = "NoTS") -> Optional[Path]:
    old_name = out_path.name
    if f"={tag}".lower() in out_path.stem.lower():
        logger.log(f"         [RENAME] {old_name} already tagged (={tag}); skipping rename.")
        return out_path

    target = out_path.with_name(f"{out_path.stem}={tag}{out_path.suffix}")
    if target.exists():
        stem, suffix = target.stem, target.suffix
        idx = 2
        while True:
            candidate = target.with_name(f"{stem}{idx}{suffix}")
            if not candidate.exists():
                target = candidate
                break
            idx += 1

    try:
        out_path.rename(target)
        logger.log(f"         [RENAME] {old_name} -> {target.name}")
        return target
    except Exception as exc:
        logger.log(f"         [RENAME-FAIL] Could not rename {old_name} -> {target.name}: {exc}")
        return None


def parse_link0_value(line: str, key: str) -> Optional[str]:
    match = re.match(rf"^\s*%{re.escape(key)}\s*=\s*(.+?)\s*$", line, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


def scale_memory_value(value: str, factor: float) -> Optional[str]:
    match = re.match(r"^\s*(\d+(?:\.\d+)?)\s*([A-Za-z]+)?\s*$", value)
    if not match:
        return None
    number = float(match.group(1))
    unit = (match.group(2) or "").strip()
    scaled = max(1.0, number * factor)
    if abs(scaled - round(scaled)) < 1e-9:
        num_text = str(int(round(scaled)))
    else:
        num_text = f"{scaled:.2f}".rstrip("0").rstrip(".")
    return num_text + unit


def apply_link0_adjustments(lines: List[str], adjustments: Optional[Link0Adjustments]) -> Tuple[List[str], List[str]]:
    if adjustments is None:
        return lines, []
    new_lines = list(lines)
    notes: List[str] = []

    if adjustments.mem_scale is not None:
        for idx, line in enumerate(new_lines):
            value = parse_link0_value(line, "mem")
            if value is None:
                continue
            scaled = scale_memory_value(value, adjustments.mem_scale)
            if scaled is None or scaled == value:
                break
            new_lines[idx] = re.sub(
                r"(?i)^\s*%mem\s*=.*$",
                f"%mem={scaled}",
                line.rstrip("\n"),
            ) + "\n"
            notes.append(f"%mem={value} -> %mem={scaled}")
            break

    return new_lines, notes


def split_route_tokens(route_text: str) -> List[str]:
    tokens: List[str] = []
    current: List[str] = []
    depth = 0
    for char in route_text.strip():
        if char.isspace() and depth == 0:
            if current:
                tokens.append("".join(current))
                current = []
            continue
        current.append(char)
        if char == "(":
            depth += 1
        elif char == ")" and depth > 0:
            depth -= 1
    if current:
        tokens.append("".join(current))
    return tokens


def split_comma_terms(text: str) -> List[str]:
    terms: List[str] = []
    current: List[str] = []
    depth = 0
    for char in text:
        if char == "," and depth == 0:
            term = "".join(current).strip()
            if term:
                terms.append(term)
            current = []
            continue
        current.append(char)
        if char == "(":
            depth += 1
        elif char == ")" and depth > 0:
            depth -= 1
    tail = "".join(current).strip()
    if tail:
        terms.append(tail)
    return terms


def parse_route_components(route_text: str) -> Tuple[str, List[str]]:
    tokens = split_route_tokens(route_text)
    if not tokens:
        return "#", []
    if tokens[0].startswith("#"):
        return tokens[0], tokens[1:]
    return "#", tokens


def rebuild_route(prefix: str, tokens: Sequence[str]) -> str:
    return collapse_spaces(" ".join([prefix, *tokens]))


def parse_keyword_terms_from_token(token: str, keyword: str) -> Optional[List[str]]:
    keyword_lower = keyword.lower()
    lowered = token.lower()
    if lowered == keyword_lower:
        return []
    if not lowered.startswith(keyword_lower):
        return None
    match = re.fullmatch(rf"(?i){re.escape(keyword)}\s*=\s*(.+)", token)
    if not match:
        return None
    value = match.group(1).strip()
    if value.startswith("(") and value.endswith(")"):
        value = value[1:-1]
    return split_comma_terms(value)


def collect_keyword_terms(tokens: Sequence[str], keyword: str) -> Tuple[bool, List[str]]:
    found = False
    merged: List[str] = []
    for token in tokens:
        terms = parse_keyword_terms_from_token(token, keyword)
        if terms is None:
            continue
        found = True
        merged = merge_opt_terms(merged, terms, keyword)
    return found, merged


def find_keyword_token(tokens: Sequence[str], keyword: str) -> Tuple[Optional[int], Optional[List[str]]]:
    keyword_lower = keyword.lower()
    for idx, token in enumerate(tokens):
        lowered = token.lower()
        if lowered == keyword_lower:
            return idx, []
        if lowered.startswith(keyword_lower + "="):
            return idx, parse_keyword_terms_from_token(token, keyword)
    return None, None


def format_keyword_token(keyword: str, terms: Sequence[str]) -> Optional[str]:
    terms = [term.strip() for term in terms if term.strip()]
    if not terms:
        return None
    if len(terms) == 1:
        return f"{keyword}={terms[0]}"
    return f"{keyword}=({','.join(terms)})"


def replace_keyword_terms(tokens: Sequence[str], keyword: str, terms: Sequence[str]) -> List[str]:
    idx, _existing = find_keyword_token(tokens, keyword)
    new_tokens = list(tokens)
    replacement = format_keyword_token(keyword, terms)
    if idx is None:
        if replacement is not None:
            new_tokens.append(replacement)
        return new_tokens
    if replacement is None:
        del new_tokens[idx]
    else:
        new_tokens[idx] = replacement
    return new_tokens


def replace_all_keyword_terms(tokens: Sequence[str], keyword: str, terms: Sequence[str]) -> List[str]:
    replacement = format_keyword_token(keyword, terms)
    new_tokens: List[str] = []
    inserted = False
    for token in tokens:
        if parse_keyword_terms_from_token(token, keyword) is not None:
            if not inserted and replacement is not None:
                new_tokens.append(replacement)
                inserted = True
            continue
        new_tokens.append(token)
    if not inserted and replacement is not None:
        new_tokens.append(replacement)
    return new_tokens


def ensure_keyword_terms(route_text: str, keyword: str, additions: Sequence[str]) -> str:
    additions = [term.strip() for term in additions if term.strip()]
    if not additions:
        return collapse_spaces(route_text)
    prefix, tokens = parse_route_components(route_text)
    _found, current_terms = collect_keyword_terms(tokens, keyword)
    merged = merge_opt_terms(current_terms, additions, keyword)
    tokens = replace_all_keyword_terms(tokens, keyword, merged)
    return rebuild_route(prefix, tokens)


def remove_keyword_terms(route_text: str, keyword: str, remove_keys: Sequence[str]) -> str:
    remove_keys = [key.strip().lower() for key in remove_keys if key.strip()]
    if not remove_keys:
        return collapse_spaces(route_text)
    prefix, tokens = parse_route_components(route_text)
    found, existing = collect_keyword_terms(tokens, keyword)
    if not found:
        return rebuild_route(prefix, tokens)
    if "__all__" in remove_keys:
        tokens = replace_all_keyword_terms(tokens, keyword, [])
        return rebuild_route(prefix, tokens)
    if existing is None:
        return rebuild_route(prefix, tokens)
    kept = [term for term in existing if route_term_key(term) not in remove_keys]
    tokens = replace_all_keyword_terms(tokens, keyword, kept)
    return rebuild_route(prefix, tokens)


def ensure_scf_terms(route_text: str, additions: Sequence[str]) -> str:
    return ensure_keyword_terms(route_text, "scf", additions)


def token_is_model_chemistry(token: str) -> bool:
    lowered = token.lower()
    if "/" in token:
        return True
    if lowered.startswith("oniom("):
        return True
    return False


def ensure_top_level_tokens(route_text: str, additions: Sequence[str]) -> str:
    additions = [term.strip() for term in additions if term.strip()]
    if not additions:
        return collapse_spaces(route_text)
    prefix, tokens = parse_route_components(route_text)
    lower_tokens = [token.lower() for token in tokens]
    for addition in additions:
        if addition.lower() not in lower_tokens:
            tokens.append(addition)
            lower_tokens.append(addition.lower())
    return rebuild_route(prefix, tokens)


def infer_keyword_for_fragment(fragment: str) -> Optional[str]:
    lowered = fragment.strip().lower()
    key = route_term_key(fragment)
    if lowered.startswith("opt="):
        return "opt"
    if lowered.startswith("scf="):
        return "scf"
    if key in {"loose", "gdiis", "calcfc", "calcall", "cartesian", "modredundant"}:
        return "opt"
    if key in {"xqc", "yqc"}:
        return "scf"
    if lowered.startswith("recalcfc=") or lowered.startswith("maxstep="):
        return "opt"
    return None


def remove_top_level_tokens(route_text: str, removals: Sequence[str]) -> str:
    removal_set = {value.strip().lower() for value in removals if value.strip()}
    if not removal_set:
        return collapse_spaces(route_text)
    prefix, tokens = parse_route_components(route_text)
    kept = [token for token in tokens if token.lower() not in removal_set]
    return rebuild_route(prefix, kept)


def collect_route_removal_items(route_text: str) -> List[RemovalItem]:
    _prefix, tokens = parse_route_components(route_text)
    items: List[RemovalItem] = []
    for token in tokens:
        lower = token.lower()
        if lower.startswith("opt=") or lower == "opt":
            items.append(RemovalItem("opt", "__all__", "opt:(entire keyword)", "__all__"))
            terms = parse_keyword_terms_from_token(token, "opt") or []
            for term in terms:
                items.append(RemovalItem("opt", route_term_key(term), f"opt:{term}", term))
            continue
        if lower.startswith("scf=") or lower == "scf":
            items.append(RemovalItem("scf", "__all__", "scf:(entire keyword)", "__all__"))
            terms = parse_keyword_terms_from_token(token, "scf") or []
            for term in terms:
                items.append(RemovalItem("scf", route_term_key(term), f"scf:{term}", term))
            continue
        if token_is_model_chemistry(token):
            continue
        items.append(RemovalItem("token", lower, token, token))
    return items


def annotate_presence(text: str, count: int, total: int) -> str:
    if total <= 1 or count >= total:
        return text
    return f"{text}[{count}/{total}]"


def summarize_route_texts(route_texts: Sequence[str]) -> str:
    usable = [route for route in route_texts if route]
    if not usable:
        return "#"

    prefix = parse_route_components(usable[0])[0]
    total = len(usable)
    order: List[Tuple[str, str]] = []

    top_display: Dict[str, str] = {}
    top_counts: Dict[str, int] = {}
    keyword_display: Dict[str, Dict[str, str]] = {"opt": {}, "scf": {}}
    keyword_counts: Dict[str, Dict[str, int]] = {"opt": {}, "scf": {}}
    keyword_order: Dict[str, List[str]] = {"opt": [], "scf": []}
    bare_counts: Dict[str, int] = {"opt": 0, "scf": 0}

    def remember_order(kind: str, key: str) -> None:
        marker = (kind, key)
        if marker not in order:
            order.append(marker)

    for route in usable:
        _route_prefix, tokens = parse_route_components(route)
        seen_top = set()
        seen_terms = {"opt": set(), "scf": set()}
        seen_bare = set()

        for token in tokens:
            opt_terms = parse_keyword_terms_from_token(token, "opt")
            if opt_terms is not None:
                remember_order("keyword", "opt")
                if opt_terms:
                    for term in opt_terms:
                        key = collapse_spaces(term).lower()
                        if key not in keyword_display["opt"]:
                            keyword_display["opt"][key] = term
                            keyword_order["opt"].append(key)
                        if key not in seen_terms["opt"]:
                            keyword_counts["opt"][key] = keyword_counts["opt"].get(key, 0) + 1
                            seen_terms["opt"].add(key)
                else:
                    seen_bare.add("opt")
                continue

            scf_terms = parse_keyword_terms_from_token(token, "scf")
            if scf_terms is not None:
                remember_order("keyword", "scf")
                if scf_terms:
                    for term in scf_terms:
                        key = collapse_spaces(term).lower()
                        if key not in keyword_display["scf"]:
                            keyword_display["scf"][key] = term
                            keyword_order["scf"].append(key)
                        if key not in seen_terms["scf"]:
                            keyword_counts["scf"][key] = keyword_counts["scf"].get(key, 0) + 1
                            seen_terms["scf"].add(key)
                else:
                    seen_bare.add("scf")
                continue

            key = collapse_spaces(token).lower()
            remember_order("token", key)
            if key not in top_display:
                top_display[key] = token
            if key not in seen_top:
                top_counts[key] = top_counts.get(key, 0) + 1
                seen_top.add(key)

        for keyword in seen_bare:
            bare_counts[keyword] += 1

    rendered: List[str] = []
    for kind, key in order:
        if kind == "token":
            rendered.append(annotate_presence(top_display[key], top_counts.get(key, 0), total))
            continue

        keyword = key
        term_keys = keyword_order[keyword]
        if term_keys:
            terms = [
                annotate_presence(keyword_display[keyword][term_key], keyword_counts[keyword].get(term_key, 0), total)
                for term_key in term_keys
            ]
            rendered.append(f"{keyword}=({','.join(terms)})")
        elif bare_counts[keyword] > 0:
            rendered.append(annotate_presence(keyword, bare_counts[keyword], total))

    return collapse_spaces(" ".join([prefix, *rendered]))


def format_removal_items_inline(
    items: Sequence[RemovalItem],
    counts: Optional[Dict[Tuple[str, str], int]] = None,
    total: int = 1,
) -> str:
    chunks: List[str] = []
    for idx, item in enumerate(items, start=1):
        label = item.display
        if counts is not None:
            label = annotate_presence(label, counts.get(removal_item_identity(item), 0), total)
        chunks.append(f"{idx}:{label}")
    return " | ".join(chunks)


def apply_route_choice(route_text: str, choice: RouteChoice) -> str:
    route = route_text
    opt_remove_keys = [item.remove_key for item in choice.removals if item.kind == "opt"]
    scf_remove_keys = [item.remove_key for item in choice.removals if item.kind == "scf"]
    token_removals = [item.exact_value for item in choice.removals if item.kind == "token"]

    route = ensure_keyword_terms(route, "opt", choice.opt_additions)
    route = ensure_scf_terms(route, choice.scf_additions)
    route = ensure_top_level_tokens(route, choice.raw_additions)

    route = remove_keyword_terms(route, "opt", opt_remove_keys)
    route = remove_keyword_terms(route, "scf", scf_remove_keys)
    route = remove_top_level_tokens(route, token_removals)
    return collapse_spaces(route)


def prompt_user(logger: TeeLogger, prompt: str) -> str:
    logger.log(prompt)
    try:
        answer = input("> ").strip()
    except EOFError:
        answer = ""
    logger.log(f"         [INPUT] {answer}")
    return answer


def parse_selection_string(raw: str, options: Sequence[MenuOption]) -> List[Tuple[str, Optional[str]]]:
    cleaned = re.sub(r"[\s,\-]+", "", raw.strip())
    if not cleaned:
        return []

    by_code = {option.code: option for option in options}
    out: List[Tuple[str, Optional[str]]] = []
    seen = set()
    index = 0
    while index < len(cleaned):
        code = cleaned[index]
        if code not in by_code:
            raise ValueError(f"Invalid selection '{code}'")
        index += 1

        inline_value: Optional[str] = None
        if index < len(cleaned) and cleaned[index] == "(":
            end = cleaned.find(")", index + 1)
            if end == -1:
                raise ValueError(f"Missing closing ')' for selection '{code}'")
            inline_value = cleaned[index + 1:end].strip()
            if not inline_value:
                raise ValueError(f"Selection '{code}' needs a value inside parentheses")
            index = end + 1

        option = by_code[code]
        if option.requires_int:
            if inline_value is None:
                raise ValueError(f"Selection '{code}' requires a number like {code}(10)")
            if not inline_value.isdigit() or int(inline_value) <= 0:
                raise ValueError(f"Selection '{code}' requires a positive integer")
        elif inline_value is not None:
            raise ValueError(f"Selection '{code}' does not take a number")

        if code not in seen:
            seen.add(code)
            out.append((code, inline_value))
    return out


def resolve_menu_selection(logger: TeeLogger, title: str) -> RouteChoice:
    logger.log(title)
    for option in MENU_OPTIONS:
        logger.log(f"  {option.code}) {option.label}")
    logger.log("  Enter codes like '1746' or '1-7-4-6'; inline values like '23(10)89(500)' or '2-3(10)-8-9(500)'.")

    while True:
        raw = prompt_user(logger, "Select addition options (blank for none):")
        try:
            selected = parse_selection_string(raw, MENU_OPTIONS)
            break
        except ValueError as exc:
            logger.log(f"[WARN] {exc}")

    choice = RouteChoice(opt_additions=[], scf_additions=[], raw_additions=[], removals=[])
    for code, inline_value in selected:
        option = next(opt for opt in MENU_OPTIONS if opt.code == code)
        if option.kind == "custom":
            raw_fragments = prompt_user(
                logger,
                "Enter complete route fragment(s), comma-separated. Example: opt=maxstep=5,scf=maxcycle=1024,nosymm",
            )
            for fragment in split_comma_terms(raw_fragments):
                keyword = infer_keyword_for_fragment(fragment)
                if keyword == "opt":
                    parsed = parse_keyword_terms_from_token(fragment, "opt")
                    choice.opt_additions.extend(parsed if parsed is not None else [fragment.strip()])
                elif keyword == "scf":
                    parsed = parse_keyword_terms_from_token(fragment, "scf")
                    choice.scf_additions.extend(parsed if parsed is not None else [fragment.strip()])
                else:
                    choice.raw_additions.append(fragment)
            continue

        if option.requires_int:
            term_value = f"{option.value}={inline_value}"
        else:
            term_value = option.value or ""

        if option.kind == "opt":
            choice.opt_additions.append(term_value)
        elif option.kind == "scf":
            choice.scf_additions.append(term_value)
        elif option.kind == "raw":
            choice.raw_additions.append(term_value)

    return choice


def clone_route_choice(choice: RouteChoice) -> RouteChoice:
    return RouteChoice(
        opt_additions=list(choice.opt_additions),
        scf_additions=list(choice.scf_additions),
        raw_additions=list(choice.raw_additions),
        removals=list(choice.removals),
    )


def parse_index_selection(raw: str, count: int) -> List[int]:
    if not raw.strip():
        return []

    seen = set()
    out: List[int] = []

    def append_value(value: int, token: str) -> None:
        if not (1 <= value <= count):
            raise ValueError(f"Invalid selection '{token}'")
        if value not in seen:
            seen.add(value)
            out.append(value)

    for part in [item for item in re.split(r"[\s,]+", raw.strip()) if item]:
        for subpart in [item for item in part.split("-") if item]:
            if not subpart.isdigit():
                raise ValueError(f"Invalid selection '{subpart}'")

            if count <= 9:
                for char in subpart:
                    append_value(int(char), char)
                continue

            value = int(subpart)
            if 1 <= value <= count:
                append_value(value, subpart)
                continue

            for char in subpart:
                append_value(int(char), char)
    return out


def resolve_removal_selection(logger: TeeLogger, route_text: str, job_name: str) -> List[RemovalItem]:
    items = collect_route_removal_items(route_text)
    logger.log(f"[ROUTE] {job_name}: {summarize_route_texts([route_text])}")
    if not items:
        logger.log("  No removable route keywords were detected outside the model chemistry token(s).")
        return []

    logger.log(f"[REMOVE] {format_removal_items_inline(items)}")
    if len(items) <= 9:
        logger.log("  Enter concatenated digits like '14' or use space/comma/dash-separated numbers.")
    else:
        logger.log("  Enter space/comma-separated numbers, or split multi-digit choices with dashes like '134-11'.")

    while True:
        raw = prompt_user(logger, "Select removal options (blank for none):")
        try:
            selected = parse_index_selection(raw, len(items))
            return [items[idx - 1] for idx in selected]
        except ValueError as exc:
            logger.log(f"[WARN] {exc}")


def removal_item_identity(item: RemovalItem) -> Tuple[str, str]:
    if item.kind == "token":
        return item.kind, item.exact_value.lower()
    return item.kind, item.remove_key


def resolve_group_removal_selection(
    logger: TeeLogger,
    route_texts: Sequence[str],
    label: str,
) -> List[RemovalItem]:
    if not route_texts:
        return []
    total = len(route_texts)
    if len(set(route_texts)) == 1:
        return resolve_removal_selection(logger, route_texts[0], f"{label} group")

    catalog: Dict[Tuple[str, str], RemovalItem] = {}
    counts: Dict[Tuple[str, str], int] = {}
    for route_text in route_texts:
        seen_case = set()
        for item in collect_route_removal_items(route_text):
            key = removal_item_identity(item)
            if key not in catalog:
                catalog[key] = item
            if key not in seen_case:
                counts[key] = counts.get(key, 0) + 1
                seen_case.add(key)

    ordered_keys = list(catalog.keys())
    items = [catalog[key] for key in ordered_keys]
    logger.log(f"[ROUTE] {label}: {summarize_route_texts(route_texts)}")
    logger.log(f"[REMOVE] {format_removal_items_inline(items, counts=counts, total=total)}")
    if len(ordered_keys) <= 9:
        logger.log("  Enter concatenated digits like '14' or use space/comma/dash-separated numbers.")
    else:
        logger.log("  Enter space/comma-separated numbers, or split multi-digit choices with dashes like '134-11'.")

    while True:
        raw = prompt_user(logger, f"Select shared removal options for {label} (blank for none):")
        try:
            selected = parse_index_selection(raw, len(ordered_keys))
            return [catalog[ordered_keys[idx - 1]] for idx in selected]
        except ValueError as exc:
            logger.log(f"[WARN] {exc}")


def find_route_block(lines: Sequence[str]) -> Tuple[Optional[int], Optional[int], Optional[str]]:
    route_start = next((idx for idx, line in enumerate(lines) if line.lstrip().startswith("#")), None)
    if route_start is None:
        return None, None, None
    route_end = route_start
    while route_end < len(lines) and lines[route_end].strip() != "":
        route_end += 1
    route_text = collapse_spaces(" ".join(line.strip() for line in lines[route_start:route_end] if line.strip()))
    return route_start, route_end, route_text


def find_charge_mult_idx(lines: Sequence[str]) -> Optional[int]:
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        if INTEGER_LINE_RE.fullmatch(stripped) and len(stripped.split()) % 2 == 0:
            return idx
    return None


def find_geometry_block(lines: Sequence[str], cm_idx: int) -> Tuple[Optional[int], Optional[int]]:
    geom_start = cm_idx + 1
    geom_end = geom_start
    saw_atom = False
    while geom_end < len(lines):
        if lines[geom_end].strip() == "":
            if saw_atom:
                break
            geom_end += 1
            continue
        saw_atom = True
        geom_end += 1
    if not saw_atom:
        return None, None
    return geom_start, geom_end


def is_numeric_token(token: str) -> bool:
    return bool(NUMERIC_TOKEN_RE.fullmatch(token))


def parse_geometry_template_line(raw_line: str) -> Optional[GeometryLineTemplate]:
    tokens = raw_line.split()
    if len(tokens) < 4:
        return None
    for start in range(1, len(tokens) - 2):
        trio = tokens[start:start + 3]
        after = tokens[start + 3] if start + 3 < len(tokens) else None
        before = tokens[start - 1] if start - 1 >= 1 else None
        if not all(is_numeric_token(token) for token in trio):
            continue
        if after is not None and is_numeric_token(after):
            continue
        if before is not None and is_numeric_token(before) and start == 1:
            continue
        return GeometryLineTemplate(
            label=tokens[0],
            prefix_tokens=tokens[1:start],
            suffix_tokens=tokens[start + 3:],
        )
    return None


def format_plain_coords(coords: Sequence[Tuple[int, float, float, float]]) -> List[str]:
    output: List[str] = []
    for atomic_number, x_coord, y_coord, z_coord in coords:
        element = ELEMENTS[atomic_number] if 0 <= atomic_number < len(ELEMENTS) else "X"
        output.append(f" {element:<2s}{x_coord:>18.8f}{y_coord:>13.8f}{z_coord:>13.8f}\n")
    return output


def format_coords_from_templates(
    coords: Sequence[Tuple[int, float, float, float]],
    templates: Sequence[GeometryLineTemplate],
) -> List[str]:
    if len(coords) != len(templates):
        return format_plain_coords(coords)

    output: List[str] = []
    for (atomic_number, x_coord, y_coord, z_coord), template in zip(coords, templates):
        element = ELEMENTS[atomic_number] if 0 <= atomic_number < len(ELEMENTS) else template.label
        line = template.label if template.label else element
        if template.prefix_tokens:
            line += " " + " ".join(template.prefix_tokens)
        line += f" {x_coord:>16.8f} {y_coord:>16.8f} {z_coord:>16.8f}"
        if template.suffix_tokens:
            line += " " + " ".join(template.suffix_tokens)
        output.append(line + "\n")
    return output


def geometry_templates_from_lines(lines: Sequence[str]) -> List[GeometryLineTemplate]:
    templates: List[GeometryLineTemplate] = []
    for line in lines:
        parsed = parse_geometry_template_line(line)
        if parsed is None:
            return []
        templates.append(parsed)
    return templates


def gjf_has_modredundant_constraints(lines: Sequence[str], geom_start_idx: int, natoms: int) -> bool:
    if natoms <= 0:
        return False
    index = geom_start_idx + natoms
    while index < len(lines) and lines[index].strip() == "":
        index += 1
    if index >= len(lines):
        return False
    return bool(MODRED_FREEZE_RE.match(lines[index].rstrip("\n")))


def format_selected_step(block: ConvergenceBlock, desc: str) -> str:
    step_text = f"opt step {block.opt_step}" if block.opt_step is not None else "opt step ?"
    return f"[PICK] Using {desc}: line {block.start_line} ({step_text})"


def last_opt_step_in_output(out_path: Path) -> Optional[int]:
    current_step: Optional[int] = None
    try:
        with out_path.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                current_step = opt_step_update(line, current_step)
    except OSError:
        return None
    return current_step


def resolve_gauss_out2in(path_hint: Optional[str]) -> Optional[List[str]]:
    if path_hint:
        candidate = Path(path_hint).expanduser()
        if candidate.exists():
            return [sys.executable, str(candidate.resolve())]
    local = Path.cwd() / "Gauss_out2in.py"
    if local.exists():
        return [sys.executable, str(local.resolve())]
    which = shutil.which("Gauss_out2in.py")
    return [which] if which else None


def resolve_orca_out2in(path_hint: Optional[str]) -> Optional[List[str]]:
    if path_hint:
        candidate = Path(path_hint).expanduser()
        if candidate.exists():
            return [sys.executable, str(candidate.resolve())]

    candidates = [
        Path.cwd() / "Gout2ORCAin.py",
        Path.cwd() / "Goniom2ORCAin.py",
        Path(__file__).resolve().parent / "Gout2ORCAin.py",
        Path(__file__).resolve().parent / "Goniom2ORCAin.py",
        Path.home() / "Desktop" / "Computational_Chemistry" / "Scripts_Executables" / "Gout2ORCAin.py",
        Path.home() / "Desktop" / "Computational_Chemistry" / "Scripts_Executables" / "Goniom2ORCAin.py",
    ]
    for candidate in candidates:
        if candidate.exists():
            return [sys.executable, str(candidate.resolve())]

    for name in ("Gout2ORCAin.py", "Goniom2ORCAin.py"):
        which = shutil.which(name)
        if which:
            return [which]
    return None


def next_job_params_for_success(out_path: Path) -> Optional[Tuple[int, str, str]]:
    if TS_SP_OUT_RE.match(out_path.name) or GENERIC_SP_OUT_RE.match(out_path.name):
        return None
    if CO_OUT_RE.match(out_path.name):
        return 1, "TS", "TS"
    if TS_OUT_RE.match(out_path.name):
        return 2, "TS_sp", "SP"
    return 2, "sp", "SP"


def expected_next_gjf_name(out_path: Path) -> Optional[str]:
    if TS_SP_OUT_RE.match(out_path.name) or GENERIC_SP_OUT_RE.match(out_path.name):
        return None
    stem = out_path.stem
    if CO_OUT_RE.match(out_path.name):
        base = stem[:-3] if stem.lower().endswith("_co") else stem
        return f"{base}_TS.gjf"
    if TS_OUT_RE.match(out_path.name):
        return f"{stem}_sp.gjf"
    return f"{stem}_sp.gjf"


def expected_next_orca_name(out_path: Path) -> Optional[str]:
    # Same target stem as the Gaussian branch; only the extension differs.
    gjf = expected_next_gjf_name(out_path)
    return None if gjf is None else f"{gjf[:-4]}.inp"


def orca_name_end(out_path: Path, name_end: str) -> str:
    """-n to pass Gout2ORCAin.py, which always writes
    <first_underscore_token>_<name_end>.inp.  Folding the rest of the stem into
    name_end reconstructs the full stem; without it every conformer sharing a
    leading token (TS1-A_Re_conformer*) collapses onto one .inp and overwrites
    the last.  Legacy un-prefixed stems (conformer11_TS -> TS_sp) are unchanged.
    """
    expected = expected_next_orca_name(out_path)
    if expected is None:
        return name_end
    head = out_path.stem.split("_")[0]
    return expected[len(head) + 1:-4]


def run_gauss_out2in(
    cmd_prefix: Sequence[str],
    out_path: Path,
    j_val: int,
    n_val: str,
    expected: str,
    logger: TeeLogger,
) -> Tuple[bool, str]:
    run_dir = out_path.parent.resolve()
    cmd = list(cmd_prefix) + [out_path.name, "-j", str(j_val), "-n", str(n_val)]
    try:
        result = subprocess.run(cmd, cwd=str(run_dir), capture_output=True, text=True)
    except Exception as exc:
        return False, f"[MAKE] Failed running Gauss_out2in.py for {out_path.name}: {exc}"

    expected_path = run_dir / expected
    if result.returncode == 0 and expected_path.exists():
        return True, f"[MAKE] Created {expected}"

    detail = ""
    combined = [line.strip() for line in (result.stdout.splitlines() + result.stderr.splitlines()) if line.strip()]
    for line in reversed(combined):
        if line.startswith("Traceback"):
            continue
        if re.match(r'^File ".*", line \d+, in ', line):
            continue
        if line == "main()":
            continue
        if re.fullmatch(r"[~^]+", line):
            continue
        detail = line
        break
    message = f"[MAKE] Gauss_out2in did NOT create {expected} (rc={result.returncode})"
    if detail:
        message += f": {detail}"
    return False, message


# Whole-.gjf Gaussian template (may carry --Link1-- steps) with two placeholders,
# [name] and [geometry].  OPT-IN ONLY: without --gauss-template the next-step .gjf
# is still built by Gauss_out2in.py exactly as before.  Named like
# ORCA_TEMPLATE_DEFAULT so the two mechanisms read the same way.
GAUSS_TEMPLATE_DEFAULT = "template_Link1.gjf"

# The template carries its OWN route and its OWN charge/multiplicity line, and
# [geometry] is filled with element+xyz only (format_plain_coords).  Anything the
# SOURCE job needs that a plain element+xyz block cannot carry is therefore lost
# without a trace, so those sources are REFUSED rather than downgraded:
#   oniom(...) -> H/L layer letters, link-atom specs ("L H 112"), the freeze
#                 column and the "0 1 0 1 0 1" charge line all vanish, turning a
#                 2-layer TS into a 1-layer job that terminates normally at the
#                 wrong level of theory.  This is the exact layer-stripping
#                 defect Gauss_out2in.py already had to be patched for.
#   gen/genecp -> the basis/ECP tail blocks after the geometry vanish.
ROUTE_FENCE_RE = re.compile(r"^\s*-{20,}\s*$")
ROUTE_START_RE = re.compile(r"^\s*#")
ONIOM_ROUTE_RE = re.compile(r"\boniom\s*\(", re.I)
GEN_BASIS_ROUTE_RE = re.compile(r"\bgen(ecp)?\b", re.I)
# " Charge =  0 Multiplicity = 1" and the ONIOM triple
# " Charge =  0 Multiplicity = 1 for low   level calculation on real  system."
OUT_CHARGE_MULT_RE = re.compile(r"^\s*Charge\s*=\s*(-?\d+)\s+Multiplicity\s*=\s*(-?\d+)")

# Stages --gauss-template is allowed to drive.  next_job_params_for_success()
# labels the CO -> TS handoff "TS" and every *_sp handoff "SP".  The shipped
# template's route is a MINIMISATION (freq opt), so letting it serve "TS" would
# replace opt=(calcfc,ts,noeigen,maxstep=10) with an ordinary optimisation; the
# resulting positive first frequency then fails the freq <= -50 gate and
# flag_out_as_nots() RENAMES the .out to =NoTS, which reads as a science result.
# Default to the single-point stage only, and require an explicit opt-in for TS.
GAUSS_TEMPLATE_STAGES_DEFAULT: Set[str] = {"SP"}


def gauss_template_applies(template: Optional[str], stage_key: str, stages: Set[str]) -> bool:
    """True only when a template was asked for AND this stage was selected for it.

    Mirrors the ORCA gate (`stage_key in orca_for_stages`).  Both halves are
    required: without the stage half the template silently hijacks a TS search.
    """
    return bool(template) and stage_key in stages


def resolve_gauss_template(path_hint: Union[str, Path], run_dir: Path) -> Optional[Path]:
    """Absolute path of an EXISTING template, or None.

    A relative hint is tried against the .out's own directory first (so a
    per-directory template_Link1.gjf wins) and then against this script's own
    directory, which is the only place template_Link1.gjf actually ships --
    mirroring resolve_orca_out2in()'s script-dir fallback.  Returning None for a
    miss (rather than a path that does not exist) is what lets main() fail once,
    at startup, instead of once per .out file.
    """
    hint = Path(path_hint).expanduser()
    candidates = [hint] if hint.is_absolute() else [run_dir / hint, Path(__file__).resolve().parent / hint]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def template_charge_mult(text: str) -> Optional[Tuple[int, int]]:
    """The template's own (charge, multiplicity), or None if it has no 2-field one.

    An ONIOM template's "0 1 0 1 0 1" is deliberately rejected here: this writer
    emits plain element+xyz and has no business filling an ONIOM input.
    """
    lines = text.splitlines(True)
    idx = find_charge_mult_idx(lines)
    if idx is None:
        return None
    fields = lines[idx].split()
    if len(fields) != 2:
        return None
    return int(fields[0]), int(fields[1])


def source_job_profile(out_path: Path) -> Tuple[Optional[str], Optional[Tuple[int, int]]]:
    """(route text, (charge, multiplicity)) of the FIRST job in out_path.

    Gaussian prints the route between two dashed fences, wrapped at a fixed width
    and broken MID-TOKEN ("... scr" / "f=(cpcm,..."), so the pieces are rejoined
    with no separator after dropping the one-space indent.  Joining on whitespace
    instead would hide "oniom(" from the caller, which is the whole point of
    reading the route.  Only the first Charge line is taken: for ONIOM that is
    the real system's overall charge.
    """
    route_parts: List[str] = []
    charge_mult: Optional[Tuple[int, int]] = None
    in_route = False
    prev_fence = False
    try:
        with out_path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                fence = bool(ROUTE_FENCE_RE.match(line))
                if in_route:
                    if fence:
                        in_route = False
                    else:
                        piece = line.rstrip("\n")
                        route_parts.append(piece[1:] if piece.startswith(" ") else piece)
                elif not route_parts and prev_fence and ROUTE_START_RE.match(line):
                    in_route = True
                    piece = line.rstrip("\n")
                    route_parts.append(piece[1:] if piece.startswith(" ") else piece)
                elif charge_mult is None:
                    match = OUT_CHARGE_MULT_RE.match(line)
                    if match:
                        charge_mult = (int(match.group(1)), int(match.group(2)))
                if route_parts and not in_route and charge_mult is not None:
                    break
                prev_fence = fence
    except OSError:
        return None, None
    return ("".join(route_parts).strip() or None), charge_mult


def write_gjf_from_template(
    template_path: Path,
    out_path: Path,
    expected: str,
    backup: bool = False,
    dry_run: bool = False,
) -> Tuple[bool, str]:
    """Write run_dir/<expected> from the template, substituting [name] and [geometry].

    dry_run runs EVERY check and differs only in the final write, so
    --no-write-inputs can never report success for a template the real run
    refuses.

    [name] -> the created file's own stem, in EVERY occurrence, so a --Link1--
    step's `geom=allcheck guess=read` reads the %chk the first step wrote; two
    different stems there would send Link1 to a checkpoint that does not exist.
    [geometry] -> the last standard orientation of out_path as element + xyz
    (format_plain_coords).  Because that carries no ONIOM layer/link-atom tokens
    and no gen/genecp tail, and because the template's charge/multiplicity is
    fixed text, such sources are refused above rather than silently converted.
    """
    try:
        text = template_path.read_text(encoding="utf-8")
    except OSError as exc:
        return False, f"[MAKE] Cannot read Gaussian template {template_path} ({type(exc).__name__})"

    if "[geometry]" not in text:
        return False, f"[MAKE] Gaussian template {template_path} has no [geometry] placeholder"

    # Gaussian ends the molecule specification at a BLANK line.  Without one the
    # next line (--Link1--, or a tail block) is read as an atom and the job dies
    # in l101 after taking the queue slot.
    if text.count("[geometry]\n\n") != text.count("[geometry]"):
        return False, (
            f"[MAKE] Gaussian template {template_path.name}: [geometry] must be followed by a blank "
            "line to terminate the molecule specification (add one empty line before --Link1--)"
        )

    template_cm = template_charge_mult(text)
    if template_cm is None:
        return False, (
            f"[MAKE] Gaussian template {template_path.name} has no 2-field charge/multiplicity line"
        )

    route, source_cm = source_job_profile(out_path)
    if route is None:
        return False, f"[MAKE] Could not read the route line of {out_path.name}; refusing to guess its job type"
    if ONIOM_ROUTE_RE.search(route):
        return False, (
            f"[MAKE] {out_path.name} is an ONIOM job; --gauss-template writes plain element+xyz and "
            "would silently drop the H/L layers, link atoms and ONIOM charge line. "
            "Use Gauss_out2in.py for ONIOM sources."
        )
    if GEN_BASIS_ROUTE_RE.search(route):
        return False, (
            f"[MAKE] {out_path.name} uses gen/genecp; --gauss-template writes no basis/ECP tail block "
            "and would silently change the basis set. Use Gauss_out2in.py for gen sources."
        )
    if source_cm is None:
        return False, (
            f"[MAKE] No 'Charge = / Multiplicity =' line in {out_path.name}; refusing to assume "
            f"{template_cm[0]} {template_cm[1]}"
        )
    if source_cm != template_cm:
        return False, (
            f"[MAKE] {out_path.name} is charge {source_cm[0]} multiplicity {source_cm[1]} but "
            f"{template_path.name} hardcodes {template_cm[0]} {template_cm[1]}; refusing to write a "
            "job that would terminate normally in the wrong electronic state"
        )

    coords = extract_last_orientation(out_path)
    if not coords:
        return False, f"[MAKE] No orientation found in {out_path.name}; cannot fill [geometry]"

    stem = expected[:-4] if expected.lower().endswith(".gjf") else expected
    # A template with no [name] is usable (it may hardcode %chk) but the caller
    # must hear about it -- a Link1 chain pointing at the wrong .chk still starts.
    warning = "" if "[name]" in text else f"\n         [WARN] no [name] in {template_path.name}; %chk left as written"
    body = "".join(format_plain_coords(coords)).rstrip("\n")
    text = text.replace("[name]", stem).replace("[geometry]", body)

    target = out_path.parent / expected
    if dry_run:
        return True, f"[MAKE] (would write {expected} from {template_path}){warning}"
    try:
        # Same .bak rule as rewrite_gjf_with_route(): only when asked, only over an
        # existing file, and never clobber a .bak that is already there.
        if backup and target.exists():
            backup_path = target.with_suffix(target.suffix + ".bak")
            if not backup_path.exists():
                backup_path.write_text(target.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")
        target.write_text(text, encoding="utf-8")
    except OSError as exc:
        return False, f"[MAKE] Cannot write {expected} ({type(exc).__name__})"
    return True, f"[MAKE] Created {expected} from {template_path.name}{warning}"


# Gout2ORCAin.py and the divergent NCI-H fork both declare
# -t/--template default "ORCA_out2in_info.txt".
ORCA_TEMPLATE_DEFAULT = "ORCA_out2in_info.txt"

# Leading %-directive of a fenced block: '%cpcm' from '%cpcm' or '%cpcm smd true'.
BLOCK_DIRECTIVE_RE = re.compile(r"^\s*(%\w+)")
# 'SMDsolvent "formic acid"' / 'solvent water' / 'CPCMsolvent = toluene' -> the name.
BLOCK_SOLVENT_RE = re.compile(r'^\s*(?:smd|cpcm)?solvent\s*[:=]?\s*"?([^"]+?)"?\s*$', re.I)
BLOCK_SMD_ON_RE = re.compile(r"\bsmd\s+true\b", re.I)

# splice_job_block_into_inp() outcomes.
SPLICE_INSERTED = "inserted"
SPLICE_PRESENT = "present"
SPLICE_CONFLICT = "conflict"
SPLICE_NO_BLOCK = "no_block"
SPLICE_NO_ROUTE = "no_route"
SPLICE_UNREADABLE = "unreadable"
SPLICE_UNWRITABLE = "unwritable"


@dataclass
class SpliceResult:
    """What splice_job_block_into_inp() did, so the caller can say so out loud."""
    status: str
    detail: str = ""


def resolve_orca_template(path_hint: Optional[Union[str, Path]], run_dir: Path) -> Path:
    """The ONE absolute info-file path used for both -t and the block read.

    The helper is run with cwd=run_dir and defaults -t to "ORCA_out2in_info.txt",
    so an omitted --orca-template means it reads run_dir/ORCA_out2in_info.txt.
    Mirroring that default here -- and passing the resolved absolute path back to
    the helper with -t -- is what guarantees the route line and the fenced
    %-block come from the SAME file.  There are eight ORCA_out2in_info.txt copies
    under NCI-J and they do NOT agree on the level of theory (Calculations/ has
    M062X/def2-TZVPP, the NCI-J root copy has M06/def2-TZVP), so a route line
    from one plus an SMD block from another terminates normally and silently
    reports the wrong method.
    """
    if path_hint:
        return Path(path_hint).expanduser().resolve()
    return (run_dir / ORCA_TEMPLATE_DEFAULT).resolve()


def block_specifies_solvation(block: Sequence[str]) -> bool:
    """True if the fenced block actually switches SMD on.

    Mirrors DI_IRC_sp/irc2orca.py :: block_is_smd() minus its project-specific
    solvent-name check, so "(+solvation block)" is only ever claimed for a block
    that really requests solvation -- a fence holding blank lines or an unrelated
    %-block must not earn that label.
    """
    return bool(BLOCK_SMD_ON_RE.search(" ".join(block)))


def block_solvent(lines: Sequence[str]) -> Optional[str]:
    """Normalised solvent name declared inside a %-block, or None."""
    for line in lines:
        match = BLOCK_SOLVENT_RE.match(line)
        if match:
            return " ".join(match.group(1).split()).lower()
    return None


def find_directive_block(lines: Sequence[str], directive: str) -> Optional[Tuple[int, int]]:
    """(start, end) indices of an existing '%<directive>' ... 'end' block, or None.

    Matched case-insensitively and independently of indentation.  The old
    idempotence test was an exact contiguous-line comparison, so it missed a
    pre-existing block that was indented differently, cased differently, or
    carried an extra directive such as num_leb -- and two %cpcm blocks naming
    different solvents run to normal termination with the WRONG solvent.
    """
    want = directive.lower()
    for idx, line in enumerate(lines):
        head = line.strip().lower()
        if head != want and not head.startswith(want + " "):
            continue
        if head.endswith(" end"):
            return idx, idx
        for j in range(idx + 1, len(lines)):
            if lines[j].strip().lower() == "end":
                return idx, j
        return idx, len(lines) - 1
    return None


def parse_fenced_job_block(text: str, j_val: int) -> Tuple[List[str], str]:
    """(block, reason) for the Job_block<N>_begin/_end fence in `text`.

    `reason` is "" only when a usable block was found.  A fence MUST be closed by
    its _end marker: the previous version set keep=True at _begin and cleared it
    only at _end, so an unterminated fence returned the whole remainder of the
    template -- including "charge-multiplicity: 0 1" -- and spliced that into the
    .inp while still reporting success.  An unterminated, empty, or
    blank-line-only fence is no block at all.  Markers are compared
    case-insensitively on the stripped line and the content is emitted verbatim,
    matching DI_IRC_sp/irc2orca.py :: info_block().
    """
    begin, end = f"job_block{j_val}_begin", f"job_block{j_val}_end"
    keep = closed = False
    block: List[str] = []
    for raw in text.splitlines():
        marker = raw.strip().lower()
        if marker == begin:
            if closed:
                continue                    # first complete fence wins
            keep, block = True, []
        elif marker == end:
            if keep:
                keep, closed = False, True
        elif keep:
            block.append(raw.rstrip())
    if keep:
        return [], f"unterminated Job_block{j_val}_begin (no Job_block{j_val}_end)"
    if not closed:
        return [], f"no Job_block{j_val} fence"
    if not any(line.strip() for line in block):
        return [], f"empty Job_block{j_val} fence"
    return block, ""


# ORCA %-directives that take a value and no 'end' ('%maxcore 8000').
ONE_LINE_DIRECTIVES = {"%maxcore", "%moinp", "%base", "%pointcharges"}
INFO_STRUCTURAL_RE = re.compile(
    r"(?i)^\s*(job\s*[_\s]*info\s*\d+\s*[:=]|job_block\d+_(begin|end)\s*$|charge.*multiplicity|!)")


def parse_unfenced_blocks(text: str, j_val: int) -> Tuple[List[List[str]], str]:
    """(blocks, error) for %-blocks written straight into the info file, no fence.

    A block runs from a line starting with '%' to its standalone 'end' (plus any
    immediately following standalone 'end' lines, for nested sub-blocks such as
    %geom Constraints ... end end), or is one line ('%pal nprocs 8 end',
    '%maxcore 8000').  It belongs to the nearest Job_info<N> line ABOVE it; a
    block with no Job_info line above it applies to every job.  Fenced content is
    skipped (parse_fenced_job_block() owns it).  An unterminated block is an
    error, never a silent truncation.  Gout2ORCAin.py ignores all of these lines.
    """
    lines = text.splitlines()
    blocks: List[List[str]] = []
    owner: Optional[int] = None
    in_fence = False
    i = 0
    while i < len(lines):
        s = lines[i].strip()
        low = s.lower()
        i += 1
        if re.fullmatch(r"job_block\d+_begin", low):
            in_fence = True
        elif re.fullmatch(r"job_block\d+_end", low):
            in_fence = False
        if in_fence or not s or s.startswith("#"):
            continue
        m = re.match(r"(?i)^job\s*[_\s]*info\s*(\d+)\s*[:=]", s)
        if m:
            owner = int(m.group(1))
            continue
        if not s.startswith("%"):
            continue
        block = [lines[i - 1].rstrip()]
        directive = low.split()[0]
        if directive not in ONE_LINE_DIRECTIVES and low.split()[-1] != "end":
            while True:
                if i >= len(lines) or lines[i].strip().startswith("%") or INFO_STRUCTURAL_RE.match(lines[i]):
                    return [], f"unterminated {directive} block (no standalone 'end')"
                block.append(lines[i].rstrip())
                i += 1
                if block[-1].strip().lower() == "end":
                    while i < len(lines) and lines[i].strip().lower() == "end":
                        block.append(lines[i].rstrip())
                        i += 1
                    break
        if owner is None or owner == j_val:
            blocks.append(block)
    return blocks, ""


# Route-line implicit solvation: CPCM, CPCMC, SMD, optionally with '(solvent)'.
ROUTE_SOLV_RE = re.compile(r"(?i)(?<![\w-])(CPCMC|CPCM|SMD)(?:\(([^)]*)\))?(?![\w-])")


def implicit_solvent_warning(inp_path: Path) -> str:
    """Non-empty if the route asks for CPCM/SMD but nothing names the solvent.

    A bare '! ... CPCM' with no %cpcm block and no CPCM(<solvent>) runs to normal
    termination with ORCA's built-in default medium, not the intended solvent.
    """
    try:
        lines = inp_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    route = " ".join(l for l in lines if l.lstrip().startswith("!"))
    hits = ROUTE_SOLV_RE.findall(route)
    if not hits or any(name.strip() for _, name in hits) or find_directive_block(lines, "%cpcm"):
        return ""
    kw = "/".join(sorted({k.upper() for k, _ in hits}))
    return (f"[WARN] SOLVENT NOT SET: route has bare {kw} but {inp_path.name} has no %cpcm "
            f"block and no {kw}(<solvent>) - ORCA will use its DEFAULT medium, not your solvent")


def read_template_job_block_status(
    orca_template: Optional[Union[str, Path]],
    j_val: int,
) -> Tuple[List[str], str]:
    """read_template_job_block() plus the reason the block is unusable, if it is."""
    if not orca_template:
        return [], "no template path"
    try:
        text = Path(orca_template).expanduser().read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return [], f"cannot read template ({type(exc).__name__})"
    return parse_fenced_job_block(text, j_val)


def read_template_job_block(orca_template: Optional[Union[str, Path]], j_val: int) -> List[str]:
    """Literal %-block lines fenced as Job_block<N>_begin/_end in the ORCA info file.

    Semantics deliberately match DI_IRC_sp/irc2orca.py :: info_block().
    Gout2ORCAin.py's parse_info_file() ignores these lines, so without this the
    generated .inp is silently GAS PHASE.

    Returns [] when there is no template, no closed fence for this job number, an
    empty fence, or the file cannot be read -- the caller then leaves the .inp
    exactly as generated and MUST say so (see read_template_job_block_status()).
    """
    return read_template_job_block_status(orca_template, j_val)[0]


def splice_job_block_into_inp(inp_path: Path, block: Sequence[str]) -> SpliceResult:
    """Insert the fenced %-block immediately after the route line.

    Placement (first '!' line, verbatim content) matches irc2orca.py ::
    write_input() so both tools produce identical inputs from the same template.

    Never creates a SECOND block of the same %-directive: if the .inp already
    carries one -- whatever its indentation, case, or extra directives, e.g. the
    %cpcm block the divergent NCI-H/Gout2ORCAin.py fork writes itself -- nothing
    is written, and a differing solvent is reported as SPLICE_CONFLICT rather
    than silently doubled.  Any unreadable/unwritable/route-less .inp is left
    untouched rather than broken.
    """
    block = list(block)
    if not block:
        return SpliceResult(SPLICE_NO_BLOCK)
    try:
        lines = inp_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return SpliceResult(SPLICE_UNREADABLE)
    if any(lines[idx:idx + len(block)] == block for idx in range(len(lines))):
        return SpliceResult(SPLICE_PRESENT, "identical block already present")

    directive = next((m.group(1) for m in (BLOCK_DIRECTIVE_RE.match(l) for l in block) if m), None)
    if directive:
        span = find_directive_block(lines, directive)
        if span is not None:
            have, want = block_solvent(lines[span[0]:span[1] + 1]), block_solvent(block)
            if have and want and have != want:
                return SpliceResult(
                    SPLICE_CONFLICT,
                    f"{directive} block already present with solvent {have!r}, "
                    f"template block wants {want!r}")
            return SpliceResult(SPLICE_PRESENT, f"{directive} block already present")

    route_idx = next((idx for idx, line in enumerate(lines) if line.lstrip().startswith("!")), None)
    if route_idx is None:
        return SpliceResult(SPLICE_NO_ROUTE, "no '!' route line")
    lines[route_idx + 1:route_idx + 1] = block
    try:
        inp_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    except OSError:
        return SpliceResult(SPLICE_UNWRITABLE)
    return SpliceResult(SPLICE_INSERTED)


def orca_block_message(inp_path: Path, template: Path, j_val: int, expected: str) -> str:
    """"[MAKE] Created ..." plus an EARNED statement of what happened to the block.

    Absence of a message is never the only signal: when no usable block is found,
    or it cannot be spliced, the reason and the template path ACTUALLY consulted
    are printed, because a gas-phase .inp otherwise terminates normally and its
    energies are not comparable to the solvated DI_IRC_sp dataset.
    """
    fenced, reason = read_template_job_block_status(template, j_val)
    try:
        text = Path(template).read_text(encoding="utf-8", errors="replace")
    except OSError:
        text = ""
    loose, loose_err = parse_unfenced_blocks(text, j_val)
    blocks = ([fenced] if fenced else []) + loose
    notes: List[str] = []
    warns: List[str] = []
    if loose_err:
        warns.append(f"[WARN] {loose_err} in {template} - that block was NOT added")
    for block in blocks:
        label = "solvation block" if block_specifies_solvation(block) else "job block"
        outcome = splice_job_block_into_inp(inp_path, block)
        if outcome.status == SPLICE_INSERTED:
            notes.append(f"+{label}")
        elif outcome.status == SPLICE_PRESENT:
            notes.append(f"{label} already present: {outcome.detail}")
        elif outcome.status == SPLICE_CONFLICT:
            warns.append(f"[WARN] {outcome.detail} - refusing to write a second "
                         f"block; verify the solvent in {expected} before submitting")
        else:
            warns.append(f"[WARN] {label} from {template} NOT spliced "
                         f"({outcome.status}{': ' + outcome.detail if outcome.detail else ''})")
    solvent_warn = implicit_solvent_warning(inp_path)
    if solvent_warn:
        warns.append(solvent_warn)
    elif not blocks:
        warns.append(f"[WARN] {reason} and no unfenced %-block in {template} - generated input is gas phase")
    created = f"[MAKE] Created {expected}" + (f" ({'; '.join(notes)})" if notes else "")
    return "\n         ".join([created] + warns)


def run_orca_out2in(
    cmd_prefix: Sequence[str],
    out_path: Path,
    j_val: int,
    n_val: str,
    expected: str,
    orca_template: Optional[str],
) -> Tuple[bool, str]:
    run_dir = out_path.parent.resolve()
    # ONE resolved path for both the subprocess and the block read.  Passing the
    # absolute -t reproduces what the helper would have done on its own (its
    # default "ORCA_out2in_info.txt" resolved against cwd=run_dir) while removing
    # any chance that the route line and the %-block come from two different
    # ORCA_out2in_info.txt copies -- previously the subprocess resolved a relative
    # -t against run_dir while the reader resolved it against CheckGauss's cwd.
    template = resolve_orca_template(orca_template, run_dir)
    cmd = list(cmd_prefix) + [out_path.name, "-j", str(j_val), "-n", str(n_val),
                              "-t", str(template)]
    try:
        result = subprocess.run(cmd, cwd=str(run_dir), capture_output=True, text=True)
    except Exception as exc:
        return False, f"[MAKE] Failed running ORCA out2in helper for {out_path.name}: {exc}"

    expected_path = run_dir / expected
    if result.returncode == 0 and expected_path.exists():
        return True, orca_block_message(expected_path, template, j_val, expected)

    detail = ""
    combined = [line.strip() for line in (result.stdout.splitlines() + result.stderr.splitlines()) if line.strip()]
    for line in reversed(combined):
        if line.startswith("Traceback"):
            continue
        if re.match(r'^File ".*", line \d+, in ', line):
            continue
        if line == "main()":
            continue
        if re.fullmatch(r"[~^]+", line):
            continue
        detail = line
        break
    message = f"[MAKE] ORCA out2in helper did NOT create {expected} (rc={result.returncode})"
    if detail:
        message += f": {detail}"
    return False, message


def classify_failure(out_path: Path, joblog_status: JoblogStatus, tail_n: int = 400) -> FailureInfo:
    if joblog_status.timed_out:
        return FailureInfo("timeout", "TIMEOUT", "Joblog indicates walltime timeout.", True)

    tail_text = read_tail_text(out_path, tail_n)
    if GALLOC_RE.search(tail_text):
        return FailureInfo("galloc", "GALLOC", "Detected galloc memory allocation failure.", True)
    if FAIL_OPT_MAX_RE.search(tail_text):
        return FailureInfo("opt_maxcycles", "OPTMAX", "Optimization hit Gaussian step-limit / Link 9999.", True)
    if FAIL_INTERNAL_RE.search(tail_text):
        return FailureInfo("internal_coords", "INTCOORD", "Internal-coordinate optimizer failure.", True)
    if FAIL_SCF_RE.search(tail_text):
        return FailureInfo("scf_convergence", "SCF", "SCF convergence failure detected.", True)
    if FAIL_MEMORY_WORDS_RE.search(tail_text):
        return FailureInfo("mem_words", "MEMWORDS", "Gaussian reports insufficient in-core memory words.", True)
    if FAIL_DISK_IO_RE.search(tail_text):
        return FailureInfo("disk_io", "DISKIO", "Disk or scratch I/O failure detected.", False)
    if FAIL_INPUT_RE.search(tail_text):
        return FailureInfo("input_format", "INPUT", "Input-format error detected.", False)
    if FAIL_CHECKPOINT_RE.search(tail_text):
        return FailureInfo("checkpoint", "CHECKPOINT", "Checkpoint path/read error detected.", False)
    return FailureInfo("unknown", "GENFAIL", "Unclassified abnormal termination.", True)


def choose_geometry_for_repair(
    out_path: Path,
    prefer_closest_block: bool,
) -> Tuple[Optional[List[Tuple[int, float, float, float]]], Optional[ConvergenceBlock], Optional[str]]:
    blocks = parse_convergence_blocks(out_path)
    if blocks:
        last_block = blocks[-1]
        if last_block.all_yes:
            coords = extract_orientation_before_line(out_path, last_block.start_line)
            if coords is not None:
                return coords, last_block, format_selected_step(last_block, "last block (all YES)")
        if prefer_closest_block:
            chosen = pick_closest_block(blocks)
            if chosen is not None:
                coords = extract_orientation_before_line(out_path, chosen.start_line)
                if coords is not None:
                    return coords, chosen, format_selected_step(
                        chosen,
                        f"closest-to-convergence block (SSE={chosen.sse_exceedance:.6g})",
                    )

    coords = extract_last_orientation(out_path)
    if coords is not None:
        return coords, None, "[PICK] Using last orientation in output."
    return None, None, None


def repair_plan_for_failure(
    failure: FailureInfo,
    gjf_lines: Sequence[str],
    geom_start: int,
    natoms: int,
    cfg: ImproveConfig,
) -> Tuple[RepairPlan, str]:
    has_modred = gjf_has_modredundant_constraints(gjf_lines, geom_start, natoms)

    if failure.kind == "timeout":
        return RepairPlan(
            label="timeout",
            edit_mode="stage",
            use_closest_block=True,
        ), failure.detail

    if failure.kind == "galloc":
        return RepairPlan(
            label="galloc",
            edit_mode="keep",
            use_last_orientation=True,
            allow_no_geometry=True,
        ), failure.detail + " Geometry can be refreshed, but no route or resource edits are needed."

    if failure.kind == "opt_maxcycles":
        return RepairPlan(
            label="opt-maxcycles",
            edit_mode="custom",
            use_closest_block=True,
            route_opt_add=cfg.opt_limit_add,
            route_scf=cfg.stage1_scf,
        ), failure.detail

    if failure.kind == "scf_convergence":
        return RepairPlan(
            label="scf-convergence",
            edit_mode="scf_only",
            use_last_orientation=True,
        ), failure.detail

    if failure.kind == "internal_coords":
        additions = cfg.internal_opt_add_modred if has_modred else cfg.internal_opt_add
        note = failure.detail
        if has_modred:
            note += " ModRedundant constraints found, so Cartesian rescue was softened."
        return RepairPlan(
            label="internal-coord",
            edit_mode="custom",
            use_last_orientation=True,
            route_opt_add=additions,
            route_scf=cfg.stage1_scf,
        ), note

    if failure.kind == "mem_words":
        return RepairPlan(
            label="mem-words",
            edit_mode="keep",
            use_last_orientation=True,
            link0=Link0Adjustments(mem_scale=cfg.mem_shortage_scale),
            allow_no_geometry=True,
        ), failure.detail + " Increasing %mem may help if scheduler resources allow it."

    if failure.kind == "disk_io":
        return RepairPlan(
            label="disk-io",
            edit_mode="manual",
            manual_only=True,
        ), failure.detail + " This usually needs more scratch/disk, not route changes."

    if failure.kind == "input_format":
        return RepairPlan(
            label="input-format",
            edit_mode="manual",
            manual_only=True,
        ), failure.detail + " Auto-rerunning would likely repeat the same parsing error."

    if failure.kind == "checkpoint":
        return RepairPlan(
            label="checkpoint",
            edit_mode="manual",
            manual_only=True,
        ), failure.detail + " Fix the %chk path or checkpoint availability first."

    return RepairPlan(
        label="generic",
        edit_mode="scf_only",
        use_last_orientation=False,
        allow_no_geometry=True,
    ), failure.detail + " Applying conservative SCF-only fallback."


def rewrite_gjf(
    gjf_path: Path,
    coords: Optional[Sequence[Tuple[int, float, float, float]]],
    plan: RepairPlan,
    backup: bool,
    cfg: ImproveConfig,
) -> Tuple[bool, str]:
    if not gjf_path.exists():
        return False, f"[WARN] No input file found to update: {gjf_path.name}"

    original_text = gjf_path.read_text(encoding="utf-8", errors="replace")
    lines = original_text.splitlines(True)

    route_start, route_end, route_text = find_route_block(lines)
    if route_start is None or route_end is None or route_text is None:
        return False, f"[ERROR] {gjf_path.name}: could not find route line starting with #"

    cm_idx = find_charge_mult_idx(lines)
    if cm_idx is None:
        return False, f"[ERROR] {gjf_path.name}: could not find charge/multiplicity line"
    geom_start, geom_end = find_geometry_block(lines, cm_idx)
    if geom_start is None or geom_end is None:
        return False, f"[ERROR] {gjf_path.name}: could not identify geometry block to replace"

    new_lines = list(lines)
    message_bits: List[str] = []
    new_route = route_text
    current_route_end = route_end

    if plan.edit_mode == "remove_opt":
        new_route = remove_opt_from_route(route_text)
        new_lines[route_start:route_end] = [new_route + "\n"]
        current_route_end = route_start + 1
        message_bits.append(f"route: '{route_text}' -> '{new_route}'")
    elif plan.edit_mode == "stage":
        new_route, stage_label = apply_stage_improvement(route_text, cfg)
        new_lines[route_start:route_end] = [new_route + "\n"]
        current_route_end = route_start + 1
        message_bits.append(f"{stage_label}")
        message_bits.append(f"route: '{route_text}' -> '{new_route}'")
    elif plan.edit_mode == "scf_only":
        new_route, stage_label = apply_scf_only_escalation(route_text, cfg)
        new_lines[route_start:route_end] = [new_route + "\n"]
        current_route_end = route_start + 1
        message_bits.append(f"{stage_label}, SCF-only")
        message_bits.append(f"route: '{route_text}' -> '{new_route}'")
    elif plan.edit_mode == "custom":
        new_route = apply_specific_opt_add(
            route_text,
            plan.route_opt_add or [],
            plan.route_scf or "",
        )
        new_lines[route_start:route_end] = [new_route + "\n"]
        current_route_end = route_start + 1
        message_bits.append(f"route: '{route_text}' -> '{new_route}'")
    elif plan.edit_mode == "keep":
        pass
    elif plan.edit_mode == "manual":
        return False, f"[HOLD] {gjf_path.name}: manual intervention required."
    else:
        return False, f"[ERROR] {gjf_path.name}: unknown edit mode '{plan.edit_mode}'"

    template_lines = lines[geom_start:geom_end]
    if coords is not None:
        templates = geometry_templates_from_lines(template_lines)
        coord_lines = format_coords_from_templates(coords, templates)
        new_lines[geom_start:geom_end] = coord_lines
        message_bits.append("updated geometry")
    elif not plan.allow_no_geometry:
        return False, f"[ERROR] {gjf_path.name}: geometry update required but no coordinates were available"

    removed_constraints = 0
    captured_pairs: List[Tuple[int, int]] = []
    inserted_title_constraints = False
    if plan.edit_mode == "remove_opt" and coords is not None:
        new_lines, removed_constraints, captured_pairs = strip_freq_only_constraints_after_geometry(
            new_lines,
            geom_start_idx=geom_start,
            natoms=len(new_lines[geom_start:geom_start + len(coord_lines)]),
        )
        if captured_pairs:
            constraint_line = format_bond_pairs_line(captured_pairs)
            new_lines, inserted_title_constraints = upsert_constraints_line_into_title(
                new_lines,
                route_end_idx=current_route_end,
                constraint_line=constraint_line,
            )

    new_lines, link0_notes = apply_link0_adjustments(new_lines, plan.link0)
    message_bits.extend(link0_notes)
    if removed_constraints:
        message_bits.append(f"removed {removed_constraints} ModRedundant freeze line(s)")
    if inserted_title_constraints and captured_pairs:
        message_bits.append(f"title_constraints='{format_bond_pairs_line(captured_pairs)}'")

    if backup:
        backup_path = gjf_path.with_suffix(gjf_path.suffix + ".bak")
        if not backup_path.exists():
            backup_path.write_text(original_text, encoding="utf-8")

    gjf_path.write_text("".join(new_lines), encoding="utf-8")
    return True, f"[EDIT] Updated {gjf_path.name} ({plan.label}) | " + " | ".join(message_bits)


def failure_menu_guidance(
    failure: FailureInfo,
    has_modred: bool,
    timeout_converged: bool,
) -> Tuple[str, List[str]]:
    if failure.kind == "timeout":
        if timeout_converged:
            return (
                "Optimization appears converged. If this was an opt+freq-style job, removing the full opt keyword is often the right next move.",
                [],
            )
        return (
            "Timeouts often respond to looser optimization settings and stronger SCF convergence aids.",
            ["1", "3", "7"],
        )
    if failure.kind == "galloc":
        return (
            "This is usually a Hoffman2 / machine-side memory-allocation issue. Route edits are optional here.",
            [],
        )
    if failure.kind == "opt_maxcycles":
        return (
            "Optimization step-limit failures often benefit from calcfc/recalcfc, looser optimization, and a larger opt maxcycle.",
            ["1", "3", "4", "9", "7"],
        )
    if failure.kind == "internal_coords":
        if has_modred:
            return (
                "Internal-coordinate trouble with ModRedundant constraints present. `opt=cartesian` can still help, but constrained jobs sometimes prefer gentler `opt` changes first.",
                ["4", "7"],
            )
        return (
            "Internal-coordinate failures often improve with `opt=cartesian`, `opt=calcfc`, and SCF stabilization.",
            ["4", "6", "7"],
        )
    if failure.kind == "scf_convergence":
        return (
            "SCF failures often respond to `scf=xqc` or `scf=yqc`; sometimes pairing that with gentler optimization settings helps too.",
            ["7"],
        )
    if failure.kind == "mem_words":
        return (
            "Gaussian is asking for more in-core memory words. That usually points to %mem or scheduler settings rather than route edits.",
            [],
        )
    if failure.kind == "disk_io":
        return (
            "Disk or scratch I/O failures usually need filesystem or scratch-space fixes first. Route edits may not be enough.",
            [],
        )
    if failure.kind == "input_format":
        return (
            "Input-format failures usually need the input file inspected directly before rerunning.",
            [],
        )
    if failure.kind == "checkpoint":
        return (
            "Checkpoint failures usually need the %chk path or checkpoint availability fixed before rerunning.",
            [],
        )
    return (
        "This failure was not classified cleanly. Conservative SCF and optimization rescue terms are a reasonable place to start.",
        ["1", "7"],
    )


def prepare_failure_case(out_path: Path, failure: FailureInfo) -> FailureCase:
    gjf_path = out_path.with_suffix(".gjf")
    route_text: Optional[str] = None
    editable = True
    has_modred = False
    is_single_point = bool(GENERIC_SP_OUT_RE.match(out_path.name))

    if gjf_path.exists():
        try:
            gjf_lines = gjf_path.read_text(encoding="utf-8", errors="replace").splitlines(True)
        except OSError:
            gjf_lines = []
        _route_start, _route_end, route_text = find_route_block(gjf_lines) if gjf_lines else (None, None, None)
        cm_idx = find_charge_mult_idx(gjf_lines) if gjf_lines else None
        geom_start, geom_end = (None, None) if cm_idx is None else find_geometry_block(gjf_lines, cm_idx)
        natoms = 0 if geom_start is None or geom_end is None else (geom_end - geom_start)
        if geom_start is not None:
            has_modred = gjf_has_modredundant_constraints(gjf_lines, geom_start, natoms)
        if route_text is None:
            editable = False
    else:
        editable = False

    coords: Optional[List[Tuple[int, float, float, float]]] = None
    geometry_note = "[PICK] Keeping the current input geometry because no output geometry was recovered."
    timeout_converged = False

    if failure.kind == "timeout":
        blocks = parse_convergence_blocks(out_path)
        if blocks and blocks[-1].all_yes:
            chosen = blocks[-1]
            coords = extract_orientation_before_line(out_path, chosen.start_line)
            if coords is not None:
                geometry_note = format_selected_step(chosen, "last block (all YES)")
                timeout_converged = True
        if coords is None:
            coords, _block, selected_line = choose_geometry_for_repair(out_path, prefer_closest_block=True)
            if coords is not None and selected_line is not None:
                geometry_note = selected_line
    elif failure.kind == "opt_maxcycles":
        coords, _block, selected_line = choose_geometry_for_repair(out_path, prefer_closest_block=True)
        if coords is not None and selected_line is not None:
            geometry_note = selected_line
    else:
        coords = extract_last_orientation(out_path)
        if coords is not None:
            last_step = last_opt_step_in_output(out_path)
            if last_step is not None:
                geometry_note = f"[PICK] Using last orientation in output (opt step {last_step})."
            else:
                geometry_note = "[PICK] Using last orientation in output."

    if coords is None:
        geometry_note = "[PICK] No recoverable output geometry found; the existing .gjf geometry will be reused."

    suggestion_note, suggested_option_codes = failure_menu_guidance(
        failure,
        has_modred=has_modred,
        timeout_converged=timeout_converged,
    )

    if not gjf_path.exists():
        suggestion_note += " No matching .gjf file was found, so this job cannot be rewritten automatically."
    elif route_text is None:
        suggestion_note += " The matching .gjf route section could not be parsed, so this job cannot be rewritten automatically."
    elif is_single_point:
        editable = False
        suggestion_note += " Single-point (_sp) failures are reported but not rewritten automatically."

    return FailureCase(
        out_path=out_path,
        gjf_path=gjf_path if gjf_path.exists() else None,
        failure=failure,
        coords=coords,
        geometry_note=geometry_note,
        suggestion_note=suggestion_note,
        suggested_option_codes=suggested_option_codes,
        route_text=route_text,
        editable=editable,
    )


def rewrite_gjf_with_route(
    gjf_path: Path,
    coords: Optional[Sequence[Tuple[int, float, float, float]]],
    new_route_text: str,
    backup: bool,
) -> Tuple[bool, str]:
    if not gjf_path.exists():
        return False, f"[WARN] No input file found to update: {gjf_path.name}"

    original_text = gjf_path.read_text(encoding="utf-8", errors="replace")
    lines = original_text.splitlines(True)

    route_start, route_end, route_text = find_route_block(lines)
    if route_start is None or route_end is None or route_text is None:
        return False, f"[ERROR] {gjf_path.name}: could not find route line starting with #"

    cm_idx = find_charge_mult_idx(lines)
    if cm_idx is None:
        return False, f"[ERROR] {gjf_path.name}: could not find charge/multiplicity line"
    geom_start, geom_end = find_geometry_block(lines, cm_idx)
    if geom_start is None or geom_end is None:
        return False, f"[ERROR] {gjf_path.name}: could not identify geometry block to replace"

    new_lines = list(lines)
    new_lines[route_start:route_end] = [new_route_text + "\n"]
    message_bits = [f"route: '{route_text}' -> '{new_route_text}'"]

    if coords is not None:
        template_lines = lines[geom_start:geom_end]
        templates = geometry_templates_from_lines(template_lines)
        coord_lines = format_coords_from_templates(coords, templates)
        new_lines[geom_start:geom_end] = coord_lines
        message_bits.append("updated geometry")
    else:
        message_bits.append("kept existing geometry")

    if backup:
        backup_path = gjf_path.with_suffix(gjf_path.suffix + ".bak")
        if not backup_path.exists():
            backup_path.write_text(original_text, encoding="utf-8")

    gjf_path.write_text("".join(new_lines), encoding="utf-8")
    return True, f"[EDIT] Updated {gjf_path.name} | " + " | ".join(message_bits)


def log_completed_summary(logger: TeeLogger, completed: Dict[str, List[str]]) -> None:
    logger.log("")
    logger.log("[SUMMARY] Successful jobs")
    for key, label in (("plain", "Ground-state / other"), ("CO", "CO"), ("TS", "TS"), ("TS_sp", "TS_sp")):
        jobs = completed.get(key, [])
        if jobs:
            logger.log(f"  {label} ({len(jobs)}): {' '.join(jobs)}")
        else:
            logger.log(f"  {label} (0)")


def log_failure_summary(logger: TeeLogger, failure_cases: Sequence[FailureCase]) -> Dict[str, List[FailureCase]]:
    groups: Dict[str, List[FailureCase]] = {}
    for case in failure_cases:
        groups.setdefault(case.failure.label, []).append(case)

    logger.log("")
    logger.log("[SUMMARY] Failed jobs by error type")
    if not groups:
        logger.log("  none")
        return groups

    for label, cases in groups.items():
        logger.log(f"  {label} ({len(cases)}): {' '.join(case.out_path.name for case in cases)}")
    return groups


def compact_geometry_note(note: str) -> str:
    lowered = note.lower()
    step_match = re.search(r"opt step (\d+)", note, re.IGNORECASE)
    step_suffix = f" (step={step_match.group(1)})" if step_match else ""
    if "last orientation" in lowered:
        return "last orientation" + step_suffix
    if "last block (all yes)" in lowered:
        return "last converged block" + step_suffix
    if "closest-to-convergence block" in lowered:
        return "closest block" + step_suffix
    if "no recoverable output geometry" in lowered or "keeping the current input geometry" in lowered:
        return "kept input geometry"
    cleaned = note.replace("[PICK]", "").replace("Using ", "").strip()
    return collapse_spaces(cleaned).rstrip(".")


def case_processing_status(case: FailureCase) -> str:
    if case.gjf_path is None:
        return "hold: no .gjf"
    if case.route_text is None:
        return "hold: route not parsed"
    return "queued"


def prompt_group_mode(logger: TeeLogger, label: str, count: int) -> str:
    prompt = f"[GROUP] {label} has {count} editable job(s). Choose [a]ll same edits, [i]ndividual handling, or [s]kip:"
    if count == 1:
        prompt += " [Enter = all]"
    while True:
        raw = prompt_user(logger, prompt).lower()
        if raw == "" and count == 1:
            return "all"
        if raw in {"a", "all"}:
            return "all"
        if raw in {"i", "ind", "individual"}:
            return "individual"
        if raw in {"s", "skip"}:
            return "skip"
        logger.log("[WARN] Please enter 'a', 'i', or 's'.")


def process_failure_case(
    case: FailureCase,
    choice: RouteChoice,
    write_inputs: bool,
    backup_inputs: bool,
    logger: TeeLogger,
    next_gjfs: List[str],
) -> None:
    if case.gjf_path is None or case.route_text is None:
        logger.log(f"[HOLD] {case.out_path.name}: no editable .gjf route was available.")
        return

    new_route = apply_route_choice(case.route_text, choice)

    if write_inputs:
        ok, message = rewrite_gjf_with_route(case.gjf_path, case.coords, new_route, backup_inputs)
        if ok:
            next_gjfs.append(case.gjf_path.name)
            geometry_mode = "updated geometry" if case.coords is not None else "kept geometry"
            logger.log(f"[EDIT] {case.gjf_path.name}: route updated | {geometry_mode}")
        else:
            logger.log(message)
    else:
        logger.log(f"[EDIT] {case.gjf_path.name}: would update | {compact_geometry_note(case.geometry_note)}")
        next_gjfs.append(case.gjf_path.name)


def process_failure_groups(
    logger: TeeLogger,
    failure_cases: Sequence[FailureCase],
    write_inputs: bool,
    backup_inputs: bool,
    next_gjfs: List[str],
) -> None:
    groups = log_failure_summary(logger, failure_cases)
    if not groups:
        return

    for label, cases in groups.items():
        logger.log("")
        logger.log(f"[GROUP] {label}")
        editable_cases = [case for case in cases if case.editable and case.gjf_path is not None and case.route_text is not None]
        held_cases = [case for case in cases if case not in editable_cases]
        if editable_cases and editable_cases[0].suggested_option_codes:
            logger.log(f"  Suggested starting menu codes: {' '.join(editable_cases[0].suggested_option_codes)}")
        if editable_cases:
            logger.log(f"  Hint: {editable_cases[0].suggestion_note}")
        for idx, case in enumerate(cases, start=1):
            editable_flag = "editable" if case in editable_cases else "hold"
            logger.log(
                f"  {idx}. {case.out_path.name} [{editable_flag}] {case.failure.detail} | {compact_geometry_note(case.geometry_note)}"
            )
        if held_cases:
            logger.log(
                "  Holds: " + " ".join(f"{case.out_path.name}({case_processing_status(case)})" for case in held_cases)
            )

        if not editable_cases:
            logger.log("  No editable jobs in this group; skipping route prompts.")
            continue

        if len(editable_cases) == 1:
            mode = "all"
        else:
            mode = prompt_group_mode(logger, label, len(editable_cases))
        if mode == "skip":
            logger.log(f"  Skipping {label}.")
            continue

        shared_choice: Optional[RouteChoice] = None
        if mode == "all":
            logger.log(f"[ROUTE] {label}: {summarize_route_texts([case.route_text or '' for case in editable_cases])}")
            shared_choice = resolve_menu_selection(
                logger,
                f"[MENU] Shared additions for {label}",
            )
            preview_routes = [apply_route_choice(case.route_text or "", shared_choice) for case in editable_cases]
            shared_choice.removals = resolve_group_removal_selection(logger, preview_routes, label)

        for case in editable_cases:
            if mode == "individual":
                if case.suggested_option_codes:
                    logger.log(f"  Suggested starting menu codes: {' '.join(case.suggested_option_codes)}")
                logger.log(f"[ROUTE] {case.out_path.name}: {summarize_route_texts([case.route_text or ''])}")
                choice = resolve_menu_selection(logger, f"[MENU] Additions for {case.out_path.name}")
                preview_route = apply_route_choice(case.route_text or "", choice)
                choice.removals = resolve_removal_selection(logger, preview_route, case.out_path.name)
            else:
                choice = clone_route_choice(shared_choice or RouteChoice([], [], [], []))
            process_failure_case(case, choice, write_inputs, backup_inputs, logger, next_gjfs)


def evaluate_one(
    out_path: Path,
    tail_n: int,
    freq_threshold: float,
    joblog_dirs: Sequence[Path],
    out2in_cmd: Optional[Sequence[str]],
    gauss_template: Optional[str],
    gauss_template_for: Set[str],
    orca_out2in_cmd: Optional[Sequence[str]],
    orca_template: Optional[str],
    orca_for_stages: Set[str],
    write_inputs: bool,
    backup_inputs: bool,
    logger: TeeLogger,
    completed: Dict[str, List[str]],
    next_gjfs: List[str],
    next_inps: List[str],
    ts_bond_delta: float,
    ts_bond_check: bool,
    failure_cases: List[FailureCase],
) -> int:
    if has_normal_termination(out_path, tail_n=tail_n):
        freq = find_first_frequency(out_path)
        if freq is None:
            if TS_SP_OUT_RE.match(out_path.name) or GENERIC_SP_OUT_RE.match(out_path.name):
                logger.log(f"[DONE] {out_path.name}: Normal termination of single point calculation.")
                return 1
            logger.log(f"[WARN] {out_path.name}: Normal termination, but no 'Frequencies --' line found.")
            return 1

        constrained = needs_constrained_freq_check(out_path)
        if constrained and not (freq <= freq_threshold):
            logger.log(
                f"[FAIL] {out_path.name}: Normal termination. First frequency={freq:.4f} "
                f"(expected <= {freq_threshold}). Tagging as =NoTS."
            )
            flag_out_as_nots(out_path, logger, tag="NoTS")
            return 1

        if ts_bond_check and TS_OUT_RE.match(out_path.name):
            bad, messages = ts_bond_drift_check_messages(out_path, ts_bond_delta)
            if bad:
                for message in messages:
                    logger.log(message)
                logger.log(f"[FAIL] {out_path.name}: TS bond distance drift exceeded threshold; tagging as =NoTS.")
                flag_out_as_nots(out_path, logger, tag="NoTS")
                return 1
            logger.log(f"[OK]   {out_path.name}: Normal termination. First frequency = {freq:.4f} cm^-1")
            for message in messages:
                logger.log(message)
        else:
            logger.log(f"[OK]   {out_path.name}: Normal termination. First frequency = {freq:.4f} cm^-1")

        if TS_SP_OUT_RE.match(out_path.name):
            completed["TS_sp"].append(out_path.name)
        elif CO_OUT_RE.match(out_path.name):
            completed["CO"].append(out_path.name)
        elif TS_OUT_RE.match(out_path.name):
            completed["TS"].append(out_path.name)
        else:
            completed["plain"].append(out_path.name)

        params = next_job_params_for_success(out_path)
        if params:
            j_val, n_val, stage_key = params
            use_orca = stage_key in orca_for_stages
            if use_orca:
                n_val = orca_name_end(out_path, n_val)
            expected = expected_next_orca_name(out_path) if use_orca else expected_next_gjf_name(out_path)
            if expected:
                if use_orca:
                    helper_name = Path(orca_out2in_cmd[-1]).name if orca_out2in_cmd else "ORCA out2in helper"
                    if write_inputs:
                        if orca_out2in_cmd is None:
                            logger.log(
                                f"         [MAKE] Skipping ORCA handoff for {out_path.name}: ORCA out2in helper not found."
                            )
                            return 0
                        ok, message = run_orca_out2in(
                            orca_out2in_cmd,
                            out_path,
                            j_val,
                            n_val,
                            expected,
                            orca_template,
                        )
                        logger.log(f"         {message}")
                        if ok:
                            next_inps.append(expected)
                    else:
                        # same resolution the real run uses, so the dry run names the
                        # file that would actually supply the route line AND the block
                        template_text = f" -t {resolve_orca_template(orca_template, out_path.parent.resolve())}"
                        logger.log(
                            f"         [MAKE] (would run {helper_name} {out_path.name} -j {j_val} -n {n_val}{template_text})"
                        )
                        next_inps.append(expected)
                else:
                    helper_name = Path(out2in_cmd[-1]).name if out2in_cmd else "Gauss_out2in.py"
                    # Stage-gated exactly like the ORCA branch above: a template
                    # selected only for SP must never serve the CO -> TS handoff.
                    use_template = gauss_template_applies(gauss_template, stage_key, gauss_template_for)
                    template = resolve_gauss_template(gauss_template, out_path.parent.resolve()) if use_template else None
                    if use_template and template is None:
                        logger.log(
                            f"         [MAKE] Skipping Gaussian handoff for {out_path.name}: "
                            f"--gauss-template {gauss_template} not found for this directory."
                        )
                        return 0
                    if template is not None:
                        ok, message = write_gjf_from_template(
                            template,
                            out_path,
                            expected,
                            backup=backup_inputs,
                            dry_run=not write_inputs,
                        )
                        logger.log(f"         {message}")
                        if ok:
                            next_gjfs.append(expected)
                    elif write_inputs:
                        if out2in_cmd is None:
                            logger.log(
                                f"         [MAKE] Skipping Gaussian handoff for {out_path.name}: Gauss_out2in.py not found."
                            )
                            return 0
                        ok, message = run_gauss_out2in(out2in_cmd, out_path, j_val, n_val, expected, logger)
                        logger.log(f"         {message}")
                        if ok:
                            next_gjfs.append(expected)
                    else:
                        logger.log(f"         [MAKE] (would run {helper_name} {out_path.name} -j {j_val} -n {n_val})")
                        next_gjfs.append(expected)
        return 0

    joblog = find_latest_joblog_by_id(out_path, joblog_dirs)
    joblog_status = read_joblog_status(joblog) if joblog is not None else JoblogStatus()
    if joblog is not None and joblog_status.running:
        logger.log(f"[RUNNING] {out_path.name}: job still running (joblog {joblog.name}). Ignoring.")
        return 0

    failure = classify_failure(out_path, joblog_status)

    if ts_bond_check and TS_OUT_RE.match(out_path.name):
        if ts_bond_drift_exceeds_threshold(out_path, ts_bond_delta, logger):
            logger.log(f"[FAIL] {out_path.name}: ({failure.label}) TS bond distance drift exceeded threshold; tagging as =NoTS.")
            flag_out_as_nots(out_path, logger, tag="NoTS")
            return 1

    case = prepare_failure_case(out_path, failure)
    failure_cases.append(case)
    logger.log(
        f"[{failure.label}] {out_path.name}: {failure.detail} | {compact_geometry_note(case.geometry_note)} | {case_processing_status(case)}"
    )
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Manage Gaussian .out jobs; create next-step Gaussian or ORCA inputs; group failed jobs by error type; interactively rebuild reruns."
    )
    parser.add_argument("inputs", nargs="+", help="One or more .out files, globs, or directories.")
    parser.add_argument("--recursive", action="store_true", help="If directories are given, search recursively for *.out.")
    parser.add_argument("--tail", type=int, default=5, help="Lines from bottom to detect Normal termination. (default 5)")
    parser.add_argument("-freq", "--freq-threshold", type=float, default=-50.0, help="CO/TS first freq must be <= this. (default -50)")
    parser.add_argument("--joblog-dir", action="append", default=[], help="Extra directory to search for joblog.<stem>.<JOBID> (may repeat).")
    parser.add_argument("--no-write-inputs", action="store_true", help="Do not modify/create .gjf or .inp files.")
    parser.add_argument("--backup-inputs", action="store_true", help="Create .bak before overwriting .gjf files.")
    parser.add_argument("--gauss-out2in", default=None, help="Path to Gauss_out2in.py (optional).")
    parser.add_argument(
        "--gauss-template",
        nargs="?",
        const=GAUSS_TEMPLATE_DEFAULT,
        default=None,
        help=(
            "Build each next-step .gjf from this whole-input template ([name] -> the new "
            f"file's stem in every %%chk, [geometry] -> the last orientation) instead of "
            f"running Gauss_out2in.py. Bare flag means {GAUSS_TEMPLATE_DEFAULT} in the .out's "
            "directory, then next to this script. Applies ONLY to the stages named by "
            "--gauss-template-for. Omit the flag for the existing Gauss_out2in.py behaviour. "
            "NOTE: the flag takes an optional value, so write --gauss-template=FILE (or put the "
            "bare flag AFTER the .out list) or it will swallow your first .out."
        ),
    )
    parser.add_argument(
        "--gauss-template-for",
        action="append",
        default=[],
        help=(
            "Stages --gauss-template may drive: TS, SP, or ALL (repeatable / comma-separated). "
            f"Default {','.join(sorted(GAUSS_TEMPLATE_STAGES_DEFAULT))}. The default template is a "
            "minimisation (freq opt), so selecting TS replaces opt=(calcfc,ts,noeigen,...) on the "
            "CO -> TS handoff and turns a TS search into an optimisation -- opt in deliberately."
        ),
    )
    parser.add_argument(
        "-orca",
        "--orca-for",
        action="append",
        default=[],
        help=(
            "Use ORCA input generation for selected next-step stages: TS, SP, or ALL. "
            "May be repeated or given as a comma-separated list."
        ),
    )
    parser.add_argument(
        "--orca-out2in",
        default=None,
        help="Path to Gout2ORCAin.py or Goniom2ORCAin.py (optional).",
    )
    parser.add_argument(
        "--orca-template",
        default=None,
        help="Optional ORCA_out2in_info.txt path passed through to the ORCA helper with -t.",
    )

    parser.add_argument("--ts-bond-delta", type=float, default=0.5, help="For TS.out: if any constrained bond satisfies |d(TS_last)-d(CO_last)| > this (A), tag TS as =NoTS. (default 0.5)")
    parser.add_argument("--no-ts-bond-check", action="store_true", help="Disable TS bond-drift check vs corresponding *_CO.out constrained bonds.")

    args = parser.parse_args()

    # nargs="?" next to a nargs="+" positional means `CheckGauss.py --gauss-template *.out`
    # binds the FIRST .out as the template value and drops it from the file list.
    # Fail loudly instead of running one file short against a nonsense template.
    if args.gauss_template and args.gauss_template.lower().endswith(".out"):
        print(
            f"[ERROR] --gauss-template swallowed '{args.gauss_template}' as its value, so that .out "
            f"is also missing from the file list. Write --gauss-template={GAUSS_TEMPLATE_DEFAULT} "
            "(with '='), or put the bare --gauss-template flag AFTER the .out list.",
            file=sys.stderr,
        )
        return 2

    try:
        orca_for_stages = parse_stage_terms(args.orca_for, "--orca-for")
        gauss_template_stages = (
            parse_stage_terms(args.gauss_template_for, "--gauss-template-for")
            if args.gauss_template_for
            else set(GAUSS_TEMPLATE_STAGES_DEFAULT)
        )
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    if args.gauss_template_for and not args.gauss_template:
        print("[ERROR] --gauss-template-for has no effect without --gauss-template.", file=sys.stderr)
        return 2

    files = prefilter_latest_stage_outs(collect_out_files(args.inputs, recursive=args.recursive))
    if not files:
        print("[ERROR] No .out files found from provided inputs.", file=sys.stderr)
        return 2

    log_path = numbered_path("jobs_log.txt")
    logger = TeeLogger(log_path)
    logger.log(f"[LOG] Writing run output to: {log_path.name}")

    out2in_cmd = resolve_gauss_out2in(args.gauss_out2in)
    if args.gauss_template:
        # Resolve ONCE here, against every .out directory, so a template that
        # exists nowhere is a startup error instead of a per-file surprise 200
        # files into a run.  Listing each distinct hit also exposes the
        # "several disagreeing copies" hazard resolve_orca_template() warns about.
        resolved = sorted({
            str(path)
            for path in (resolve_gauss_template(args.gauss_template, f.parent.resolve()) for f in files)
            if path is not None
        })
        if not resolved:
            logger.log(
                f"[ERROR] --gauss-template {args.gauss_template}: not found in any .out directory "
                f"nor in {Path(__file__).resolve().parent}."
            )
            logger.close()
            return 2
        logger.log(f"[MAKE] Gaussian template handoff for stage(s): {', '.join(sorted(gauss_template_stages))}")
        for path in resolved:
            logger.log(f"[MAKE]   template: {path}")
    # The template covers only the selected stages, so Gauss_out2in.py is still
    # needed for the rest -- warn unless the template covers everything.
    if out2in_cmd is None and not (args.gauss_template and gauss_template_stages >= {"TS", "SP"}):
        logger.log("[WARN] Gauss_out2in.py not found (successful jobs will not generate next-step inputs).")

    orca_out2in_cmd = resolve_orca_out2in(args.orca_out2in) if orca_for_stages else None
    if orca_for_stages:
        logger.log(f"[MAKE] ORCA handoff enabled for stage(s): {', '.join(sorted(orca_for_stages))}")
        if orca_out2in_cmd is None and not args.no_write_inputs:
            logger.log("[WARN] ORCA out2in helper not found (selected ORCA stages will not generate next-step inputs).")

    joblog_dirs = [Path(path).expanduser().resolve() for path in args.joblog_dir]
    completed: Dict[str, List[str]] = {"plain": [], "CO": [], "TS": [], "TS_sp": []}
    next_gjfs: List[str] = []
    next_inps: List[str] = []
    failure_cases: List[FailureCase] = []
    failures = 0

    try:
        for out_path in files:
            failures += evaluate_one(
                out_path=out_path,
                tail_n=args.tail,
                freq_threshold=args.freq_threshold,
                joblog_dirs=joblog_dirs,
                out2in_cmd=out2in_cmd,
                gauss_template=args.gauss_template,
                gauss_template_for=gauss_template_stages,
                orca_out2in_cmd=orca_out2in_cmd,
                orca_template=args.orca_template,
                orca_for_stages=orca_for_stages,
                write_inputs=not args.no_write_inputs,
                backup_inputs=args.backup_inputs,
                logger=logger,
                completed=completed,
                next_gjfs=next_gjfs,
                next_inps=next_inps,
                ts_bond_delta=args.ts_bond_delta,
                ts_bond_check=not args.no_ts_bond_check,
                failure_cases=failure_cases,
            )

        log_completed_summary(logger, completed)
        process_failure_groups(
            logger=logger,
            failure_cases=failure_cases,
            write_inputs=not args.no_write_inputs,
            backup_inputs=args.backup_inputs,
            next_gjfs=next_gjfs,
        )

        seen_gauss = set()
        deduped_gjfs: List[str] = []
        for item in next_gjfs:
            if item not in seen_gauss:
                seen_gauss.add(item)
                deduped_gjfs.append(item)

        seen_orca = set()
        deduped_inps: List[str] = []
        for item in next_inps:
            if item not in seen_orca:
                seen_orca.add(item)
                deduped_inps.append(item)

        next_gjf_path = numbered_path("next_gjfs.txt")
        next_gjf_path.write_text(" ".join(deduped_gjfs) + "\n", encoding="utf-8")
        next_inp_path = numbered_path("next_inps.txt")
        next_inp_path.write_text(" ".join(deduped_inps) + "\n", encoding="utf-8")
        logger.log("")
        logger.log(f"[NEXT] Gaussian list -> {next_gjf_path.name}")
        logger.log(" ".join(deduped_gjfs) if deduped_gjfs else "(none)")
        logger.log(f"[NEXT] ORCA list -> {next_inp_path.name}")
        logger.log(" ".join(deduped_inps) if deduped_inps else "(none)")
    finally:
        logger.close()

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
