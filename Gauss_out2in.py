#!/usr/bin/env python3

import os
import sys
import re
import argparse
from collections import deque
from pathlib import Path

parser = argparse.ArgumentParser(description="Convert Gaussian .out file to .xyz or .gjf format.")
parser.add_argument("input", nargs="+", help="Input Gaussian output file(s) (.out)")
parser.add_argument("-f", "--format", choices=["xyz", "gjf"], help="Output format: xyz or gjf", default="gjf")
parser.add_argument("-j", "--job_number", help="The number of the job you want to use from the format file", default=1)
parser.add_argument("-t", "--template", help="Template file: 'Job_infoN <route>' lines plus optional quoted multi-line "
                         "PSEUDO \"...\" (used when the route has gen/genecp/pseudo) and SOLVATION \"...\" "
                         "(used when the route has scrf=(...,Read)) blocks", default="Gauss_out2in_info.txt")
parser.add_argument("-n", "--name_end", help="Add the ending for the input file name (sp, TS, TS_sp...), default is _sp", default="sp")
parser.add_argument("-nns", "--no_name_split", action="store_true", help="Splits the name on _ in the input file", default=False)
parser.add_argument("-s", "--submit", default=False,
                        help="Submission criteria: write as processors-time-memory (p-t-m). i.e.: 20-24-38")
parser.add_argument("--release-frozen", action="store_true", default=False,
                        help="Reset the source's negative (frozen) atom flags to 0 in the generated input. "
                             "OFF by default: clearing them silently changes the optimisation model.")

args = parser.parse_args()

INTEGER_LINE_RE = re.compile(r"^\s*[+-]?\d+(?:\s+[+-]?\d+)+\s*$")
NUMERIC_TOKEN_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[DEde][+-]?\d+)?$")
# opt=modredundant constraint line, e.g. 'B 51 106 F' / 'D 1 2 3 4 F'.  Belongs to the
# constrained optimisation only and is dropped on the way to the TS job.
MODRED_LINE_RE = re.compile(r"^\s*[BADLXYZ](?:\s+-?\d+)+\s+[A-Z](?:\s+\S+)*\s*$", re.IGNORECASE)
# Frozen-atom flag in the column before the coordinates (Gaussian: negative = frozen).
FROZEN_FLAG_RE = re.compile(r"-\d+")
# CheckGauss.flag_out_as_nots() renames a rejected job to '<stem>=NoTS.out' (and
# '<stem>=NoTS2.out' on collision).  Neither the sibling source .gjf nor the
# next-stage input ever carries that tag.
RESULT_TAG_RE = re.compile(r"(?:=[A-Za-z][A-Za-z0-9]*)+$")
# Previous-stage token at the end of a stem, stripped before appending -n.
# Mirrors CheckGauss.expected_next_gjf_name(): '<base>_CO' -> '<base>_TS',
# '<base>_TS' -> '<base>_TS_sp', anything else -> '<stem>_sp'.
STAGE_SUFFIX_RE = re.compile(r"_(?:CO|TS)$", re.IGNORECASE)
# Link 0 resource directives of the source .gjf, carried into the new input.
LINK0_NPROC_RE = re.compile(r"^\s*%(?:nprocshared|nproc)\s*=\s*(\S+)\s*$", re.IGNORECASE)
LINK0_MEM_RE = re.compile(r"^\s*%mem\s*=\s*(\S+)\s*$", re.IGNORECASE)
# Project norm, used only when the source .gjf declares nothing.
DEFAULT_NPROC = "20"
DEFAULT_MEM = "80GB"
# Element card opening a gen/ECP block: 'Ru 0', 'C N F P H O 0'.  Gaussian's gen
# syntax terminates the element list with a literal 0; requiring it keeps MM
# parameter blocks (HrmStr1 CA O2 570.0 1.229) from being mistaken for a basis.
ELEMENT_CARD_RE = re.compile(r"^\s*[A-Za-z]{1,2}(?:\s+[A-Za-z]{1,2})*\s+0\s*$")
# ONIOM layer designations Gaussian accepts after the coordinates.
LAYER_LETTERS = {"H", "M", "L"}

for finput in args.input:
    fformat = args.format
    job_number = args.job_number
    template_file = args.template
    name_end = args.name_end
    no_name_split = args.no_name_split
    submit = args.submit
    release_frozen = args.release_frozen

    def strip_result_tag(stem):
        """'conformer3_CO=NoTS' -> 'conformer3_CO'.  CheckGauss tags a rejected job
        in place, so the .out grows a '=NoTS' suffix that the sibling .gjf it was
        generated from never had.  Both the source-.gjf lookup and the next-stage
        name must be derived from the untagged stem."""
        return RESULT_TAG_RE.sub("", stem)

    def next_input_stem(stem):
        """Stem the -n ending is appended to.  Only the previous stage token is
        dropped, so every other underscore-separated token survives:

            TS1-A_Re_conformer8_CO  -n TS     -> TS1-A_Re_conformer8_TS
            TS1-A_Re_conformer8_TS  -n TS_sp  -> TS1-A_Re_conformer8_TS_sp
            conformer3_CO           -n TS     -> conformer3_TS

        The old rule kept only the FIRST underscore token, so every conformer
        sharing a leading token collapsed onto one output file and overwrote the
        last, and the name never matched CheckGauss.expected_next_gjf_name() ->
        the handoff was reported as failed.  Compare CheckGauss.orca_name_end()."""
        return STAGE_SUFFIX_RE.sub("", stem)

    def read_link0_resources(gjf_path):
        """(%nprocshared, %mem) carried from the source .gjf's Link 0 block.
        Without them Gaussian runs single-threaded on its ~800 MB default."""
        nproc = None
        mem = None
        if gjf_path is not None and Path(gjf_path).exists():
            for line in Path(gjf_path).read_text(encoding="utf-8", errors="replace").splitlines():
                if line.lstrip().startswith("#"):
                    break
                match = LINK0_NPROC_RE.match(line)
                if match and nproc is None:
                    nproc = match.group(1)
                    continue
                match = LINK0_MEM_RE.match(line)
                if match and mem is None:
                    mem = match.group(1)
        return nproc or DEFAULT_NPROC, mem or DEFAULT_MEM

    def check_normal_termination(filepath, num_lines=20):
        with open(filepath, "r") as f:
            last_lines = deque(f, maxlen=num_lines)
        for line in last_lines:
            if "Normal termination of Gaussian" in line:
                return True
        return False

    if not check_normal_termination(finput):
        print(f"[WARNING] Job did not terminate normally: {finput}")
        continue  # Or return / continue, depending on context

    def GetChargeMult(filepath):
        Chrg = None
        Mult = None
        with open(filepath, "r") as f:
            for line in f:
                if "Charge" and "Multiplicity" in line:
                    objects = line.split()
                    Chrg = objects[2]
                    Mult = objects[5]
                    break
        return Chrg, Mult

    def find_charge_mult_idx(lines):
        for idx, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                continue
            if INTEGER_LINE_RE.fullmatch(stripped) and len(stripped.split()) % 2 == 0:
                return idx
        return None

    def find_geometry_block(lines, cm_idx):
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

    def is_numeric_token(token):
        return bool(NUMERIC_TOKEN_RE.fullmatch(token))

    def parse_geometry_template_line(raw_line):
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
            return (tokens[0], tokens[1:start], tokens[start + 3 :])
        return None

    def geometry_templates_from_lines(lines):
        """(templates, offending_line).  offending_line is the first line that could not
        be parsed, so the caller can name it instead of silently falling back to plain
        element+xyz and downgrading an ONIOM job to a plain one."""
        templates = []
        for line in lines:
            parsed = parse_geometry_template_line(line)
            if parsed is None:
                return [], line.rstrip("\n")
            templates.append(parsed)
        return templates, None

    def split_tail_sections(lines):
        """Blank-line-separated sections following the geometry block, in file order."""
        sections = []
        current = []
        for line in lines:
            if line.strip():
                current.append(line.rstrip("\n"))
            elif current:
                sections.append(current)
                current = []
        if current:
            sections.append(current)
        return sections

    def is_modredundant_section(section):
        return bool(section) and all(MODRED_LINE_RE.match(line) for line in section)

    def is_basis_section(section):
        """A gen/genecp basis block ('****'-terminated) or a BARE ECP block, which
        carries no '****' at all:

            Ru 0
            sdd

        Missing the bare form let the source's ECP be copied AND the template's
        PSEUDO block appended, giving Gaussian two conflicting ECPs for Ru."""
        if any(line.strip() == "****" for line in section):
            return True
        return len(section) >= 2 and bool(ELEMENT_CARD_RE.match(section[0]))

    def keep_tail_sections(sections):
        """ONIOM tail sections the next job still needs (connectivity, MM parameters,
        ONIOM additional-parameter blocks).

        The CO's modredundant constraints and the source's own basis/ECP sections are
        dropped.  The basis is never reusable either way: when the new route asks for
        gen/genecp the template's PSEUDO block supplies it, and when it does not, a
        stray Gen/ECP block is appended to a job that declares no Gen keyword.  Each
        such section is skipped INDIVIDUALLY -- the old 'break' also discarded every
        trailing MM/HrmStr parameter block that legitimately followed the basis."""
        kept = []
        for section in sections:
            if is_modredundant_section(section):
                continue
            if is_basis_section(section):
                continue
            kept.append(section)
        return kept

    def oniom_fail(fout, scratch_path, message):
        """The single exit door once the output file is open.  Everything is written
        to a scratch file, so bailing out here removes only the scratch and leaves any
        existing good input untouched -- a re-run that trips a new fail-loud path can
        no longer destroy the previous *_TS.gjf."""
        fout.close()
        Path(scratch_path).unlink(missing_ok=True)
        sys.exit(message)

    def count_frozen_flags(templates):
        return sum(
            1
            for _, prefix_tokens, _ in templates
            for token in prefix_tokens
            if FROZEN_FLAG_RE.fullmatch(token)
        )

    def template_element(label):
        """Element symbol from an ONIOM atom label: 'C-CA--0.2500' -> 'C', 'Ru' -> 'Ru'.
        MM type and partial charge follow the symbol after a '-'."""
        return label.split("(")[0].split("-")[0].strip()

    def oniom_annotation_problem(templates):
        """Refuse to dress a non-ONIOM source up as an ONIOM input.  Every atom of a
        Gaussian ONIOM job carries a layer letter (H/M/L) after its coordinates; if the
        source .gjf has none, the atom-count guard still passes and Gaussian silently
        promotes all atoms to the high layer -- a full high-level calculation billed as
        ONIOM, exit code 0."""
        for index, (label, _, suffix_tokens) in enumerate(templates, start=1):
            if not suffix_tokens or suffix_tokens[0].upper() not in LAYER_LETTERS:
                return (
                    f"atom {index} ({label}) carries no ONIOM layer letter (H/M/L) after its "
                    f"coordinates"
                )
        return None

    def element_order_problem(coords, templates):
        """Atom COUNTS matching is not enough: if the source .gjf lists the same atoms
        in a different order, every layer letter, MM type and link-atom spec is attached
        to the wrong atom and the job still runs."""
        for index, ((atomic_number, _, _, _), (label, _, _)) in enumerate(zip(coords, templates), start=1):
            out_symbol = code.get(str(atomic_number), "X")
            gjf_symbol = template_element(label)
            if out_symbol.upper() != gjf_symbol.upper():
                return (
                    f"atom {index} is {out_symbol} in the output but {gjf_symbol} "
                    f"({label}) in the template"
                )
        return None

    def release_frozen_atoms(templates):
        """A constrained optimisation freezes atoms with a negative flag in the column
        before the coordinates.  That constraint belongs to the CO, not to the TS search,
        so reset it to 0 (free).  The column itself is kept so the ONIOM layer and
        link-atom columns stay where they were."""
        released = 0
        thawed = []
        for label, prefix_tokens, suffix_tokens in templates:
            new_prefix = list(prefix_tokens)
            for i, token in enumerate(new_prefix):
                if FROZEN_FLAG_RE.fullmatch(token):
                    new_prefix[i] = "0"
                    released += 1
            thawed.append((label, new_prefix, suffix_tokens))
        return thawed, released

    def load_source_gjf_template(gjf_path):
        """(charge_mult_line, geom_templates, tail_sections, gjf_path, problem).

        gjf_path is derived from the UNTAGGED .out stem by the caller, so a job that
        CheckGauss renamed to '<stem>=NoTS.out' still finds '<stem>.gjf'."""
        gjf_path = Path(gjf_path)
        if not gjf_path.exists():
            return None, None, [], gjf_path, f"{gjf_path.name} does not exist"

        lines = gjf_path.read_text(encoding="utf-8", errors="replace").splitlines(True)
        cm_idx = find_charge_mult_idx(lines)
        if cm_idx is None:
            return None, None, [], gjf_path, f"no charge/multiplicity line found in {gjf_path.name}"

        charge_mult_line = lines[cm_idx].strip()
        geom_start, geom_end = find_geometry_block(lines, cm_idx)
        if geom_start is None or geom_end is None:
            return charge_mult_line, None, [], gjf_path, f"no geometry block found in {gjf_path.name}"

        templates, bad_line = geometry_templates_from_lines(lines[geom_start:geom_end])
        if bad_line is not None:
            return charge_mult_line, None, [], gjf_path, f"unparseable geometry line in {gjf_path.name}: {bad_line!r}"

        return charge_mult_line, templates, split_tail_sections(lines[geom_end:]), gjf_path, None

    def openTemplate(name="Gauss_out2in_info.txt", pseudo=False):
        """
        Reads:
          - Job_info{job_number} route line (either on same line, or the next non-empty line)
          - Optional PSEUDO " ... " block (can span multiple lines until closing quote);
            appended when the route has gen/genecp/pseudo
          - Optional SOLVATION " ... " block, same syntax, either order; appended
            (after PSEUDO) when the route has Read inside scrf=(...), e.g.
            scrf=(smd,solvent=Generic,Read)

        Returns: (route_info, pseudo_block_or_None, solvation_block_or_None)
        """
        route_info = None

        # Allow -j to be given as string; normalize to int-like string for matching
        job_tag = f"Job_info{job_number}"

        with open(name, "r") as file:
            lines = file.readlines()

        # ---- Find route_info ----
        for i, line in enumerate(lines):
            if job_tag in line:
                # If route is on same line after the first token, use it
                parts = line.split(None, 1)
                if len(parts) > 1 and parts[1].strip():
                    route_info = parts[1].rstrip("\n")
                else:
                    # Otherwise, take the next non-empty, non-comment line
                    j = i + 1
                    while j < len(lines):
                        nxt = lines[j].strip("\n")
                        if nxt.strip() and not nxt.lstrip().startswith("#"):
                            route_info = nxt.rstrip("\n")
                            break
                        j += 1
                break

        return route_info, quoted_block(lines, "PSEUDO"), quoted_block(lines, "SOLVATION")

    def quoted_block(lines, keyword):
        """Text between the first pair of double quotes at/after the first line that
        starts with `keyword` (case-insensitive); may span lines.  None if absent.
        Stops at its OWN closing quote, so PSEUDO and SOLVATION blocks may appear in
        either order without one swallowing the other."""
        block = None
        start_idx = None
        for i, line in enumerate(lines):
            if re.match(rf"^\s*{keyword}\b", line, flags=re.IGNORECASE):
                start_idx = i
                break

        if start_idx is not None:
            # From the keyword line onward, find the first quote, then collect until the matching closing quote.
            collected = []
            in_quotes = False
            found_first_quote = False

            for i in range(start_idx, len(lines)):
                L = lines[i]

                if not found_first_quote:
                    # Find first quote on this line
                    qpos = L.find('"')
                    if qpos != -1:
                        found_first_quote = True
                        in_quotes = True
                        # take after first quote
                        remainder = L[qpos + 1 :]
                        # If the closing quote is also on the same remainder:
                        qpos2 = remainder.find('"')
                        if qpos2 != -1:
                            collected.append(remainder[:qpos2])
                            in_quotes = False
                            break
                        else:
                            collected.append(remainder)
                    # else: keep scanning
                else:
                    if in_quotes:
                        qpos = L.find('"')
                        if qpos != -1:
                            collected.append(L[:qpos])
                            in_quotes = False
                            break
                        else:
                            collected.append(L)

            if found_first_quote:
                block = "".join(collected)
                # Preserve internal newlines; remove only leading/trailing blank space/newlines
                block = block.strip("\n")

        return block

    def getEnergy(structure):
        for line in structure.split("\n"):
            if line.startswith(" SCF Done:"):
                arr = line.split("=")
                return float(re.split(" +", arr[1].strip())[0])
        return 1000.0

    infoBlock = ""
    optimized = False
    optimized_structure = ""
    with open(finput, "r") as fin:
        isStructure = True
        isInfo = True
        structures = []
        currentStructure = ""
        for line in fin:
            if line.startswith(" GradGrad"):
                if isInfo:
                    isInfo = False
                if currentStructure != "":
                    structures.append((getEnergy(currentStructure), currentStructure))
                    currentStructure = ""
                isStructure = not isStructure
            elif isInfo:
                infoBlock += line
            elif isStructure:
                currentStructure += line
            else:
                if line.find("Optimized") != -1:
                    optimized = True

        if optimized:
            optimized_structure = currentStructure
        else:
            if currentStructure != "":
                structures.append((getEnergy(currentStructure), currentStructure))
            structures = sorted(structures, key=lambda item: item[0])
            optimized_structure = structures[0][1]

    def findInList(dataList, target):
        for i in range(0, len(dataList)):
            if dataList[i].find(target) != -1:
                return i
        return -1

    def getCoordinates(fpath):
        # read all lines
        print(f"Extracting coords from {fpath}")
        with open(fpath) as f:
            lines = f.readlines()

        header = " Center     Atomic      Atomic             Coordinates (Angstroms)"
        # find all occurrences of the header, take the last one
        idxs = [i for i, L in enumerate(lines) if header in L]
        if not idxs:
            raise RuntimeError(f"No final coordinates block found in {fpath}")
        start = idxs[-1]

        # skip header + two sub-headers + separator
        data_lines = []
        for L in lines[start + 3 :]:
            if L.strip().startswith("-----"):
                break
            data_lines.append(L)

        coords = []
        for L in data_lines:
            parts = L.split()
            # parts = [center, atomic_num, atomic_type, x, y, z]
            atomic_num = int(parts[1])
            x, y, z = float(parts[3]), float(parts[4]), float(parts[5])
            coords.append((atomic_num, x, y, z))

        return coords

    code = {"1": "H", "2": "He", "3": "Li", "4": "Be", "5": "B",
    "6": "C", "7": "N", "8": "O",  "9": "F", "10": "Ne",
    "11": "Na", "12": "Mg", "13": "Al", "14": "Si", "15": "P",
    "16": "S",  "17": "Cl", "18": "Ar", "19": "K",  "20": "Ca",
    "21": "Sc", "22": "Ti", "23": "V",  "24": "Cr", "25": "Mn",
    "26": "Fe", "27": "Co", "28": "Ni", "29": "Cu", "30": "Zn",
    "31": "Ga", "32": "Ge", "33": "As", "34": "Se", "35": "Br",
    "36": "Kr", "37": "Rb", "38": "Sr", "39": "Y",  "40": "Zr",
    "41": "Nb", "42": "Mo", "43": "Tc", "44": "Ru", "45": "Rh",
    "46": "Pd", "47": "Ag", "48": "Cd", "49": "In", "50": "Sn",
    "51": "Sb", "52": "Te", "53": "I",  "54": "Xe", "55": "Cs",
    "56": "Ba", "57": "La", "58": "Ce", "59": "Pr", "60": "Nd",
    "61": "Pm", "62": "Sm", "63": "Eu", "64": "Gd", "65": "Tb",
    "66": "Dy", "67": "Ho", "68": "Er", "69": "Tm", "70": "Yb",
    "71": "Lu", "72": "Hf", "73": "Ta", "74": "W",  "75": "Re",
    "76": "Os", "77": "Ir", "78": "Pt", "79": "Au", "80": "Hg",
    "81": "Tl", "82": "Pb", "83": "Bi", "84": "Po", "85": "At",
    "86": "Rn", "87": "Fr", "88": "Ra", "89": "Ac", "90": "Th",
    "91": "Pa", "92": "U",  "93": "Np", "94": "Pu", "95": "Am",
    "96": "Cm", "97": "Bk", "98": "Cf", "99": "Es", "100": "Fm",
    "101": "Md", "102": "No", "103": "Lr", "104": "Rf", "105": "Db",
    "106": "Sg", "107": "Bh", "108": "Hs", "109": "Mt", "110": "Ds",
    "111": "Rg", "112": "Uub", "113": "Uut", "114": "Uuq", "115": "Uup",
    "116": "Uuh", "117": "Uus", "118": "Uuo"}

    def format_plain_coords(coords):
        output = []
        for atomic_number, x_coord, y_coord, z_coord in coords:
            symbol = code.get(str(atomic_number), "X")
            output.append(f" {symbol:<2s}{x_coord:>18.8f}{y_coord:>13.8f}{z_coord:>13.8f}\n")
        return output

    def format_coords_from_templates(coords, templates):
        if not templates or len(coords) != len(templates):
            return format_plain_coords(coords)

        output = []
        for (atomic_number, x_coord, y_coord, z_coord), template in zip(coords, templates):
            label, prefix_tokens, suffix_tokens = template
            symbol = code.get(str(atomic_number), label or "X")
            line = label if label else symbol
            if prefix_tokens:
                line += " " + " ".join(prefix_tokens)
            line += f" {x_coord:>16.8f} {y_coord:>16.8f} {z_coord:>16.8f}"
            if suffix_tokens:
                line += " " + " ".join(suffix_tokens)
            output.append(line + "\n")
        return output

    def main():
        # --- output / source names -------------------------------------------------
        # Both are derived from the UNTAGGED stem so that a CheckGauss-renamed
        # '<stem>=NoTS.out' still resolves to '<stem>.gjf' and to the same next-stage
        # name CheckGauss.expected_next_gjf_name() will look for.
        input_path = Path(finput)
        source_stem = strip_result_tag(input_path.stem)
        prefix_stem = source_stem if no_name_split else next_input_stem(source_stem)
        prefix = str(input_path.with_name(prefix_stem))

        chk = prefix + f"_{name_end}.chk"
        foutput = prefix + f"_{name_end}." + fformat
        source_gjf_path = input_path.with_name(source_stem + ".gjf")

        # Written to scratch and moved into place only on success, so a failed re-run
        # cannot destroy an existing good input.
        out_path = Path(foutput)
        scratch_path = out_path.with_name(out_path.name + f".tmp{os.getpid()}")

        fout = open(scratch_path, "w")
        try:
            # extract final optimized coords directly from the .out file
            atoms = getCoordinates(finput)

            # print format specific headers
            route_info = None
            pseudo_block = None
            solvation_block = None
            preserve_oniom_format = False
            source_charge_mult_line = None
            source_geom_templates = None
            source_tail_sections = []
            if fformat == "xyz":
                fout.write(str(len(atoms)) + "\n\n")
            else:
                if template_file:
                    route_info, pseudo_block, solvation_block = openTemplate(template_file)
                    Charge, Multiplicity = GetChargeMult(finput)
                    (
                        source_charge_mult_line,
                        source_geom_templates,
                        raw_tail_sections,
                        source_gjf_path,
                        source_problem,
                    ) = load_source_gjf_template(source_gjf_path)
                    if not route_info:
                        oniom_fail(fout, scratch_path, "Missing the submission info")
                    route_uses_oniom = bool(re.search(r"\boniom\s*\(", route_info, flags=re.IGNORECASE))
                    # ONIOM annotation (layer, link atom, MM type/charge) lives in the source
                    # .gjf only -- a .out gives element+xyz.  gen/genecp is a separate concern
                    # (it gates the PSEUDO tail block below), so it must not gate this.
                    preserve_oniom_format = route_uses_oniom
                    if preserve_oniom_format:
                        if source_problem:
                            oniom_fail(fout, scratch_path,
                                f"Route line requests ONIOM, but the per-atom ONIOM annotation cannot be "
                                f"recovered: {source_problem}.  Refusing to write a plain-Gaussian input."
                            )
                        if len(source_geom_templates) != len(atoms):
                            oniom_fail(fout, scratch_path,
                                f"Route line requests ONIOM, but {source_gjf_path.name} has "
                                f"{len(source_geom_templates)} atoms while {Path(finput).name} has "
                                f"{len(atoms)}.  Refusing to write a plain-Gaussian input."
                            )
                        layer_problem = oniom_annotation_problem(source_geom_templates)
                        if layer_problem:
                            oniom_fail(fout, scratch_path,
                                f"Route line requests ONIOM, but {source_gjf_path.name} is not ONIOM-annotated: "
                                f"{layer_problem}.  Refusing to write an input whose atoms would all default "
                                f"to the high layer."
                            )
                        order_problem = element_order_problem(atoms, source_geom_templates)
                        if order_problem:
                            oniom_fail(fout, scratch_path,
                                f"{source_gjf_path.name} and {Path(finput).name} disagree on atom order: "
                                f"{order_problem}.  Refusing to attach the ONIOM annotation to the wrong atoms."
                            )
                        if release_frozen:
                            source_geom_templates, n_released = release_frozen_atoms(source_geom_templates)
                            if n_released:
                                print(f"[NOTE] Released {n_released} frozen-atom flag(s) from {source_gjf_path.name} "
                                      f"for the {name_end} job (--release-frozen)", file=sys.stderr)
                        else:
                            n_frozen = count_frozen_flags(source_geom_templates)
                            if n_frozen:
                                print(f"[WARN] Carrying {n_frozen} frozen-atom flag(s) from {source_gjf_path.name} "
                                      f"into the {name_end} job; pass --release-frozen to reset them to 0",
                                      file=sys.stderr)
                        # Only an ONIOM follow-up reuses the source tail; a plain job
                        # (e.g. the single point) must not inherit the CO's sections.
                        source_tail_sections = keep_tail_sections(raw_tail_sections)
                        charge_mult_line = source_charge_mult_line
                    else:
                        if not Charge or not Multiplicity:
                            oniom_fail(fout, scratch_path, "Missing the charge and/or multiplicity info")
                        charge_mult_line = Charge + " " + Multiplicity
                    nproc, mem = read_link0_resources(source_gjf_path)
                    fout.write(
                        f"%nprocshared={nproc}\n%mem={mem}\n%chk={chk}\n"
                        + route_info + "\n\nTitle Card Required\n\n" + charge_mult_line + "\n"
                    )

            # write geometry
            if fformat == "gjf" and route_info and preserve_oniom_format:
                coord_lines = format_coords_from_templates(atoms, source_geom_templates)
            else:
                coord_lines = format_plain_coords(atoms)

            for atom_line in coord_lines:
                fout.write(atom_line)

            # ONIOM tail sections carried over from the source .gjf (connectivity, MM
            # parameters); they precede the gen/pseudo block Gaussian expects last.
            for section in source_tail_sections:
                fout.write("\n" + "\n".join(section) + "\n")

            # Independent tail blocks, each after 1 blank line, PSEUDO first and SOLVATION
            # last: route has pseudo/genecp/gen -> PSEUDO; route has Read inside scrf(...)
            # -> SOLVATION.  3 additional blank lines after the last one.
            if fformat == "gjf" and route_info:
                tail_blocks = []
                if re.search(r"\b(pseudo|genecp|gen)\b", route_info, flags=re.IGNORECASE):
                    if pseudo_block is None or not pseudo_block.strip():
                        oniom_fail(fout, scratch_path,
                            "Route line requests pseudo/gen/genecp, but no PSEUDO \"...\" block was found "
                            "in the template file.  Refusing to write an input with no basis set."
                        )
                    tail_blocks.append(pseudo_block)
                if re.search(r"\bscrf\s*=?\s*\([^)]*\bread\b", route_info, flags=re.IGNORECASE):
                    if solvation_block is None or not solvation_block.strip():
                        oniom_fail(fout, scratch_path,
                            "Route line requests scrf=(...,Read), but no SOLVATION \"...\" block was found "
                            "in the template file.  Refusing to write an input with no solvent parameters."
                        )
                    tail_blocks.append(solvation_block)
                if tail_blocks:
                    fout.write("\n" + "\n\n".join(b.rstrip() for b in tail_blocks) + "\n\n\n\n")
                else:
                    fout.write("\n\n")
        except BaseException:
            # SystemExit from oniom_fail has already cleaned up; anything else has not.
            if not fout.closed:
                fout.close()
            scratch_path.unlink(missing_ok=True)
            raise

        fout.close()
        os.replace(scratch_path, out_path)

    if __name__ == "__main__":
        main()
