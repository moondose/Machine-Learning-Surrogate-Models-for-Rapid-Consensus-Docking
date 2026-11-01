#!/usr/bin/env python3
"""Rebuild the lowest-score LeDock pose using trusted SDF template chemistry.

Input layouts:
    ledock_batch.py: work/<original ligand>.mol2/ligand.dok.
    Historical output: recursively located <molecule_id>.dok files.

The lowest LeDock score is selected from all parsed clusters. Bond connectivity
and atom order come from a preprocessed DiffDock SDF template; only coordinates
are replaced with the selected DOK coordinates. Full-atom order is checked first,
then heavy-atom order. Atom correspondence uses the original element-sequence
check and assumes the template and LeDock input preserve the same atom order.
No coordinate fitting or geometry optimization is performed. GNINA templates
can optionally cross-check chemistry. Unusable molecules are excluded.

Requires Python 3.10+, RDKit and tqdm.
Example:
    python preprocess_ledock.py --base-dir /path/to/ledock_output \
        --template-dir /path/to/prepared/sdf \
        --gnina-template-dir /path/to/gnina_preprocessed/molecules \
        --output-dir /path/to/new_ledock_preprocessed

Templates may be named <molecule_id>.sdf (prepared ligand) or
<molecule_id>_pose0.sdf (preprocessed docking pose). --template-dir selects
them; --dd-template-dir remains an alias. Use the same atom order as the
prepared LeDock ligand. DiffDock predictions are not required. Use --require-gnina-match to
require a matching GNINA chemistry check. Outputs are molecules/*.sdf,
csvs/Results_scoring.csv, template diagnostics, errors.csv and processing counts.
Comments and command-line messages are English.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

from rdkit import Chem, RDLogger
from rdkit.Chem import rdDetermineBonds
from rdkit.Geometry import Point3D
from tqdm import tqdm

#
warnings.filterwarnings("ignore", category=UserWarning)
RDLogger.DisableLog("rdApp.warning")



ELEMENTS = {
    "H", "HE", "LI", "BE", "B", "C", "N", "O", "F", "NE", "NA", "MG",
    "AL", "SI", "P", "S", "CL", "AR", "K", "CA", "SC", "TI", "V", "CR",
    "MN", "FE", "CO", "NI", "CU", "ZN", "GA", "GE", "AS", "SE", "BR",
    "KR", "RB", "SR", "Y", "ZR", "NB", "MO", "TC", "RU", "RH", "PD",
    "AG", "CD", "IN", "SN", "SB", "TE", "I", "XE",
}


@dataclass
class DokPose:
    score: float
    pose_index: int
    symbols: list[str]
    coords: list[tuple[float, float, float]]


def normalize_element(atom_name: str, element_field: str = "") -> str:
    """Return a normal RDKit element symbol from PDB/DOK atom fields."""
    raw = re.sub(r"[^A-Za-z]", "", element_field or "").upper()
    if raw in ELEMENTS:
        return raw[0] + raw[1:].lower()

    raw = re.sub(r"[^A-Za-z]", "", atom_name or "").upper()
    if not raw:
        raise ValueError(f"Cannot infer element from atom name {atom_name!r}")

    if len(raw) >= 2 and raw[:2] in ELEMENTS:
        elem = raw[:2]
    elif raw[0] in ELEMENTS:
        elem = raw[0]
    else:
        raise ValueError(f"Unsupported element inferred from atom name {atom_name!r}")
    return elem[0] + elem[1:].lower()


def parse_atom_line(line: str) -> tuple[str, tuple[float, float, float]]:
    """Parse one PDB-like ATOM/HETATM line from a LeDock DOK file."""
    atom_name = line[12:16].strip()
    element_field = line[76:78].strip() if len(line) >= 78 else ""

    try:
        x = float(line[30:38])
        y = float(line[38:46])
        z = float(line[46:54])
    except ValueError:
        parts = line.split()
        if len(parts) < 8:
            raise ValueError(f"Cannot parse coordinates from line: {line.rstrip()}")
        atom_name = atom_name or parts[2]
        x, y, z = map(float, parts[-3:])

    return normalize_element(atom_name, element_field), (x, y, z)


def parse_ledock_dok(dok_path: Path) -> list[DokPose]:
    """Parse all poses from a LeDock DOK file."""
    poses = []
    symbols = []
    coords = []
    current_score = None
    pose_index = -1

    score_re = re.compile(r"Score:\s*([-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?)")

    with dok_path.open("r", errors="ignore") as handle:
        for line in handle:
            if "REMARK Cluster" in line and "Score:" in line:
                if current_score is not None and symbols:
                    poses.append(DokPose(current_score, pose_index, symbols, coords))
                pose_index += 1
                match = score_re.search(line)
                current_score = float(match.group(1)) if match else None
                symbols = []
                coords = []
                continue

            if current_score is not None and line.startswith(("ATOM", "HETATM")):
                symbol, xyz = parse_atom_line(line)
                symbols.append(symbol)
                coords.append(xyz)
                continue

            if line.startswith("END"):
                if current_score is not None and symbols:
                    poses.append(DokPose(current_score, pose_index, symbols, coords))
                current_score = None
                symbols = []
                coords = []

    if current_score is not None and symbols:
        poses.append(DokPose(current_score, pose_index, symbols, coords))

    return poses


def load_template(path: Path) -> Chem.Mol:
    supplier = Chem.SDMolSupplier(str(path), removeHs=False, sanitize=False)
    mol = supplier[0] if supplier and len(supplier) else None
    if mol is None:
        raise ValueError(f"Could not read template SDF: {path}")
    return mol


def atom_symbols(mol: Chem.Mol, heavy_only: bool = False) -> list[str]:
    symbols = []
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        if heavy_only and symbol == "H":
            continue
        symbols.append(symbol)
    return symbols


def canonical_heavy_smiles(mol: Chem.Mol) -> str | None:
    """Best-effort chemistry fingerprint for DD/GNINA consistency checks."""
    copy = Chem.Mol(mol)
    try:
        Chem.SanitizeMol(copy)
    except Exception:
        try:
            rdDetermineBonds.DetermineBondOrders(copy)
            Chem.SanitizeMol(copy)
        except Exception:
            return None
    heavy = Chem.RemoveHs(copy, sanitize=False)
    try:
        Chem.SanitizeMol(heavy)
    except Exception:
        pass
    return Chem.MolToSmiles(heavy, canonical=True, isomericSmiles=True)


def compare_templates(dd_mol: Chem.Mol, gnina_mol: Chem.Mol | None) -> str:
    if gnina_mol is None:
        return "missing_gnina"

    dd_smiles = canonical_heavy_smiles(dd_mol)
    gnina_smiles = canonical_heavy_smiles(gnina_mol)
    if dd_smiles and gnina_smiles:
        return "match" if dd_smiles == gnina_smiles else "smiles_mismatch"

    dd_heavy = sorted(atom_symbols(dd_mol, heavy_only=True))
    gnina_heavy = sorted(atom_symbols(gnina_mol, heavy_only=True))
    return "formula_match" if dd_heavy == gnina_heavy else "formula_mismatch"


def rebuild_pose(template: Chem.Mol, pose: DokPose) -> tuple[Chem.Mol, str]:
    template_symbols = atom_symbols(template, heavy_only=False)

    if pose.symbols == template_symbols:
        mol = Chem.Mol(template)
        conf = Chem.Conformer(mol.GetNumAtoms())
        conf.Set3D(True)
        for idx, xyz in enumerate(pose.coords):
            conf.SetAtomPosition(idx, Point3D(*xyz))
        mol.RemoveAllConformers()
        mol.AddConformer(conf, assignId=True)
        return mol, "full_atom_match"

    template_heavy = atom_symbols(template, heavy_only=True)
    pose_heavy = [symbol for symbol in pose.symbols if symbol != "H"]
    if pose_heavy != template_heavy:
        raise ValueError(
            "DOK/template atom order mismatch: "
            f"DOK heavy {len(pose_heavy)} atoms != template heavy {len(template_heavy)} atoms"
        )

    heavy_mol = Chem.RemoveHs(Chem.Mol(template), sanitize=False)
    heavy_coords = [coord for symbol, coord in zip(pose.symbols, pose.coords) if symbol != "H"]
    conf = Chem.Conformer(heavy_mol.GetNumAtoms())
    conf.Set3D(True)
    for idx, xyz in enumerate(heavy_coords):
        conf.SetAtomPosition(idx, Point3D(*xyz))
    heavy_mol.RemoveAllConformers()
    heavy_mol.AddConformer(conf, assignId=True)
    return heavy_mol, "heavy_atom_match"


def write_pose(mol: Chem.Mol, output_path: Path, zinc_id: str, score: float, pose_index: int, status: str) -> None:
    mol.SetProp("_Name", zinc_id)
    mol.SetProp("LeDock_score", str(score))
    mol.SetProp("LeDock_pose_index", str(pose_index))
    mol.SetProp("template_coordinate_status", status)
    writer = Chem.SDWriter(str(output_path))
    writer.write(mol)
    writer.close()


def zinc_id_from_dok(path: Path) -> str:
    if path.name == "ligand.dok" and path.parent.suffix.lower() == ".mol2":
        return path.parent.stem
    return path.stem



def failed_batch_ids(base):
    """Read batch completion states before considering saved pose artifacts."""
    report = Path(base) / 'results.csv'
    if not report.is_file():
        return set()
    with report.open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    return {Path(row['ligand']).stem for row in rows if row['status'] != 'success'}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rebuild LeDock SDF poses from DOK coordinates and DD/GNINA template chemistry."
    )
    parser.add_argument("--base-dir", "-base-dir", type=Path, required=True)
    parser.add_argument("--template-dir", "--dd-template-dir", "-dd-template-dir", dest="dd_template_dir", type=Path, required=True,
                        help="Prepared SDF chemistry templates, or preprocessed DiffDock templates")
    parser.add_argument("--gnina-template-dir", "-gnina-template-dir", type=Path)
    parser.add_argument("--output-dir", "-output-dir", type=Path, required=True)
    parser.add_argument("--limit", "-limit", type=int, default=None, help="Optional number of DOK files to process for testing")
    parser.add_argument(
        "--require-gnina-match", "-require-gnina-match",
        action="store_true",
        help="Skip ligands when DD and GNINA template chemistry do not match",
    )
    args = parser.parse_args()
    if not args.base_dir.is_dir() or not args.dd_template_dir.is_dir():
        parser.error("Input and SDF template directories must exist")
    if args.gnina_template_dir is not None and not args.gnina_template_dir.is_dir():
        parser.error("GNINA template directory does not exist")
    if args.require_gnina_match and args.gnina_template_dir is None:
        parser.error("--require-gnina-match requires --gnina-template-dir")
    if args.output_dir.exists():
        parser.error("Choose a new output directory")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")

    mol_out_dir = args.output_dir / "molecules"
    csv_out_dir = args.output_dir / "csvs"
    mol_out_dir.mkdir(parents=True, exist_ok=True)
    csv_out_dir.mkdir(parents=True, exist_ok=True)

    config_path = args.base_dir / 'run_config.json'
    if config_path.is_file():
        config = json.loads(config_path.read_text())
        dok_files = sorted(args.base_dir / 'work' / Path(filename).relative_to(Path(config['ligands'])) / 'ligand.dok'
                           for filename in config['ligand_files'])
    else:
        dok_files = sorted(args.base_dir.rglob("*.dok"))
    failed = failed_batch_ids(args.base_dir)
    if args.limit is not None:
        dok_files = dok_files[: args.limit]

    if not dok_files:
        parser.error("No DOK files found")
    ids = [zinc_id_from_dok(path) for path in dok_files]
    if len(ids) != len(set(ids)):
        parser.error("Duplicate molecule IDs in the input DOK files")
    stats = {
        "processed": 0,
        "written": 0,
        "skipped_existing": 0,
        "missing_dd": 0,
        "missing_gnina": 0,
        "gnina_mismatch": 0,
        "no_pose": 0,
        "atom_mismatch": 0,
        "errors": 0,
    }
    error_rows = []
    scoring_rows = []
    diagnostic_rows = []

    for dok_path in tqdm(dok_files, desc="Rebuilding LeDock poses", ascii=True):
        stats["processed"] += 1
        zinc_id = zinc_id_from_dok(dok_path)
        output_sdf = mol_out_dir / f"{zinc_id}_pose0.sdf"
        if output_sdf.exists():
            stats["skipped_existing"] += 1
            continue

        dd_path = args.dd_template_dir / f"{zinc_id}_pose0.sdf"
        if not dd_path.is_file():
            dd_path = args.dd_template_dir / f"{zinc_id}.sdf"
        gnina_path = (args.gnina_template_dir / f"{zinc_id}_pose0.sdf"
                      if args.gnina_template_dir else None)

        if zinc_id in failed:
            error_rows.append((zinc_id, str(dok_path), "Docking job did not complete successfully"))
            stats["errors"] += 1
            continue
        if not dok_path.is_file():
            error_rows.append((zinc_id, str(dok_path), "Missing DOK output"))
            stats["no_pose"] += 1
            continue
        if not dd_path.exists():
            error_rows.append((zinc_id, str(dok_path), "Missing SDF chemistry template"))
            stats["missing_dd"] += 1
            continue

        try:
            dd_mol = load_template(dd_path)
            gnina_mol = load_template(gnina_path) if gnina_path and gnina_path.exists() else None
            template_status = compare_templates(dd_mol, gnina_mol)

            if template_status == "missing_gnina":
                stats["missing_gnina"] += 1
                if args.require_gnina_match:
                    error_rows.append((zinc_id, str(dok_path), "Missing GNINA template"))
                    continue
            elif template_status not in {"match", "formula_match"}:
                stats["gnina_mismatch"] += 1
                if args.require_gnina_match:
                    error_rows.append((zinc_id, str(dok_path), template_status))
                    continue

            poses = parse_ledock_dok(dok_path)
            poses = [pose for pose in poses if math.isfinite(pose.score)
                     and all(math.isfinite(v) for xyz in pose.coords for v in xyz)]
            if not poses:
                error_rows.append((zinc_id, str(dok_path), "No finite scored pose"))
                stats["no_pose"] += 1
                continue

            best_pose = min(poses, key=lambda pose: pose.score)
            rebuilt_mol, coordinate_status = rebuild_pose(dd_mol, best_pose)
            rebuilt_mol.SetProp("DD_GNINA_template_check", template_status)

            write_pose(rebuilt_mol, output_sdf, zinc_id, best_pose.score, best_pose.pose_index, coordinate_status)
            scoring_rows.append({
                "energy": best_pose.score,
                "rank": 1,
                "software": "leDock",
                "pose_index": best_pose.pose_index,
                "zinc_id": zinc_id,
                "rel_path": f"/molecules/{zinc_id}_pose0.sdf",
            })
            diagnostic_rows.append({
                "zinc_id": zinc_id,
                "coordinate_status": coordinate_status,
                "template_check": template_status,
                "dok_path": str(dok_path),
            })
            stats["written"] += 1

        except ValueError as exc:
            error_rows.append((zinc_id, str(dok_path), str(exc)))
            stats["atom_mismatch"] += 1
        except Exception as exc:
            error_rows.append((zinc_id, str(dok_path), str(exc)))
            stats["errors"] += 1
            print(f"Error processing {dok_path}: {exc}", file=sys.stderr)

    scoring_csv = csv_out_dir / "Results_scoring.csv"
    with scoring_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "energy",
                "rank",
                "software",
                "pose_index",
                "zinc_id",
                "rel_path",
            ],
            delimiter=";",
        )
        for row in scoring_rows:
            writer.writerow(row)

    diagnostics_csv = csv_out_dir / "template_rebuild_diagnostics.csv"
    with diagnostics_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["zinc_id", "coordinate_status", "template_check", "dok_path"],
            delimiter=";",
        )
        writer.writeheader()
        for row in diagnostic_rows:
            writer.writerow(row)

    with (args.output_dir / "errors.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["molecule_id", "dok_path", "reason"])
        writer.writerows(error_rows)

    stats_path = args.output_dir / "preprocessing_stats.txt"
    with stats_path.open("w") as handle:
        for key, value in stats.items():
            handle.write(f"{key}: {value}\n")

    print("\nProcessing complete")
    print(f"Output folder: {args.output_dir}")
    for key, value in stats.items():
        print(f"{key}: {value}")
    return 0 if stats["errors"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
