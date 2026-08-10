"""Structure-input parsing and validation for MammalTox."""

from __future__ import annotations

from io import BytesIO
from functools import lru_cache
from pathlib import Path

import pandas as pd
from rdkit import Chem

from .descriptor_calculation import canonicalize_smiles


COMPOUND_COLUMNS = [
    "Compound_ID",
    "Compound_Name",
    "Input_SMILES",
    "Source_Error",
]

SMILES_COLUMN_NAMES = {
    "smiles",
    "canonical smiles",
    "canonical_smiles",
    "canonical-smiles",
}
ID_COLUMN_NAMES = {
    "id",
    "taid",
    "compound id",
    "compound_id",
    "compound-id",
}
NAME_COLUMN_NAMES = {
    "name",
    "compound name",
    "compound_name",
    "compound-name",
}

DOWNLOAD_EXAMPLE_STRUCTURES = (
    ("Example_001", "CCO"),
    ("Example_002", "CC(=O)O"),
    ("Example_003", "c1ccccc1"),
    ("Example_004", "Oc1ccccc1"),
    ("Example_005", "Nc1ccccc1"),
    ("Example_006", "Clc1ccccc1"),
    ("Example_007", "Fc1ccccc1"),
    ("Example_008", "C1CCCCC1"),
    ("Example_009", "n1ccccc1"),
    ("Example_010", "CCN(CC)CC"),
    ("Example_011", "CC(=O)Nc1ccccc1"),
    ("Example_012", "O=C(O)c1ccccc1"),
)


def _empty_compounds(error: str) -> pd.DataFrame:
    return pd.DataFrame(
        [{"Compound_ID": "", "Compound_Name": "", "Input_SMILES": "", "Source_Error": error}],
        columns=COMPOUND_COLUMNS,
    )


def _matching_column(df: pd.DataFrame, accepted: set[str], contains: str | None = None) -> str | None:
    for column in df.columns:
        normalized = str(column).strip().lower()
        if normalized in accepted or (contains and contains in normalized):
            return str(column)
    return None


def _normalize_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return _empty_compounds("Input file is empty")

    smiles_column = _matching_column(df, SMILES_COLUMN_NAMES, contains="smiles")
    id_column = _matching_column(df, ID_COLUMN_NAMES)
    name_column = _matching_column(df, NAME_COLUMN_NAMES)
    if smiles_column is None:
        rows = []
        for index in range(len(df)):
            rows.append(
                {
                    "Compound_ID": str(df.iloc[index][id_column]) if id_column else f"Compound_{index + 1:03d}",
                    "Compound_Name": str(df.iloc[index][name_column]) if name_column else "",
                    "Input_SMILES": "",
                    "Source_Error": "Missing SMILES column",
                }
            )
        return pd.DataFrame(rows or _empty_compounds("Missing SMILES column"), columns=COMPOUND_COLUMNS)

    rows = []
    for index, row in df.reset_index(drop=True).iterrows():
        identifier = row.get(id_column, "") if id_column else ""
        name = row.get(name_column, "") if name_column else ""
        rows.append(
            {
                "Compound_ID": "" if pd.isna(identifier) else str(identifier).strip(),
                "Compound_Name": "" if pd.isna(name) else str(name).strip(),
                "Input_SMILES": "" if pd.isna(row[smiles_column]) else str(row[smiles_column]).strip(),
                "Source_Error": "",
            }
        )
    return pd.DataFrame(rows, columns=COMPOUND_COLUMNS)


def parse_delimited_table(data: bytes, suffix: str) -> pd.DataFrame:
    """Parse CSV or XLSX bytes into the standard compound columns."""
    try:
        if suffix == ".csv":
            table = pd.read_csv(BytesIO(data))
        else:
            table = pd.read_excel(BytesIO(data))
    except pd.errors.EmptyDataError:
        return _empty_compounds("Input file is empty")
    except Exception as exc:
        return _empty_compounds(f"Input file could not be read: {exc}")
    return _normalize_table(table)


def parse_smi_text(text: str, source_prefix: str = "Compound") -> pd.DataFrame:
    """Parse one SMILES per line with an optional identifier/name after it."""
    rows = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(maxsplit=1)
        if not parts or not parts[0].strip():
            rows.append(
                {
                    "Compound_ID": f"{source_prefix}_{line_number:03d}",
                    "Compound_Name": "",
                    "Input_SMILES": "",
                    "Source_Error": f"Malformed SMI/TXT row at line {line_number}",
                }
            )
            continue
        optional_name = parts[1].strip() if len(parts) == 2 else ""
        rows.append(
            {
                "Compound_ID": optional_name or f"{source_prefix}_{line_number:03d}",
                "Compound_Name": optional_name,
                "Input_SMILES": parts[0].strip(),
                "Source_Error": "",
            }
        )
    if not rows:
        return _empty_compounds("Input file is empty")
    return pd.DataFrame(rows, columns=COMPOUND_COLUMNS)


def validate_single_smiles_input(text: str) -> tuple[str, str]:
    """Validate exactly one SMILES and return canonical isomeric SMILES plus status."""
    value = str(text or "").strip()
    if not value:
        return "", "empty"

    lines = [line.strip() for line in value.splitlines() if line.strip()]
    if len(lines) != 1:
        return "", "multiple"

    tokens = lines[0].split()
    if len(tokens) > 1:
        valid_tokens = sum(canonicalize_smiles(token)[0] is not None for token in tokens)
        return "", "multiple" if valid_tokens > 1 else "Invalid SMILES"

    canonical, _, error = canonicalize_smiles(lines[0])
    return canonical or "", error or "valid"


def smiles_to_editor_molblock(smiles: str) -> tuple[str, str | None]:
    """Convert one valid SMILES into Ketcher's editable Molfile representation."""
    canonical, molecule, error = canonicalize_smiles(smiles)
    if molecule is None or canonical is None:
        return "", error or "Invalid SMILES"
    try:
        return Chem.MolToMolBlock(molecule, includeStereo=True), None
    except Exception:
        return "", "Structure could not be converted for the editor"


def editor_molblock_to_smiles(molblock: str) -> tuple[str, str | None]:
    """Convert Ketcher's Molfile output to canonical isomeric SMILES."""
    value = str(molblock or "")
    if not value.strip():
        return "", "empty"
    try:
        molecule = Chem.MolFromMolBlock(
            value,
            sanitize=True,
            removeHs=True,
            strictParsing=True,
        )
        if molecule is None:
            return "", "Drawn structure could not be parsed"
        return Chem.MolToSmiles(
            molecule,
            canonical=True,
            isomericSmiles=True,
        ), None
    except Exception:
        return "", "Drawn structure could not be parsed"


def parse_sdf(data: bytes, source_prefix: str = "Compound") -> pd.DataFrame:
    """Parse one or more SDF records and retain unreadable records as failures."""
    try:
        from rdkit import Chem
    except Exception as exc:
        return _empty_compounds(f"SDF support is unavailable: {exc}")

    try:
        supplier = Chem.ForwardSDMolSupplier(BytesIO(data), sanitize=True, removeHs=True)
        records = list(supplier)
    except Exception as exc:
        return _empty_compounds(f"Unable to parse SDF file: {exc}")

    if not records:
        return _empty_compounds("SDF file contains no molecular records")

    rows = []
    for index, molecule in enumerate(records, start=1):
        fallback_id = f"{source_prefix}_{index:03d}"
        if molecule is None:
            rows.append(
                {
                    "Compound_ID": fallback_id,
                    "Compound_Name": "",
                    "Input_SMILES": "",
                    "Source_Error": "Unable to parse SDF record",
                }
            )
            continue

        properties = molecule.GetPropsAsDict(includePrivate=False, includeComputed=False)
        name = molecule.GetProp("_Name").strip() if molecule.HasProp("_Name") else ""
        identifier = ""
        for key in ["Compound_ID", "COMPOUND_ID", "ID", "TAID", "PUBCHEM_COMPOUND_CID"]:
            if key in properties and str(properties[key]).strip():
                identifier = str(properties[key]).strip()
                break
        try:
            smiles = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
        except Exception:
            smiles = ""
        rows.append(
            {
                "Compound_ID": identifier or name or fallback_id,
                "Compound_Name": name,
                "Input_SMILES": smiles,
                "Source_Error": "" if smiles else "Unable to convert SDF record to SMILES",
            }
        )
    return pd.DataFrame(rows, columns=COMPOUND_COLUMNS)


def parse_uploaded_file(filename: str, data: bytes) -> pd.DataFrame:
    """Parse a supported MammalTox input file from browser-uploaded bytes."""
    suffix = Path(filename).suffix.lower()
    prefix = Path(filename).stem or "Compound"
    if not data:
        return _empty_compounds("Input file is empty")
    if suffix in {".csv", ".xlsx"}:
        return parse_delimited_table(data, suffix)
    if suffix in {".smi", ".txt"}:
        try:
            text = data.decode("utf-8-sig")
        except UnicodeDecodeError:
            return _empty_compounds("SMI/TXT file is not valid UTF-8 text")
        return parse_smi_text(text, prefix)
    if suffix == ".sdf":
        return parse_sdf(data, prefix)
    return _empty_compounds(f"Unsupported file format: {suffix or 'unknown'}")


def validate_compound_inputs(compounds: pd.DataFrame) -> pd.DataFrame:
    """Validate structures with RDKit and report canonical SMILES and duplicates."""
    working = compounds.copy().reset_index(drop=True)
    for column in COMPOUND_COLUMNS:
        if column not in working.columns:
            working[column] = ""

    canonical_values: list[str] = []
    errors: list[str] = []
    for _, row in working.iterrows():
        source_error = str(row.get("Source_Error", "") or "").strip()
        if source_error:
            canonical_values.append("")
            errors.append(source_error)
            continue
        canonical, _, error = canonicalize_smiles(row.get("Input_SMILES", ""))
        canonical_values.append(canonical or "")
        errors.append(error or "")

    working["Canonical_SMILES"] = canonical_values
    working["Input_Error"] = errors
    first_seen: dict[str, str] = {}
    duplicate_of: list[str] = []
    for index, canonical in enumerate(canonical_values):
        if not canonical or errors[index]:
            duplicate_of.append("")
            continue
        identifier = str(working.loc[index, "Compound_ID"] or f"row {index + 1}")
        if canonical in first_seen:
            duplicate_of.append(first_seen[canonical])
        else:
            first_seen[canonical] = identifier
            duplicate_of.append("")
    working["Duplicate_Of"] = duplicate_of
    working["Input_Status"] = working["Input_Error"].where(working["Input_Error"].ne(""), "Valid")
    return working


def example_preview_frame() -> pd.DataFrame:
    """Return the concise three-row preview shown in the web interface."""
    return pd.DataFrame(
        {
            "Compound_ID": ["Compound_001", "Compound_002", "Compound_003"],
            "SMILES": ["CCO", "CC(=O)O", "c1ccccc1"],
        }
    )


@lru_cache(maxsize=1)
def example_input_bytes() -> bytes:
    """Return a validated, chemically diverse 12-row CSV download."""
    rows = []
    for compound_id, smiles in DOWNLOAD_EXAMPLE_STRUCTURES:
        canonical, _, error = canonicalize_smiles(smiles)
        if error or canonical is None:
            raise ValueError(f"Invalid example SMILES for {compound_id}: {smiles}")
        rows.append({"Compound_ID": compound_id, "SMILES": smiles})
    return pd.DataFrame(rows).to_csv(index=False, lineterminator="\n").encode("utf-8")
