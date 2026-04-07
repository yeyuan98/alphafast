#!/usr/bin/env python3
# Copyright 2026 Romero Lab, Duke University
#
# Licensed under CC-BY-NC-SA 4.0. This file is part of AlphaFast,
# a derivative work of AlphaFold 3 by DeepMind Technologies Limited.
# https://creativecommons.org/licenses/by-nc-sa/4.0/

"""Create a mixed-type benchmark test set from PDB mmCIF files.

Samples 40 structures across 5 categories:
  - protein-monomer: 1 protein chain, no other polymer chains
  - protein-protein: ≥2 protein chains, no RNA/DNA
  - protein-ligand:  1 protein chain + ≥1 ligand, no RNA/DNA
  - protein-rna:     ≥1 protein chain + ≥1 RNA chain
  - protein-dna:     ≥1 protein chain + ≥1 DNA chain (no RNA)

Usage:
    python benchmarks/create_mixed_benchmark.py \
        --mmcif_dir /path/to/mmcif_files \
        --output_dir benchmarks/benchmark_set_mixed_40

    # Custom samples per category
    python benchmarks/create_mixed_benchmark.py \
        --mmcif_dir /path/to/mmcif_files \
        --output_dir benchmarks/benchmark_set_mixed_40 \
        --samples_per_category 8 \
        --seed 42
"""

import argparse
import glob
import gzip
import json
import os
import random
import re
import string
import sys
from collections import defaultdict


# ── mmCIF parsing helpers ─────────────────────────────────────────────


def _read_mmcif(path: str) -> str:
    """Read an mmCIF file (plain or gzipped)."""
    if path.endswith(".gz"):
        import gzip as gz
        with gz.open(path, "rt", encoding="utf-8", errors="replace") as f:
            return f.read()
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def _parse_loop_block(text: str, required_field: str) -> list[dict[str, str]]:
    """Parse a CIF loop_ block that contains the required_field.

    Returns a list of row dicts keyed by column name.
    """
    # Find the loop_ block containing the required field
    loop_pat = re.compile(r"loop_\s*\n((?:_\S+\s*\n)+)((?:(?!loop_|_\S+\.).*\n?)*)", re.MULTILINE)
    for m in loop_pat.finditer(text):
        col_block = m.group(1)
        columns = [c.strip() for c in col_block.strip().split("\n") if c.strip()]
        if required_field not in columns:
            continue
        data_block = m.group(2).strip()
        if not data_block:
            continue
        # Parse data rows — simple token splitting (handles single-quoted values)
        rows = []
        tokens = _tokenize_cif(data_block)
        ncols = len(columns)
        for i in range(0, len(tokens) - ncols + 1, ncols):
            row = {columns[j]: tokens[i + j] for j in range(ncols)}
            rows.append(row)
        return rows
    return []


def _tokenize_cif(text: str) -> list[str]:
    """Tokenize CIF data values, handling single-quoted and semicolon-delimited strings."""
    tokens = []
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        if c in (" ", "\t", "\r", "\n"):
            i += 1
        elif c == "#":
            # Comment — skip to end of line
            while i < n and text[i] != "\n":
                i += 1
        elif c in ("'", '"'):
            # Quoted string
            quote = c
            i += 1
            start = i
            while i < n and not (text[i] == quote and (i + 1 >= n or text[i + 1] in (" ", "\t", "\r", "\n"))):
                i += 1
            tokens.append(text[start:i])
            if i < n:
                i += 1  # skip closing quote
        elif c == ";":
            # Multi-line string (starts with ; at beginning of line)
            i += 1
            start = i
            while i < n:
                nl = text.find("\n;", i)
                if nl == -1:
                    i = n
                    break
                i = nl + 2
                break
            tokens.append(text[start:i - 2].strip() if i > start else "")
        else:
            start = i
            while i < n and text[i] not in (" ", "\t", "\r", "\n"):
                i += 1
            tokens.append(text[start:i])
    return tokens


def _get_single_value(text: str, field: str) -> str | None:
    """Get a single (non-loop) CIF field value."""
    pat = re.compile(rf"^{re.escape(field)}\s+(\S+)", re.MULTILINE)
    m = pat.search(text)
    if m:
        val = m.group(1).strip("'\"")
        return val if val not in ("?", ".") else None
    return None


# ── Structure classification ──────────────────────────────────────────


def classify_structure(mmcif_text: str) -> dict:
    """Classify a PDB structure by chain composition.

    Returns dict with:
        pdb_id, resolution, n_protein, n_rna, n_dna, n_ligand,
        total_residues, category, release_date
    """
    info = {
        "pdb_id": None,
        "resolution": None,
        "n_protein": 0,
        "n_rna": 0,
        "n_dna": 0,
        "n_ligand": 0,
        "total_residues": 0,
        "category": None,
        "release_date": None,
    }

    # PDB ID
    info["pdb_id"] = _get_single_value(mmcif_text, "_entry.id")

    # Resolution
    res_str = _get_single_value(mmcif_text, "_refine.ls_d_res_high")
    if res_str is None:
        res_str = _get_single_value(mmcif_text, "_em_3d_reconstruction.resolution")
    if res_str is not None:
        try:
            info["resolution"] = float(res_str)
        except ValueError:
            pass

    # Release date
    info["release_date"] = _get_single_value(
        mmcif_text, "_pdbx_audit_revision_history.revision_date"
    )

    # Entity types from _entity_poly.type
    entity_rows = _parse_loop_block(mmcif_text, "_entity_poly.type")
    entity_types = {}  # entity_id -> type
    entity_seqs = {}   # entity_id -> sequence length

    for row in entity_rows:
        eid = row.get("_entity_poly.entity_id", "")
        etype = row.get("_entity_poly.type", "").lower()
        seq = row.get("_entity_poly.pdbx_seq_one_letter_code_can", "")
        # Clean sequence
        seq = seq.replace("\n", "").replace(" ", "").replace("?", "")
        entity_types[eid] = etype
        entity_seqs[eid] = len(seq)

    # Count chains per entity from _pdbx_entity_poly_na_nonstandard or _struct_asym
    asym_rows = _parse_loop_block(mmcif_text, "_struct_asym.entity_id")
    chain_entity_map = {}
    for row in asym_rows:
        chain_id = row.get("_struct_asym.id", "")
        eid = row.get("_struct_asym.entity_id", "")
        chain_entity_map[chain_id] = eid

    # Count unique polymer chains and non-polymer entities
    protein_chains = 0
    rna_chains = 0
    dna_chains = 0
    total_residues = 0

    counted_entities = set()
    for chain_id, eid in chain_entity_map.items():
        etype = entity_types.get(eid, "")
        seq_len = entity_seqs.get(eid, 0)

        if "polypeptide" in etype:
            protein_chains += 1
            total_residues += seq_len
        elif "polyribonucleotide" in etype and "deoxy" not in etype:
            rna_chains += 1
            total_residues += seq_len
        elif "polydeoxyribonucleotide" in etype:
            dna_chains += 1
            total_residues += seq_len

    # Count non-polymer entities (ligands), excluding common crystallization
    # artifacts and ions that are not biologically meaningful ligands.
    # Without this filter, almost every structure has ligand_count > 0 (due to
    # buffer/cryo molecules), making the protein-monomer category nearly empty.
    _ARTIFACT_CCD_CODES = frozenset({
        # Ions
        "NA", "CL", "MG", "ZN", "CA", "K", "MN", "FE", "FE2", "CO", "NI",
        "CU", "CU1", "CD", "IOD", "BR", "XE",
        # Common buffers / cryo / crystallization agents
        "SO4", "PO4", "GOL", "EDO", "PEG", "PGE", "MPD", "DMS", "ACT",
        "FMT", "TRS", "CIT", "BME", "MES", "EPE", "IMD", "SCN", "NO3",
        "AZI", "1PE", "P6G", "MLI", "TAR", "SUC", "NH4",
        # Water-like
        "HOH", "DOD",
    })
    entity_all = _parse_loop_block(mmcif_text, "_entity.type")
    nonpoly_entity_ids = set()
    for row in entity_all:
        etype = row.get("_entity.type", "").lower()
        if etype == "non-polymer":
            nonpoly_entity_ids.add(row.get("_entity.id", ""))
        # water entities are skipped

    # Resolve CCD codes for non-polymer entities to filter artifacts
    ligand_rows = _parse_loop_block(mmcif_text, "_pdbx_entity_nonpoly.comp_id")
    entity_ccd_map = {}
    for row in ligand_rows:
        eid = row.get("_pdbx_entity_nonpoly.entity_id", "")
        ccd = row.get("_pdbx_entity_nonpoly.comp_id", "")
        if ccd and ccd not in ("?", "."):
            entity_ccd_map[eid] = ccd

    ligand_count = 0
    for eid in nonpoly_entity_ids:
        ccd = entity_ccd_map.get(eid, "")
        if ccd.upper() not in _ARTIFACT_CCD_CODES:
            ligand_count += 1

    info["n_protein"] = protein_chains
    info["n_rna"] = rna_chains
    info["n_dna"] = dna_chains
    info["n_ligand"] = ligand_count
    info["total_residues"] = total_residues

    # Classify
    if protein_chains >= 1 and rna_chains >= 1:
        info["category"] = "protein-rna"
    elif protein_chains >= 1 and dna_chains >= 1 and rna_chains == 0:
        info["category"] = "protein-dna"
    elif protein_chains >= 2 and rna_chains == 0 and dna_chains == 0:
        info["category"] = "protein-protein"
    elif protein_chains == 1 and ligand_count >= 1 and rna_chains == 0 and dna_chains == 0:
        info["category"] = "protein-ligand"
    elif protein_chains == 1 and rna_chains == 0 and dna_chains == 0 and ligand_count == 0:
        info["category"] = "protein-monomer"
    else:
        info["category"] = None  # doesn't fit any target category

    return info


# ── JSON generation ───────────────────────────────────────────────────


def _seq_to_json_chain(entity_type: str, sequence: str, chain_id: str) -> dict:
    """Convert a sequence to AF3 JSON chain format."""
    if "polypeptide" in entity_type:
        return {"protein": {"id": chain_id, "sequence": sequence}}
    elif "polyribonucleotide" in entity_type and "deoxy" not in entity_type:
        return {"rna": {"id": chain_id, "sequence": sequence}}
    elif "polydeoxyribonucleotide" in entity_type:
        return {"dna": {"id": chain_id, "sequence": sequence}}
    return None


def generate_af3_json(mmcif_text: str, pdb_id: str) -> dict | None:
    """Generate AF3 JSON input from mmCIF content.

    Returns the JSON dict or None if the structure can't be converted.
    """
    entity_rows = _parse_loop_block(mmcif_text, "_entity_poly.type")
    asym_rows = _parse_loop_block(mmcif_text, "_struct_asym.entity_id")

    # Map entity_id -> (type, sequence)
    entity_info = {}
    for row in entity_rows:
        eid = row.get("_entity_poly.entity_id", "")
        etype = row.get("_entity_poly.type", "").lower()
        seq = row.get("_entity_poly.pdbx_seq_one_letter_code_can", "")
        seq = seq.replace("\n", "").replace(" ", "").replace("?", "")
        entity_info[eid] = (etype, seq)

    # Map chain -> entity
    chain_entity = {}
    for row in asym_rows:
        chain_id = row.get("_struct_asym.id", "")
        eid = row.get("_struct_asym.entity_id", "")
        chain_entity[chain_id] = eid

    # Collect non-polymer entities for ligands
    entity_all = _parse_loop_block(mmcif_text, "_entity.type")
    nonpolymer_entities = set()
    for row in entity_all:
        if row.get("_entity.type", "").lower() == "non-polymer":
            nonpolymer_entities.add(row.get("_entity.id", ""))

    # Get ligand CCD codes from _pdbx_entity_nonpoly.comp_id
    ligand_rows = _parse_loop_block(mmcif_text, "_pdbx_entity_nonpoly.comp_id")
    entity_ccd = {}
    for row in ligand_rows:
        eid = row.get("_pdbx_entity_nonpoly.entity_id", "")
        ccd = row.get("_pdbx_entity_nonpoly.comp_id", "")
        if ccd and ccd not in ("?", "."):
            entity_ccd[eid] = ccd

    # Build chain list (assign IDs A, B, C, ...)
    chain_letters = list(string.ascii_uppercase)
    sequences = []
    letter_idx = 0

    for chain_id, eid in sorted(chain_entity.items()):
        if letter_idx >= len(chain_letters):
            break
        if eid in entity_info:
            etype, seq = entity_info[eid]
            if not seq:
                continue
            chain_json = _seq_to_json_chain(etype, seq, chain_letters[letter_idx])
            if chain_json:
                sequences.append(chain_json)
                letter_idx += 1
        elif eid in nonpolymer_entities:
            ccd_code = entity_ccd.get(eid)
            if ccd_code:
                sequences.append({
                    "ligand": {
                        "id": chain_letters[letter_idx],
                        "ccdCodes": [ccd_code],
                    }
                })
                letter_idx += 1

    if not sequences:
        return None

    return {
        "name": pdb_id,
        "modelSeeds": [42],
        "sequences": sequences,
        "dialect": "alphafold3",
        "version": 2,
    }


# ── Main ──────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Create a mixed-type benchmark test set from PDB mmCIF files."
    )
    parser.add_argument(
        "--mmcif_dir", required=True,
        help="Directory containing mmCIF files (*.cif or *.cif.gz)",
    )
    parser.add_argument(
        "--output_dir", required=True,
        help="Output directory for AF3 JSON files",
    )
    parser.add_argument(
        "--samples_per_category", type=int, default=8,
        help="Number of samples per category (default: 8)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--max_resolution", type=float, default=3.0,
        help="Maximum resolution in Angstroms (default: 3.0)",
    )
    parser.add_argument(
        "--min_residues", type=int, default=50,
        help="Minimum total residues (default: 50)",
    )
    parser.add_argument(
        "--max_residues", type=int, default=1500,
        help="Maximum total residues (default: 1500)",
    )
    parser.add_argument(
        "--max_template_date", default="2021-09-30",
        help="Maximum release date (default: 2021-09-30)",
    )
    args = parser.parse_args()

    categories = [
        "protein-monomer",
        "protein-protein",
        "protein-ligand",
        "protein-rna",
        "protein-dna",
    ]

    # Find all mmCIF files
    patterns = [
        os.path.join(args.mmcif_dir, "*.cif"),
        os.path.join(args.mmcif_dir, "*.cif.gz"),
        os.path.join(args.mmcif_dir, "**", "*.cif"),
        os.path.join(args.mmcif_dir, "**", "*.cif.gz"),
    ]
    mmcif_files = set()
    for pattern in patterns:
        mmcif_files.update(glob.glob(pattern, recursive=True))
    mmcif_files = sorted(mmcif_files)

    if not mmcif_files:
        print(f"ERROR: No mmCIF files found in {args.mmcif_dir}")
        sys.exit(1)

    print(f"Found {len(mmcif_files)} mmCIF files")
    print(f"Filters: resolution ≤ {args.max_resolution}Å, "
          f"residues {args.min_residues}-{args.max_residues}, "
          f"date ≤ {args.max_template_date}")
    print()

    # Classify all structures
    candidates = defaultdict(list)
    processed = 0
    errors = 0

    for path in mmcif_files:
        processed += 1
        if processed % 1000 == 0:
            counts = {k: len(v) for k, v in candidates.items()}
            print(f"  Processed {processed}/{len(mmcif_files)} files... {counts}")

        try:
            text = _read_mmcif(path)
            info = classify_structure(text)
        except Exception:
            errors += 1
            continue

        # Apply filters
        if info["category"] is None:
            continue
        if info["resolution"] is None or info["resolution"] > args.max_resolution:
            continue
        if info["total_residues"] < args.min_residues or info["total_residues"] > args.max_residues:
            continue
        if info["release_date"] and info["release_date"] > args.max_template_date:
            continue

        info["path"] = path
        candidates[info["category"]].append(info)

    print()
    print("Candidates per category:")
    for cat in categories:
        print(f"  {cat}: {len(candidates[cat])}")
    print(f"  (errors: {errors})")
    print()

    # Sample
    random.seed(args.seed)
    selected = {}
    for cat in categories:
        pool = candidates[cat]
        n = min(args.samples_per_category, len(pool))
        if n < args.samples_per_category:
            print(f"WARNING: Only {n} candidates for {cat} "
                  f"(requested {args.samples_per_category})")
        selected[cat] = random.sample(pool, n)

    # Generate AF3 JSON files
    os.makedirs(args.output_dir, exist_ok=True)
    total_written = 0

    for cat in categories:
        for info in selected[cat]:
            pdb_id = info["pdb_id"] or os.path.basename(info["path"]).split(".")[0]
            text = _read_mmcif(info["path"])
            af3_json = generate_af3_json(text, pdb_id)
            if af3_json is None:
                print(f"  SKIP: Could not generate JSON for {pdb_id}")
                continue

            output_path = os.path.join(args.output_dir, f"{pdb_id}.json")
            with open(output_path, "w") as f:
                json.dump(af3_json, f, indent=2)
            total_written += 1

    # Summary
    print(f"\nBenchmark test set created: {total_written} files")
    print(f"Output directory: {args.output_dir}")
    print()
    print("Per-category breakdown:")
    for cat in categories:
        items = selected[cat]
        pdb_ids = [i["pdb_id"] or "?" for i in items]
        print(f"  {cat} ({len(items)}): {', '.join(pdb_ids)}")


if __name__ == "__main__":
    main()
