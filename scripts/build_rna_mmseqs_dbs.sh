#!/bin/bash
# Copyright 2026 Romero Lab, Duke University
#
# Licensed under CC-BY-NC-SA 4.0. This file is part of AlphaFast,
# a derivative work of AlphaFold 3 by DeepMind Technologies Limited.
# https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# Build MMseqs2 databases from RNA FASTA files for nucleotide search.
#
# This converts the RNA FASTA databases (used by nhmmer) into MMseqs2
# format so they can be searched with 'mmseqs search --search-type 3'.
# GPU is NOT supported for nucleotide search, but MMseqs2 CPU nucleotide
# search is significantly faster than nhmmer.
#
# Usage:
#   ./scripts/build_rna_mmseqs_dbs.sh <db_dir> [<output_dir>]
#
# Arguments:
#   db_dir       Directory containing RNA FASTA files
#   output_dir   Where to write MMseqs2 databases (default: <db_dir>/mmseqs_rna)
#
# Requirements:
#   - mmseqs in PATH
#   - ~400 GB peak RAM for index building (nt_rna is the largest)
#   - ~500 GB disk for index files
#
# After building, pass --rna_mmseqs_db_dir=<output_dir> to the pipeline.

set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <db_dir> [<output_dir>]"
    echo ""
    echo "Build MMseqs2 nucleotide databases from RNA FASTA files."
    echo "After building, use --rna_mmseqs_db_dir to enable MMseqs2 nucleotide search."
    exit 1
fi

DB_DIR="$1"
OUTPUT_DIR="${2:-${DB_DIR}/mmseqs_rna}"

if ! command -v mmseqs &>/dev/null; then
    echo "ERROR: mmseqs not found in PATH"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "Build RNA MMseqs2 Databases"
echo "=========================================="
echo "Source dir:  $DB_DIR"
echo "Output dir:  $OUTPUT_DIR"
echo "mmseqs:      $(which mmseqs)"
echo "=========================================="
echo ""

# Database name -> source FASTA filename
declare -A RNA_DBS=(
    ["rfam"]="rfam_14_9_clust_seq_id_90_cov_80_rep_seq.fasta"
    ["rnacentral"]="rnacentral_active_seq_id_90_cov_80_linclust.fasta"
    ["nt_rna"]="nt_rna_2023_02_23_clust_seq_id_90_cov_80_rep_seq.fasta"
)

total=${#RNA_DBS[@]}
current=0

for db_name in "${!RNA_DBS[@]}"; do
    current=$((current + 1))
    fasta="${DB_DIR}/${RNA_DBS[$db_name]}"
    target="${OUTPUT_DIR}/${db_name}"

    echo "[$current/$total] $db_name"

    if [ -f "${target}.dbtype" ]; then
        echo "  SKIP: Already exists at ${target}"
        echo ""
        continue
    fi

    if [ ! -f "$fasta" ]; then
        echo "  WARNING: Source FASTA not found: $fasta"
        echo "  Skipping."
        echo ""
        continue
    fi

    echo "  Source: $fasta ($(du -h "$fasta" | cut -f1))"
    echo "  Creating MMseqs2 database..."
    time mmseqs createdb "$fasta" "$target"

    if [ ! -f "${target}.dbtype" ]; then
        echo "  ERROR: Database creation failed"
        echo ""
        continue
    fi
    echo "  OK: Nucleotide database created"

    # Build k-mer index for fast search. Without this, MMseqs2 rebuilds
    # the index from the raw database on every query, which is extremely
    # slow for large databases (e.g. nt_rna at 76GB).
    # Large databases use --split to avoid OOM (nt_rna needs ~372GB RAM
    # unsplit; --split 4 reduces peak RAM to ~93GB per chunk).
    if [ -f "${target}.idx" ]; then
        echo "  SKIP: Index already exists"
    else
        echo "  Creating search index (this may take a while for large databases)..."
        idx_tmp=$(mktemp -d)
        # Use --split 4 for databases >10GB to stay within typical RAM limits.
        db_size_gb=$(du -g "$fasta" 2>/dev/null | cut -f1)
        split_flag=""
        if [ "${db_size_gb:-0}" -gt 10 ]; then
            split_flag="--split 4"
            echo "  Using --split 4 for large database (${db_size_gb}G source FASTA)"
        fi
        time mmseqs createindex "$target" "$idx_tmp" --search-type 3 $split_flag
        rm -rf "$idx_tmp"
        if [ -f "${target}.idx" ]; then
            echo "  OK: Index created ($(du -h "${target}.idx" | cut -f1))"
        else
            echo "  WARNING: Index creation failed (searches will be slower)"
        fi
    fi
    echo ""
done

echo "=========================================="
echo "Done!"
echo "=========================================="
echo "Output directory: $OUTPUT_DIR"
echo ""
echo "To use with AlphaFast:"
echo "  python run_data_pipeline.py \\"
echo "      --rna_mmseqs_db_dir $OUTPUT_DIR \\"
echo "      ... (other flags)"
echo ""
echo "This replaces nhmmer with MMseqs2 CPU nucleotide search."
echo "=========================================="
