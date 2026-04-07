#!/bin/bash
# Copyright 2026 Romero Lab, Duke University
#
# Licensed under CC-BY-NC-SA 4.0. This file is part of AlphaFast,
# a derivative work of AlphaFold 3 by DeepMind Technologies Limited.
# https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# Generate a mixed-type benchmark test set (40 structures, 5 categories).
#
# This script runs inside a Docker/Singularity container with access to the
# PDB mmCIF files database.
#
# Usage:
#   # Using Docker
#   ./scripts/benchmarks/generate_mixed_test_set.sh \
#       --mmcif_dir /path/to/databases/mmcif_files \
#       --output_dir benchmarks/benchmark_set_mixed_40
#
#   # Using Docker container
#   ./scripts/benchmarks/generate_mixed_test_set.sh \
#       --mmcif_dir /path/to/databases/mmcif_files \
#       --output_dir benchmarks/benchmark_set_mixed_40 \
#       --container romerolabduke/alphafast:latest
#
# Categories (8 samples each):
#   protein-monomer, protein-protein, protein-ligand, protein-rna, protein-dna

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Defaults
MMCIF_DIR=""
OUTPUT_DIR="${REPO_DIR}/benchmarks/benchmark_set_mixed_40"
CONTAINER_IMAGE=""
SAMPLES=8
SEED=42

usage() {
    echo "Usage: $0 --mmcif_dir DIR [OPTIONS]"
    echo ""
    echo "Generate a mixed-type benchmark test set (40 structures)."
    echo ""
    echo "Required:"
    echo "  --mmcif_dir DIR       Path to PDB mmCIF files directory"
    echo ""
    echo "Optional:"
    echo "  --output_dir DIR      Output directory (default: benchmarks/benchmark_set_mixed_40)"
    echo "  --container IMAGE     Run inside Docker container"
    echo "  --samples N           Samples per category (default: 8)"
    echo "  --seed N              Random seed (default: 42)"
    exit 1
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --mmcif_dir)    MMCIF_DIR="$2"; shift 2 ;;
        --output_dir)   OUTPUT_DIR="$2"; shift 2 ;;
        --container)    CONTAINER_IMAGE="$2"; shift 2 ;;
        --samples)      SAMPLES="$2"; shift 2 ;;
        --seed)         SEED="$2"; shift 2 ;;
        --help|-h)      usage ;;
        *)              echo "Unknown argument: $1"; usage ;;
    esac
done

if [ -z "$MMCIF_DIR" ]; then
    echo "ERROR: --mmcif_dir is required."
    usage
fi

echo "=========================================="
echo "Mixed-Type Benchmark Test Set Generator"
echo "=========================================="
echo "mmCIF dir:  $MMCIF_DIR"
echo "Output dir: $OUTPUT_DIR"
echo "Samples:    $SAMPLES per category ($(( SAMPLES * 5 )) total)"
echo "Seed:       $SEED"
echo "=========================================="
echo ""

mkdir -p "$OUTPUT_DIR"

if [ -n "$CONTAINER_IMAGE" ]; then
    docker run --rm \
        -v "${MMCIF_DIR}:/data/mmcif_files:ro" \
        -v "${OUTPUT_DIR}:/data/output" \
        -v "${REPO_DIR}:/app/alphafold:ro" \
        "$CONTAINER_IMAGE" \
        python /app/alphafold/benchmarks/create_mixed_benchmark.py \
            --mmcif_dir /data/mmcif_files \
            --output_dir /data/output \
            --samples_per_category "$SAMPLES" \
            --seed "$SEED"
else
    python "${REPO_DIR}/benchmarks/create_mixed_benchmark.py" \
        --mmcif_dir "$MMCIF_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --samples_per_category "$SAMPLES" \
        --seed "$SEED"
fi

echo ""
echo "=========================================="
echo "Benchmark test set generated!"
echo "Output: $OUTPUT_DIR"
echo ""
echo "To run predictions:"
echo "  ./scripts/run_alphafast.sh \\"
echo "      --input_dir $OUTPUT_DIR \\"
echo "      --output_dir ./output \\"
echo "      --db_dir /path/to/databases \\"
echo "      --weights_dir /path/to/weights"
echo "=========================================="
