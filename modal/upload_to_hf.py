"""
Upload AlphaFold3 databases from Modal volume to HuggingFace Hub.

Uploads directly from Modal container (no local download needed).
Uses upload_large_folder() for resumable, chunked uploads of 100GB+ files.
Large mmcif_files directory is tarred first to stay under HF's 100K file limit.

Prerequisites:
    modal secret create huggingface HF_TOKEN=hf_xxxxx

Usage:
    # Check what would be uploaded (dry run, no token needed)
    modal run modal/upload_to_hf.py::dry_app

    # Upload to HuggingFace
    modal run modal/upload_to_hf.py::app

    # Custom repo name
    modal run modal/upload_to_hf.py::app --repo RomeroLab-Duke/af3-mmseqs-db

    # Skip mmcif tar (if already uploaded)
    modal run modal/upload_to_hf.py::app --skip-mmcif
"""

import sys

import modal

DATABASE_VOLUME_NAME = "af3-databases"
DATABASE_MOUNT_PATH = "/databases"
STAGING_DIR = "/tmp/staging"
SPLIT_THRESHOLD = 50 * (1024**3)  # 50 GB - HF struggles with files > 50GB
SPLIT_CHUNK_SIZE = "50G"

DEFAULT_REPO = "RomeroLab-Duke/af3-mmseqs-db"

db_volume = modal.Volume.from_name(DATABASE_VOLUME_NAME)

MMCIF_TAR_URL = "https://storage.googleapis.com/alphafold-databases/v3.0/pdb_2022_09_28_mmcif_files.tar.zst"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("huggingface_hub[hf_xet]")
    .apt_install("tar", "zstd", "wget")
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)

dry_run_image = modal.Image.debian_slim(python_version="3.12")


def _collect_upload_files(db_path: str, skip_mmcif: bool = False) -> dict:
    """Collect files to upload, grouped by category.

    Returns:
        Dict with keys: 'mmseqs', 'root_files', 'mmcif_tar'
        Each value is a list of (local_path, repo_path) tuples.
    """
    from pathlib import Path

    db = Path(db_path)
    result = {"mmseqs": [], "root_files": [], "mmcif_tar": []}

    # MMseqs padded databases
    mmseqs_dir = db / "mmseqs"
    if mmseqs_dir.exists():
        for f in sorted(mmseqs_dir.iterdir()):
            if f.is_file():
                result["mmseqs"].append((str(f), f"mmseqs/{f.name}"))

    # Root-level files (RNA FASTAs etc.)
    for f in sorted(db.iterdir()):
        if f.is_file():
            result["root_files"].append((str(f), f.name))

    # mmcif_files → tar (167K files, HF recommends < 100K per repo)
    mmcif_dir = db / "mmcif_files"
    if not skip_mmcif and mmcif_dir.exists() and any(mmcif_dir.iterdir()):
        result["mmcif_tar"] = [("__tar__", "mmcif_files.tar.zst")]

    return result


def _print_summary(files: dict, db_path: str) -> None:
    """Print upload summary."""
    from pathlib import Path

    total_size = 0
    for category, file_list in files.items():
        if not file_list:
            continue
        cat_size = 0
        for local_path, repo_path in file_list:
            if local_path == "__tar__":
                mmcif_dir = Path(db_path) / "mmcif_files"
                for f in mmcif_dir.rglob("*"):
                    if f.is_file():
                        cat_size += f.stat().st_size
            else:
                cat_size += Path(local_path).stat().st_size
        total_size += cat_size
        print(f"  {category}: {len(file_list)} files, {cat_size / (1024**3):.1f} GB")

    print(f"  TOTAL: {total_size / (1024**3):.1f} GB")


# ============================================================
# Dry-run app (no secrets required)
# ============================================================

dry_app = modal.App("af3-upload-to-hf-dryrun")


@dry_app.function(
    image=dry_run_image,
    volumes={DATABASE_MOUNT_PATH: db_volume},
    timeout=600,
)
def dry_run_check(repo_id: str = DEFAULT_REPO, skip_mmcif: bool = False):
    """Dry run - show what would be uploaded (no HF token needed)."""
    print("=" * 60)
    print(f"[DRY RUN] Upload to HuggingFace: {repo_id}")
    print(f"Source volume: {DATABASE_VOLUME_NAME}")
    print("=" * 60)
    print()

    files = _collect_upload_files(DATABASE_MOUNT_PATH, skip_mmcif=skip_mmcif)
    _print_summary(files, DATABASE_MOUNT_PATH)

    print()
    print("Dry run - no files uploaded.")
    print()
    print("To upload, run:")
    print("  modal secret create huggingface HF_TOKEN=hf_xxxxx")
    print(f"  modal run modal/upload_to_hf.py --repo {repo_id}")


@dry_app.local_entrypoint()
def dry_main(
    repo: str = DEFAULT_REPO,
    skip_mmcif: bool = False,
):
    dry_run_check.remote(repo_id=repo, skip_mmcif=skip_mmcif)


# ============================================================
# Upload app (requires huggingface secret)
# ============================================================

app = modal.App("af3-upload-to-hf")


@app.function(
    image=image,
    volumes={DATABASE_MOUNT_PATH: db_volume},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=3600 * 24,
    cpu=8,
    memory=32768,
    ephemeral_disk=512 * 1024,  # 512 GB for mmcif tar + staging
)
def upload(repo_id: str = DEFAULT_REPO, skip_mmcif: bool = False):
    """Upload database files to HuggingFace Hub.

    Creates a staging directory with symlinks to volume files, then uses
    upload_large_folder() for resumable, chunked, multi-threaded uploads.
    This avoids the infinite-retry issue that upload_file() has with 100GB+ files.
    """
    import os
    import shutil
    import subprocess
    from pathlib import Path

    from huggingface_hub import HfApi

    token = os.environ["HF_TOKEN"]
    api = HfApi(token=token)

    print("=" * 60)
    print(f"Upload to HuggingFace: {repo_id}")
    print(f"Source volume: {DATABASE_VOLUME_NAME}")
    print("=" * 60)
    print()

    # Ensure repo exists
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
    print(f"Repo ready: https://huggingface.co/datasets/{repo_id}")
    print()

    files = _collect_upload_files(DATABASE_MOUNT_PATH, skip_mmcif=skip_mmcif)
    _print_summary(files, DATABASE_MOUNT_PATH)
    print()

    # Build staging directory with symlinks for small files, split for large
    staging = Path(STAGING_DIR)
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    db = Path(DATABASE_MOUNT_PATH)

    def _stage_file(local_path: str, repo_path: str) -> None:
        """Symlink small files, split large files into chunks."""
        src = Path(local_path)
        size = src.stat().st_size
        size_gb = size / (1024**3)
        dst = staging / repo_path

        dst.parent.mkdir(parents=True, exist_ok=True)

        if size <= SPLIT_THRESHOLD:
            dst.symlink_to(src)
            print(f"  Staged: {repo_path} ({size_gb:.1f} GB)")
        else:
            # Split into 50GB chunks: file.part00, file.part01, ...
            print(f"  Splitting: {repo_path} ({size_gb:.1f} GB) into {SPLIT_CHUNK_SIZE} chunks...")
            result = subprocess.run(
                [
                    "split", "-b", SPLIT_CHUNK_SIZE,
                    "-d", "--suffix-length=2",
                    str(src),
                    str(dst) + ".part",
                ],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                raise RuntimeError(f"split failed for {repo_path}: {result.stderr}")
            parts = sorted(dst.parent.glob(f"{dst.name}.part*"))
            for p in parts:
                part_gb = p.stat().st_size / (1024**3)
                print(f"    {p.parent.name}/{p.name} ({part_gb:.1f} GB)")
            print(f"  Split into {len(parts)} parts")

    # Stage mmseqs files
    if files["mmseqs"]:
        (staging / "mmseqs").mkdir(exist_ok=True)
        for local_path, repo_path in files["mmseqs"]:
            _stage_file(local_path, repo_path)

    # Stage root files
    if files["root_files"]:
        for local_path, repo_path in files["root_files"]:
            _stage_file(local_path, repo_path)

    # Download mmcif tar.zst from Google Cloud (much faster than re-tarring
    # 167K files from network volume, which can take 9+ hours)
    if files["mmcif_tar"]:
        print()
        print("Downloading mmcif tar.zst from Google Cloud...")
        print(f"  Source: {MMCIF_TAR_URL}")
        tar_path = staging / "mmcif_files.tar.zst"
        result = subprocess.run(
            ["wget", "-q", "--show-progress", "-O", str(tar_path), MMCIF_TAR_URL],
        )
        if result.returncode != 0:
            raise RuntimeError(f"wget failed for mmcif tar.zst")
        tar_size_gb = tar_path.stat().st_size / (1024**3)
        print(f"  Downloaded: mmcif_files.tar.zst ({tar_size_gb:.1f} GB)")
        # Split if too large for HF
        if tar_path.stat().st_size > SPLIT_THRESHOLD:
            _stage_file(str(tar_path), "mmcif_files.tar.zst")
            tar_path.unlink()  # remove original, keep parts

    print()
    print("=" * 60)
    print("Starting upload_large_folder (resumable, multi-threaded)...")
    print("=" * 60)

    api.upload_large_folder(
        folder_path=str(staging),
        repo_id=repo_id,
        repo_type="dataset",
    )

    print()
    print("=" * 60)
    print("Upload complete!")
    print(f"https://huggingface.co/datasets/{repo_id}")
    print("=" * 60)


@app.local_entrypoint()
def main(
    repo: str = DEFAULT_REPO,
    dry_run: bool = False,
    skip_mmcif: bool = False,
):
    """Upload AF3 databases from Modal volume to HuggingFace Hub.

    Args:
        repo: HuggingFace dataset repo (default: RomeroLab-Duke/af3-mmseqs-db)
        dry_run: Only show what would be uploaded
        skip_mmcif: Skip mmcif_files tar upload
    """
    if dry_run:
        print("Use: modal run modal/upload_to_hf.py::dry_app --repo", repo)
        print("(dry-run requires separate app to skip secret validation)")
        sys.exit(1)

    upload.remote(repo_id=repo, skip_mmcif=skip_mmcif)
