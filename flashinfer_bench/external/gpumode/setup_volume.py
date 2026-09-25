"""One-time setup / incremental update: sync flashinfer-trace into a Modal Volume.

Usage:
    modal run flashinfer_bench/external/gpumode/setup_volume.py          # initial clone
    modal run flashinfer_bench/external/gpumode/setup_volume.py --update # incremental pull
"""

import modal

VOLUME_NAME = "flashinfer-trace"
MOUNT_PATH = "/data/flashinfer-trace"
REPO_URL = "https://huggingface.co/datasets/flashinfer-ai/flashinfer-trace"

volume = modal.Volume.from_name(VOLUME_NAME)

image = modal.Image.debian_slim().apt_install("git", "git-lfs").run_commands("git lfs install")

app = modal.App("flashinfer-trace-setup")


@app.function(image=image, volumes={MOUNT_PATH: volume}, timeout=3600)
def sync_dataset(update: bool = False):
    """Clone or update flashinfer-trace in the volume."""
    import subprocess
    from pathlib import Path

    target = Path(MOUNT_PATH)
    exists = (target / ".git").exists()

    if exists and not update:
        print("Volume already populated. Use --update to pull latest changes. Skipping.")
        return

    if not exists:
        print("Cloning flashinfer-trace with full LFS...")
        subprocess.run(["git", "clone", REPO_URL, str(target)], check=True)
        subprocess.run(["git", "lfs", "pull"], cwd=str(target), check=True)
    else:
        print("Updating flashinfer-trace...")
        subprocess.run(["git", "pull", "--ff-only"], cwd=str(target), check=True)
        subprocess.run(["git", "lfs", "pull"], cwd=str(target), check=True)

    volume.commit()
    print("Done.")


@app.local_entrypoint()
def main(update: bool = False):
    sync_dataset.remote(update=update)
