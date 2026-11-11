"""Locate checkout-owned assets without depending on the working directory."""
from pathlib import Path
import os


def repository_root() -> Path:
    override = os.environ.get("SUPERMIMIC_ROOT")
    if override:
        root = Path(override).expanduser().resolve()
        if not (root / "assets-manifest.json").is_file():
            raise FileNotFoundError(f"SUPERMIMIC_ROOT has no assets-manifest.json: {root}")
        return root
    source_root = Path(__file__).resolve().parents[2]
    if (source_root / "assets-manifest.json").is_file():
        return source_root
    # A non-editable wheel may be used with a separate checkout holding assets.
    for root in (Path.cwd(), *Path.cwd().parents):
        if (root / "assets-manifest.json").is_file():
            return root
    raise FileNotFoundError("Use an editable install, or set SUPERMIMIC_ROOT to the asset checkout")


REPO_ROOT = repository_root()
DATA_ROOT = REPO_ROOT / "data"
