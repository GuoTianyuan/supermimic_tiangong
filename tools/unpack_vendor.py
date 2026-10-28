"""Unpack checked-in, pinned third-party sources without any network requests."""
from pathlib import Path
import hashlib
import json
import tarfile

ROOT = Path(__file__).resolve().parents[1]
manifest = json.loads((ROOT / 'vendor/manifest.json').read_text())
for name, info in manifest.items():
    archive = ROOT / 'vendor' / name
    if hashlib.sha256(archive.read_bytes()).hexdigest() != info['sha256']:
        raise RuntimeError(f'Archive checksum mismatch: {name}')
    dest = ROOT / info['destination']
    marker = dest / '.supermimic-source-sha256'
    if marker.exists() and marker.read_text().strip() == info['sha256']:
        continue
    # Never overwrite an existing third-party checkout or partial extraction.
    if dest.exists():
        raise RuntimeError(f'{dest} exists without matching source marker; move it aside and retry')
    dest.mkdir(parents=True)
    with tarfile.open(archive) as tar:
        tar.extractall(dest, filter='data')
    if name.startswith('newton-'):
        (dest / 'pyproject.toml').write_text('''[build-system]
requires = ["setuptools>=70,<82", "wheel"]
build-backend = "setuptools.build_meta"
[project]
name = "newton"
version = "1.8.0.dev0"
description = "Newton source snapshot at 71eeba16a717d4dcf0b9dd7056a7b6f863054dc3"
license = {text = "Apache-2.0"}
[tool.setuptools.packages.find]
include = ["newton", "newton.*"]
[tool.setuptools.package-data]
"*" = ["*.json", "*.usd", "*.usda", "*.txt", "*.h", "*.cpp", "*.cu", "*.pyi", "py.typed"]
''')
    marker.write_text(info['sha256'] + '\n')
    print(f'Unpacked {name}: {dest}')
