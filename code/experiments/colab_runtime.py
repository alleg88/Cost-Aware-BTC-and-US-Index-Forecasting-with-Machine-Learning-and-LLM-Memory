"""Install missing packages without replacing Colab's working Python stack."""
from __future__ import annotations

import argparse
import importlib
from importlib import metadata
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import tomllib


CODE_ROOT = Path(__file__).resolve().parents[1]
IMPORT_NAMES = {
    "scikit-learn": "sklearn", "pyyaml": "yaml", "protobuf": "google.protobuf",
    "sentence-transformers": "sentence_transformers",
}


def is_colab() -> bool:
    try:
        metadata.version("google-colab")
        return True
    except metadata.PackageNotFoundError:
        return False


def _name(requirement: str) -> str:
    return re.sub(r"[-_.]+", "-", re.split(r"[<>=!~\[; ]", requirement, maxsplit=1)[0]).lower()


def _requirements(code_root: Path) -> list[str]:
    with (code_root / "pyproject.toml").open("rb") as handle:
        project = tomllib.load(handle)["project"]
    return project["dependencies"] + project.get("optional-dependencies", {}).get("extras", [])


def install(code_root: Path = CODE_ROOT) -> None:
    """Keep installed versions; constrain transitive resolution to the same stack."""
    installed = {
        _name(dist.metadata["Name"]): dist.version
        for dist in metadata.distributions() if dist.metadata.get("Name")
    }
    missing = [requirement for requirement in _requirements(code_root) if _name(requirement) not in installed]
    if missing:
        with tempfile.TemporaryDirectory(prefix="colab-packages-") as temporary:
            constraints = Path(temporary) / "installed.txt"
            constraints.write_text("\n".join(f"{name}=={version}" for name, version in sorted(installed.items())), encoding="utf-8")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-q", "--constraint", str(constraints), *missing,
            ])
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q", "--no-deps", "-e", str(code_root),
    ])


def check(code_root: Path = CODE_ROOT) -> None:
    """Verify real imports; the full test suite checks behaviour on this stack."""
    versions = {"python": sys.version.split()[0]}
    for requirement in _requirements(code_root):
        name = _name(requirement)
        versions[name] = metadata.version(name)
        importlib.import_module(IMPORT_NAMES.get(name, name.replace("-", "_")))
    print(json.dumps(versions, indent=2, sort_keys=True), flush=True)
    print("Native Colab package imports passed.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install", action="store_true")
    args = parser.parse_args()
    if args.install:
        install()
    else:
        check()
