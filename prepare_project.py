"""Prepare a verified ZIP export for the Git-based reference checks (standard library only)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess


def prepare(root: Path) -> None:
    root = Path(root).resolve(strict=True)
    if shutil.which("git") is None:
        raise RuntimeError("Install Git, reopen the terminal, then run this command again.")
    if (root / ".git").exists():
        subprocess.run(["git", "rev-parse", "--verify", "HEAD"], cwd=root, check=True,
                       stdout=subprocess.DEVNULL)
        print("Existing Git checkout preserved.")
        return
    manifest_path = root / "release_files.json"
    if not manifest_path.is_file():
        raise ValueError("Use the code ZIP from Releases or clone the Git repository.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not manifest:
        raise ValueError("The release file manifest is empty or invalid.")
    for name, expected in manifest.items():
        relative = PurePosixPath(name)
        if (relative.is_absolute() or ".." in relative.parts or ".git" in relative.parts
                or "\\" in name or ":" in name or "\0" in name):
            raise ValueError(f"Unsafe archive path: {name!r}")
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"Missing or external archive file: {name}")
        with path.open("rb") as handle:
            actual = hashlib.file_digest(handle, "sha256").hexdigest()
        if actual != expected:
            raise ValueError(f"Archive checksum mismatch: {name}")
    if "code/pyproject.toml" not in manifest:
        raise ValueError("The Python package definition is missing from the manifest.")

    def git(*args: str, data: bytes | None = None) -> None:
        subprocess.run(["git", *args], cwd=root, input=data, check=True,
                       stdout=subprocess.DEVNULL)

    git("-c", "init.templateDir=", "init", "-b", "main")
    git("config", "core.autocrlf", "false")
    git("config", "core.safecrlf", "false")
    git("config", "core.longpaths", "true")
    git("add", "-f", "--pathspec-from-file=-", "--pathspec-file-nul",
        data=b"\0".join(name.encode("utf-8") for name in sorted(manifest)) + b"\0")
    git("-c", "user.name=Research artifact", "-c", "user.email=artifact@example.invalid",
        "-c", "commit.gpgSign=false", "-c", "core.hooksPath=", "commit", "--no-verify",
        "-m", "Research snapshot")
    print(f"Verified {len(manifest)} files; local references are ready.")


if __name__ == "__main__":
    prepare(Path(__file__).resolve().parent)
