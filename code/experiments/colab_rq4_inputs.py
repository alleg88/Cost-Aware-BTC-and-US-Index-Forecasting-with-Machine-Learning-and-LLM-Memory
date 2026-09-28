"""Load individual, checksum-verified RQ4 files from Google Drive in Colab."""
from concurrent.futures import ThreadPoolExecutor
from importlib.util import find_spec
from pathlib import Path
from tempfile import gettempdir
from urllib.request import urlopen
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time


def _hash(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def prepare(manifest_url, manifest_sha256):
    payload = urlopen(manifest_url, timeout=60).read()
    if hashlib.sha256(payload).hexdigest() != manifest_sha256:
        raise ValueError("The registered RQ4 input list has changed.")
    manifest = json.loads(payload)
    packages = {"numpy": "numpy", "pandas": "pandas", "pyarrow": "pyarrow", "matplotlib": "matplotlib"}
    missing = [package for module, package in packages.items() if find_spec(module) is None]
    if missing:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *missing], check=True)
    root = (Path(gettempdir()) / "msc_rq4_weights").resolve()
    root.mkdir(exist_ok=True)
    mounted = Path("/content/drive/MyDrive/MSc Colab Data - Public")

    def fetch(item):
        relative, record = item
        target = (root / relative).resolve()
        if not target.is_relative_to(root):
            raise ValueError("RQ4 input path escaped the local cache.")
        if target.is_file() and _hash(target) == record["sha256"]:
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        mounted_source = mounted / relative
        if mounted_source.is_file() and _hash(mounted_source) == record["sha256"]:
            shutil.copyfile(mounted_source, target)
            return
        temporary = target.with_name(target.name + ".download")
        url = "https://drive.usercontent.google.com/download?export=download&confirm=t&id=" + record["id"]
        for attempt in range(3):
            try:
                temporary.write_bytes(urlopen(url, timeout=60).read())
                if _hash(temporary) != record["sha256"]:
                    raise ValueError(f"RQ4 input content mismatch: {relative}")
                temporary.replace(target)
                return
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)

    print(f"Loading {len(manifest['files'])} RQ4 files from Google Drive...", flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        for completed, _ in enumerate(pool.map(fetch, manifest["files"].items()), 1):
            if completed % 25 == 0:
                print(f"Checked {completed}/{len(manifest['files'])} files.", flush=True)
    os.chdir(root)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    print("RQ4 data ready.", flush=True)
    return root
