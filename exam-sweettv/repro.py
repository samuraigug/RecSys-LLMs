"""Small shared reproducibility utilities; no credentials or external writes."""
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import resource

ROOT = Path(__file__).resolve().parent


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def now():
    return datetime.now(timezone.utc).isoformat()


def environment():
    packages = ["numpy", "scipy", "scikit-learn", "implicit", "duckdb", "threadpoolctl"]
    for optional in ('catboost', 'lightfm', 'lightgbm', 'onnxruntime', 'tokenizers', 'huggingface_hub'):
        try:
            importlib.metadata.version(optional)
        except importlib.metadata.PackageNotFoundError:
            continue
        packages.append(optional)
    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": {p: importlib.metadata.version(p) for p in packages},
            "blas_threads": 1, "als_threads": 1}


def peak_memory_mb():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value / (1024 ** 2 if platform.system() == "Darwin" else 1024)


def code_hashes():
    return {p.name: sha256(p) for p in sorted(ROOT.glob("*.py"))}
