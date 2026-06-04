"""JobFile: manifest parsing, bare-script autowrap, versioning, hashing.

Public API:
    load_jobfile(path) -> JobFile
    resolve_params(jobfile, overrides) -> dict
    render_command(jobfile, params) -> str
    content_hash(jobfile) -> str
    input_hashes(jobfile, params) -> dict
"""
from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from jobctl.db.models import JobFile


# ---------------------------------------------------------------------------
# Extension -> command prefix map for bare-script autowrap
# ---------------------------------------------------------------------------

_EXT_MAP: dict[str, str] = {
    ".py": "python",
    ".sbatch": "sbatch",
    ".sh": "bash",
    ".jl": "julia",
    ".m": "matlab -nodesktop -nosplash -batch",
    ".R": "Rscript",
    ".r": "Rscript",
}


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def load_jobfile(path: str) -> JobFile:
    """Load a JobFile from a .jobfile.yaml manifest or a bare script.

    Bare scripts (.py, .sbatch, .sh, .jl, .m, .R) are auto-wrapped with an
    appropriate command prefix. The returned JobFile has content_hash set.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"JobFile path not found: {path}")

    suffix = p.suffix.lower()
    # Use original suffix for extension checks (case-sensitive for .R)
    orig_suffix = p.suffix

    # Detect manifest vs bare script
    is_manifest = (
        suffix == ".yaml" or suffix == ".yml"
    ) and (
        ".jobfile." in p.name or p.name.endswith(".jobfile.yaml") or p.name.endswith(".jobfile.yml")
    )

    # Also allow plain .yaml files that contain a "command" key
    if (suffix in (".yaml", ".yml")) and not is_manifest:
        raw = yaml.safe_load(p.read_text()) or {}
        if "command" in raw or "name" in raw:
            is_manifest = True

    if is_manifest:
        return _load_manifest(p)
    else:
        return _autowrap_script(p)


def resolve_params(jobfile: JobFile, overrides: dict) -> dict:
    """Apply defaults, cast types, and check required params.

    Args:
        jobfile: the JobFile with params_schema.
        overrides: user-provided param values (may be strings).

    Returns:
        Fully resolved dict with correct Python types.

    Raises:
        ValueError: if a required param is not in overrides.
    """
    schema = jobfile.params_schema or {}
    resolved: dict[str, Any] = {}

    for name, spec in schema.items():
        if name in overrides:
            val = overrides[name]
        elif "default" in spec:
            val = spec["default"]
        elif spec.get("required", False):
            raise ValueError(f"Required parameter '{name}' is missing.")
        else:
            continue

        # Cast to declared type
        ptype = spec.get("type", "str")
        val = _cast(val, ptype)
        resolved[name] = val

    # Pass through any extra overrides not in schema
    for k, v in overrides.items():
        if k not in resolved:
            resolved[k] = v

    return resolved


def render_command(jobfile: JobFile, params: dict) -> str:
    """Substitute params into command_template.

    Example: "julia {script} --chi {chi}" + {"script": "s.jl", "chi": 60}
             -> "julia s.jl --chi 60"
    """
    return jobfile.command_template.format(**{k: str(v) for k, v in params.items()})


def content_hash(jobfile: JobFile) -> str:
    """Deterministic hash of command_template + referenced script bytes.

    If the source_path is a script file (not a manifest), its bytes are included.
    """
    h = hashlib.sha256()
    h.update(jobfile.command_template.encode("utf-8"))
    # Include referenced script bytes if source_path points to a non-manifest
    src = Path(jobfile.source_path) if jobfile.source_path else None
    if src and src.exists():
        suffix = src.suffix.lower()
        if suffix not in (".yaml", ".yml"):
            h.update(src.read_bytes())
        else:
            # Manifest: hash its raw bytes for stability
            h.update(src.read_bytes())
    return h.hexdigest()


def input_hashes(jobfile: JobFile, params: dict) -> dict[str, str]:
    """Compute {path: sha256} for script file + path-typed params.

    Returns a dict mapping each path-typed param value to its sha256 digest.
    Always includes the source script if it's a non-manifest file.
    """
    schema = jobfile.params_schema or {}
    result: dict[str, str] = {}

    # Hash source script if it's a bare script
    src = Path(jobfile.source_path) if jobfile.source_path else None
    if src and src.exists() and src.suffix.lower() not in (".yaml", ".yml"):
        result[str(src)] = _sha256(src)

    for name, spec in schema.items():
        ptype = spec.get("type", "str")
        if ptype == "path" and name in params:
            p = Path(str(params[name]))
            if p.exists():
                result[str(p)] = _sha256(p)

    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_manifest(p: Path) -> JobFile:
    raw = yaml.safe_load(p.read_text()) or {}

    name = raw.get("name", p.stem)
    command_template = raw.get("command", "")
    params_schema = raw.get("params", {})
    backend_prefs = raw.get("backends", [])
    artifact_patterns = raw.get("artifacts", [])
    expectation_text = raw.get("expectation", None)

    # Normalize backend_prefs: list of dicts
    if not isinstance(backend_prefs, list):
        backend_prefs = [backend_prefs]

    # Normalize artifact_patterns: list of strings
    if isinstance(artifact_patterns, str):
        artifact_patterns = [artifact_patterns]

    jf_id = _stable_id(name, str(p))
    now = datetime.now(timezone.utc).isoformat()

    jf = JobFile(
        id=jf_id,
        name=name,
        version=1,
        source_path=str(p),
        command_template=command_template,
        params_schema=params_schema,
        backend_prefs=backend_prefs,
        artifact_patterns=artifact_patterns,
        expectation_contract_id=None,
        content_hash="",
        created_at=now,
    )
    # Set content_hash after construction
    jf.content_hash = content_hash(jf)
    return jf


def _autowrap_script(p: Path) -> JobFile:
    """Create an implicit JobFile for a bare script."""
    orig_suffix = p.suffix
    suffix_lower = orig_suffix.lower()

    # Try original suffix first (handles .R), then lowercase
    prefix = _EXT_MAP.get(orig_suffix) or _EXT_MAP.get(suffix_lower)
    if prefix is None:
        raise ValueError(
            f"Cannot auto-wrap script with extension '{p.suffix}'. "
            f"Supported: {list(_EXT_MAP.keys())}"
        )

    name = p.stem
    command_template = f"{prefix} {p}"
    jf_id = _stable_id(name, str(p))
    now = datetime.now(timezone.utc).isoformat()

    jf = JobFile(
        id=jf_id,
        name=name,
        version=1,
        source_path=str(p),
        command_template=command_template,
        params_schema={},
        backend_prefs=[{"backend": "local"}],
        artifact_patterns=[],
        expectation_contract_id=None,
        content_hash="",
        created_at=now,
    )
    jf.content_hash = content_hash(jf)
    return jf


def _stable_id(name: str, path: str) -> str:
    """Generate a stable ID from name + path."""
    h = hashlib.sha256(f"{name}:{path}".encode()).hexdigest()[:16]
    return f"jf-{h}"


def _sha256(p: Path) -> str:
    """Compute sha256 of a file, returning 'sha256:<hex>'."""
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return f"sha256:{h.hexdigest()}"


def _cast(val: Any, ptype: str) -> Any:
    """Cast a value to the declared param type."""
    if ptype == "int":
        return int(val)
    elif ptype == "float":
        return float(val)
    elif ptype == "bool":
        if isinstance(val, bool):
            return val
        return str(val).lower() in ("true", "1", "yes")
    elif ptype in ("str", "path"):
        return str(val)
    # Unknown type: return as-is
    return val
