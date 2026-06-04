"""jobctl — JobFile-native research run gateway.

Public API re-exports for convenience:

    # Enums
    from jobctl import State, Health, Match, ArtifactType

    # Dataclasses
    from jobctl import JobFile, Run, Artifact, Criterion, ExpectationContract, Feedback, Server

    # JobFile helpers
    from jobctl import load_jobfile, resolve_params, render_command

    # Layer selectors
    from jobctl import get_analyzer, get_backend, get_notifiers
"""
from __future__ import annotations

__version__ = "0.1.0"

# ---------------------------------------------------------------------------
# Enums (re-export from db.models)
# ---------------------------------------------------------------------------
from jobctl.db.models import (  # noqa: F401
    ArtifactType,
    Health,
    Match,
    State,
)

# ---------------------------------------------------------------------------
# Dataclasses (re-export from db.models)
# ---------------------------------------------------------------------------
from jobctl.db.models import (  # noqa: F401
    Artifact,
    Criterion,
    ExpectationContract,
    Feedback,
    JobFile,
    Run,
    Server,
)

# ---------------------------------------------------------------------------
# JobFile helpers
# ---------------------------------------------------------------------------
from jobctl.jobfile import (  # noqa: F401
    content_hash,
    input_hashes,
    load_jobfile,
    render_command,
    resolve_params,
)

# ---------------------------------------------------------------------------
# Layer selectors
# ---------------------------------------------------------------------------
from jobctl.analysis.base import get_analyzer  # noqa: F401
from jobctl.backends.base import get_backend  # noqa: F401
from jobctl.notify.notify import get_notifiers  # noqa: F401
