"""ApiClient + ensure_daemon().

ApiClient is a thin HTTP wrapper around the jobctl daemon REST API.
All methods accept/return plain Python dicts and lists.

ensure_daemon() auto-starts `jobctl serve` if the daemon is not reachable.
"""
from __future__ import annotations

import logging
import subprocess
import sys
import time
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_TERMINAL_STATES = {"completed", "failed", "cancelled", "stuck", "timeout"}
_DEFAULT_POLL_INTERVAL = 1.0
_DEFAULT_TIMEOUT = 600.0


class ApiClient:
    """HTTP client for the jobctl daemon.

    Args:
        base_url:   Base URL of the daemon (e.g. "http://127.0.0.1:7421").
        transport:  Optional HTTPX / Starlette TestClient transport (for testing).
    """

    def __init__(self, base_url: str = "http://127.0.0.1:7421", transport=None) -> None:
        self.base_url = base_url.rstrip("/")
        self._transport = transport  # Injected TestClient for testing

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get(self, path: str, **kwargs) -> httpx.Response:
        url = f"{self.base_url}{path}"
        if self._transport is not None:
            # Use TestClient directly
            return self._transport.get(url, **kwargs)
        return httpx.get(url, **kwargs)

    def _post(self, path: str, json: dict | None = None, **kwargs) -> httpx.Response:
        url = f"{self.base_url}{path}"
        if self._transport is not None:
            return self._transport.post(url, json=json, **kwargs)
        return httpx.post(url, json=json, **kwargs)

    def _raise_for(self, resp: httpx.Response, context: str = "") -> None:
        if resp.status_code >= 400:
            raise RuntimeError(
                f"API error {resp.status_code} [{context}]: {resp.text}"
            )

    # ------------------------------------------------------------------
    # /health
    # ------------------------------------------------------------------

    def health(self) -> dict:
        resp = self._get("/health")
        self._raise_for(resp, "health")
        return resp.json()

    # ------------------------------------------------------------------
    # /jobfiles
    # ------------------------------------------------------------------

    def register(self, path: str) -> dict:
        """Register a JobFile from a file path."""
        resp = self._post("/jobfiles", json={"path": path})
        self._raise_for(resp, "register")
        return resp.json()

    def jobfiles(self) -> list[dict]:
        """List all registered JobFiles."""
        resp = self._get("/jobfiles")
        self._raise_for(resp, "jobfiles")
        return resp.json()

    # ------------------------------------------------------------------
    # /runs
    # ------------------------------------------------------------------

    def submit(
        self,
        jobfile_id: str | None = None,
        jobfile_name: str | None = None,
        params: dict | None = None,
        backend_override: dict | None = None,
    ) -> dict:
        """Submit a new run; returns the run dict (includes memory_hint)."""
        body: dict = {"params": params or {}}
        if jobfile_id:
            body["jobfile_id"] = jobfile_id
        if jobfile_name:
            body["jobfile_name"] = jobfile_name
        if backend_override:
            body["backend_override"] = backend_override
        resp = self._post("/runs", json=body)
        self._raise_for(resp, "submit")
        return resp.json()

    def get_run(self, run_id: str) -> dict:
        resp = self._get(f"/runs/{run_id}")
        self._raise_for(resp, f"get_run:{run_id}")
        return resp.json()

    def list_runs(
        self,
        state: str | None = None,
        jobfile_id: str | None = None,
    ) -> list[dict]:
        params: dict = {}
        if state:
            params["state"] = state
        if jobfile_id:
            params["jobfile_id"] = jobfile_id
        resp = self._get("/runs", params=params)
        self._raise_for(resp, "list_runs")
        return resp.json()

    def cancel(self, run_id: str) -> dict:
        resp = self._post(f"/runs/{run_id}/cancel")
        self._raise_for(resp, f"cancel:{run_id}")
        return resp.json()

    def rerun(self, run_id: str) -> dict:
        resp = self._post(f"/runs/{run_id}/rerun")
        self._raise_for(resp, f"rerun:{run_id}")
        return resp.json()

    def logs(self, run_id: str, stream: str = "stdout", tail: int = 200) -> str:
        resp = self._get(f"/runs/{run_id}/logs", params={"stream": stream, "tail": tail})
        self._raise_for(resp, f"logs:{run_id}")
        return resp.text

    def artifacts(self, run_id: str) -> list[dict]:
        resp = self._get(f"/runs/{run_id}/artifacts")
        self._raise_for(resp, f"artifacts:{run_id}")
        return resp.json()

    # ------------------------------------------------------------------
    # /servers
    # ------------------------------------------------------------------

    def servers(self) -> list[dict]:
        resp = self._get("/servers")
        self._raise_for(resp, "servers")
        return resp.json()

    # ------------------------------------------------------------------
    # /feedback
    # ------------------------------------------------------------------

    def feedback(self, run_id: str, kind: str = "note", text: str = "") -> dict:
        """Post user feedback for a run."""
        resp = self._post(
            f"/runs/{run_id}/feedback",
            json={"kind": kind, "text": text},
        )
        self._raise_for(resp, f"feedback:{run_id}")
        return resp.json()

    def list_feedback(self, run_id: str) -> list[dict]:
        resp = self._get(f"/runs/{run_id}/feedback")
        self._raise_for(resp, f"list_feedback:{run_id}")
        return resp.json()

    # ------------------------------------------------------------------
    # /expect
    # ------------------------------------------------------------------

    def expect(self, jobfile_id: str | None = None) -> list[dict]:
        """List expectation contracts."""
        params: dict = {}
        if jobfile_id:
            params["jobfile_id"] = jobfile_id
        resp = self._get("/expect", params=params)
        self._raise_for(resp, "expect")
        return resp.json()

    def confirm_criterion(self, criterion_id: str) -> dict:
        resp = self._post("/expect/confirm", json={"criterion_id": criterion_id})
        self._raise_for(resp, f"confirm_criterion:{criterion_id}")
        return resp.json()

    def propose_criteria(self, run_id: str, feedback_text: str) -> list[dict]:
        resp = self._post(
            "/expect/propose",
            json={"run_id": run_id, "feedback_text": feedback_text},
        )
        self._raise_for(resp, "propose_criteria")
        return resp.json()

    # ------------------------------------------------------------------
    # /memory/query
    # ------------------------------------------------------------------

    def memory_query(
        self,
        jobfile_id: str | None = None,
        name: str | None = None,
        params: dict | None = None,
        input_hashes: dict | None = None,
    ) -> dict:
        qparams: dict = {}
        if jobfile_id:
            qparams["jobfile_id"] = jobfile_id
        if name:
            qparams["name"] = name
        resp = self._get("/memory/query", params=qparams)
        self._raise_for(resp, "memory_query")
        return resp.json()

    # ------------------------------------------------------------------
    # await_run — long-poll until terminal
    # ------------------------------------------------------------------

    def await_run(
        self,
        run_id: str,
        poll_interval: float = _DEFAULT_POLL_INTERVAL,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> dict:
        """Long-poll /runs/{id} until it reaches a terminal state.

        Args:
            run_id:        The run to watch.
            poll_interval: Seconds between polls.
            timeout:       Give up after this many seconds.

        Returns:
            The final run dict.

        Raises:
            TimeoutError:  If the run hasn't finished within *timeout* seconds.
            RuntimeError:  On API error.
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            run = self.get_run(run_id)
            if run.get("state") in _TERMINAL_STATES:
                return run
            time.sleep(poll_interval)

        raise TimeoutError(
            f"Run {run_id} did not reach a terminal state within {timeout}s"
        )


# ---------------------------------------------------------------------------
# ensure_daemon
# ---------------------------------------------------------------------------

def ensure_daemon(
    config: dict | None = None,
    wait_timeout: float = 10.0,
    poll_interval: float = 0.5,
) -> str:
    """Ensure the jobctl daemon is running; start it if not.

    Checks /health on the configured host:port.  If the check fails, spawns
    `jobctl serve` as a background subprocess and waits until /health responds
    (up to *wait_timeout* seconds).

    Args:
        config:       Config dict (keys: daemon_host, daemon_port).
        wait_timeout: Seconds to wait for the spawned daemon to come up.
        poll_interval: Seconds between health-check polls after spawning.

    Returns:
        The base URL of the running daemon, e.g. "http://127.0.0.1:7421".
    """
    if config is None:
        config = {}

    host = config.get("daemon_host", "127.0.0.1")
    port = config.get("daemon_port", 7421)
    base_url = f"http://{host}:{port}"

    # Check if already running
    try:
        resp = httpx.get(f"{base_url}/health", timeout=2.0)
        if resp.status_code == 200:
            return base_url
    except Exception:
        pass

    # Spawn daemon
    logger.info("Starting jobctl daemon on %s", base_url)
    subprocess.Popen(
        [sys.executable, "-m", "jobctl.cli.main", "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )

    # Wait for it to come up
    deadline = time.time() + wait_timeout
    while time.time() < deadline:
        time.sleep(poll_interval)
        try:
            resp = httpx.get(f"{base_url}/health", timeout=2.0)
            if resp.status_code == 200:
                return base_url
        except Exception:
            pass

    logger.warning("Daemon did not come up within %ss", wait_timeout)
    return base_url
