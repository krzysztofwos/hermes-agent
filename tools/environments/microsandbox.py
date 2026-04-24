"""Microsandbox execution environment for sandboxed command execution.

Microsandbox (https://microsandbox.dev) runs each sandbox as a libkrun microVM
with its own kernel.  Each ``msb exec`` call against a named sandbox runs a
command inside that VM.  There is no long-running daemon — the CLI process
itself boots and manages the VM.

This backend spins up one long-lived sandbox per environment instance via
``msb create``, runs commands with ``msb exec``, and tears down with
``msb stop`` + ``msb remove`` in :meth:`cleanup`.

Host requirements: Linux with ``/dev/kvm`` present and readable (or macOS on
Apple Silicon).  The ``msb`` binary must be on PATH or at :envvar:`MSB_PATH`.
"""

import logging
import os
import re
import shutil
import subprocess
import sys
import uuid
from typing import Optional

from tools.environments.base import BaseEnvironment, _popen_bash
from tools.environments.local import _HERMES_PROVIDER_ENV_BLOCKLIST

logger = logging.getLogger(__name__)


_msb_executable: Optional[str] = None  # resolved once, cached
_ENV_VAR_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def find_msb() -> Optional[str]:
    """Locate the ``msb`` CLI binary.

    Honors :envvar:`MSB_PATH` if set, then falls back to :func:`shutil.which`.
    Returns the absolute path, or ``None`` if msb cannot be found.
    """
    global _msb_executable
    if _msb_executable is not None:
        return _msb_executable

    env_path = os.environ.get("MSB_PATH")
    if env_path and os.path.isfile(env_path) and os.access(env_path, os.X_OK):
        _msb_executable = env_path
        return env_path

    found = shutil.which("msb")
    if found:
        _msb_executable = found
        return found

    return None


def _normalize_forward_env_names(forward_env: list[str] | None) -> list[str]:
    """Return a deduplicated list of valid environment variable names."""
    normalized: list[str] = []
    seen: set[str] = set()
    for item in forward_env or []:
        if not isinstance(item, str):
            logger.warning("Ignoring non-string microsandbox_forward_env entry: %r", item)
            continue
        key = item.strip()
        if not key:
            continue
        if not _ENV_VAR_NAME_RE.match(key):
            logger.warning("Ignoring invalid microsandbox_forward_env entry: %r", item)
            continue
        if key in seen:
            continue
        seen.add(key)
        normalized.append(key)
    return normalized


def _normalize_env_dict(env: dict | None) -> dict[str, str]:
    """Validate and normalize a microsandbox_env dict to ``{str: str}``."""
    if not env:
        return {}
    if not isinstance(env, dict):
        logger.warning("microsandbox_env is not a dict: %r", env)
        return {}
    normalized: dict[str, str] = {}
    for key, value in env.items():
        if not isinstance(key, str) or not _ENV_VAR_NAME_RE.match(key.strip()):
            logger.warning("Ignoring invalid microsandbox_env key: %r", key)
            continue
        key = key.strip()
        if not isinstance(value, str):
            if isinstance(value, (int, float, bool)):
                value = str(value)
            else:
                logger.warning(
                    "Ignoring non-string microsandbox_env value for %r: %r", key, value
                )
                continue
        normalized[key] = value
    return normalized


def _load_hermes_env_vars() -> dict[str, str]:
    """Load ~/.hermes/.env values without failing command execution."""
    try:
        from hermes_cli.config import load_env
        return load_env() or {}
    except Exception:
        return {}


def _ensure_msb_available() -> str:
    """Resolve and health-check the msb CLI before use.

    Returns the absolute path, or raises :class:`RuntimeError` with a pointer
    at microsandbox install docs.
    """
    msb_exe = find_msb()
    if not msb_exe:
        raise RuntimeError(
            "msb executable not found on PATH or at MSB_PATH. "
            "Install microsandbox: curl -fsSL https://install.microsandbox.dev | sh"
        )

    try:
        result = subprocess.run(
            [msb_exe, "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except FileNotFoundError:
        raise RuntimeError(f"msb executable at {msb_exe} could not be executed.")
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"'{msb_exe} --version' timed out. Your microsandbox install may be broken."
        )

    if result.returncode != 0:
        raise RuntimeError(
            f"'{msb_exe} --version' failed (exit {result.returncode}, "
            f"stderr={result.stderr.strip()})."
        )

    # Preflight /dev/kvm on Linux so failures surface with a clear message
    # rather than a deep-inside-libkrun error on the first exec.
    if sys.platform.startswith("linux") and not os.access("/dev/kvm", os.R_OK | os.W_OK):
        raise RuntimeError(
            "/dev/kvm is not readable/writable by this process. "
            "Microsandbox requires KVM access. Ensure the user is in the 'kvm' "
            "group (or the container has --device /dev/kvm and --privileged), "
            "and that the host has virtualization enabled in BIOS."
        )

    return msb_exe


class MicrosandboxEnvironment(BaseEnvironment):
    """Microsandbox-backed execution: one libkrun microVM per environment.

    The VM has its own kernel and rootfs, so commands run here can't read the
    host's or agent's filesystem, environment secrets, or bind mounts.  Only
    variables explicitly whitelisted via ``microsandbox_forward_env`` or
    ``microsandbox_env`` reach the guest.

    Persistence: set ``persistent_filesystem=True`` to keep the sandbox around
    across :meth:`cleanup` (useful for multi-turn tasks).  Default is
    ephemeral — the VM is stopped and removed at cleanup.
    """

    def __init__(
        self,
        image: str,
        cwd: str = "/root",
        timeout: int = 60,
        cpu: float = 0,
        memory: int = 0,
        disk: int = 0,  # accepted for API parity; msb has no per-sandbox disk cap flag
        persistent_filesystem: bool = False,
        task_id: str = "default",
        volumes: list = None,
        forward_env: list[str] | None = None,
        env: dict | None = None,
        network: bool = True,
        max_duration: str | None = None,
        idle_timeout: str | None = None,
    ):
        if cwd == "~":
            cwd = "/root"
        super().__init__(cwd=cwd, timeout=timeout)
        self._persistent = persistent_filesystem
        self._task_id = task_id
        self._forward_env = _normalize_forward_env_names(forward_env)
        self._env = _normalize_env_dict(env)
        self._sandbox_name: Optional[str] = None

        # Fail fast if msb (and on Linux, /dev/kvm) is unavailable.
        self._msb_exe = _ensure_msb_available()

        # Sanitize volumes — same shape as docker_volumes ("HOST:GUEST" strings).
        if volumes is not None and not isinstance(volumes, list):
            logger.warning("microsandbox_volumes config is not a list: %r", volumes)
            volumes = []

        create_args: list[str] = []
        if cpu and cpu > 0:
            # msb accepts an integer count of vCPUs; round up.
            create_args.extend(["--cpus", str(max(1, int(cpu)))])
        if memory and memory > 0:
            create_args.extend(["--memory", f"{memory}M"])
        if not network:
            # Requires msb compiled with the "net" feature; harmless otherwise.
            create_args.append("--no-network")

        for vol in (volumes or []):
            if not isinstance(vol, str):
                logger.warning("Microsandbox volume entry is not a string: %r", vol)
                continue
            vol = vol.strip()
            if not vol or ":" not in vol:
                logger.warning("Microsandbox volume %r missing colon, skipping", vol)
                continue
            create_args.extend(["--volume", vol])

        if max_duration:
            create_args.extend(["--max-duration", str(max_duration)])
        if idle_timeout:
            create_args.extend(["--idle-timeout", str(idle_timeout)])

        # Unique sandbox name so multiple instances (including parallel tasks)
        # don't collide.  ``--replace`` is additional belt-and-braces.
        self._sandbox_name = f"hermes-{task_id}-{uuid.uuid4().hex[:8]}"

        run_cmd = [
            self._msb_exe, "create",
            "--name", self._sandbox_name,
            "--replace",
            "--quiet",
            *create_args,
            image,
        ]
        logger.debug("Starting microsandbox: %s", " ".join(run_cmd))
        result = subprocess.run(
            run_cmd,
            capture_output=True,
            text=True,
            timeout=300,  # image pull + VM boot can take a while
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"msb create failed (exit {result.returncode}): "
                f"{result.stderr.strip() or result.stdout.strip()}"
            )
        logger.info("Created microsandbox %s from %s", self._sandbox_name, image)

        self._init_env_args = self._build_init_env_args()

        # Initialize session snapshot inside the VM.
        self.init_session()

    def _build_init_env_args(self) -> list[str]:
        """Build ``--env KEY=VALUE`` args for injecting host env vars at init_session.

        Only explicit docker-style forwards (``microsandbox_forward_env``) bypass
        the Hermes provider blocklist.  Skill-declared passthroughs still go
        through the blocklist filter (same policy as the Docker backend).
        """
        exec_env: dict[str, str] = dict(self._env)

        explicit_forward_keys = set(self._forward_env)
        passthrough_keys: set[str] = set()
        try:
            from tools.env_passthrough import get_all_passthrough
            passthrough_keys = set(get_all_passthrough())
        except Exception:
            pass
        forward_keys = explicit_forward_keys | (
            passthrough_keys - _HERMES_PROVIDER_ENV_BLOCKLIST
        )
        hermes_env = _load_hermes_env_vars() if forward_keys else {}
        for key in sorted(forward_keys):
            value = os.getenv(key)
            if value is None:
                value = hermes_env.get(key)
            if value is not None:
                exec_env[key] = value

        args = []
        for key in sorted(exec_env):
            args.extend(["--env", f"{key}={exec_env[key]}"])
        return args

    def _run_bash(
        self,
        cmd_string: str,
        *,
        login: bool = False,
        timeout: int = 120,
        stdin_data: str | None = None,
    ) -> subprocess.Popen:
        """Spawn a bash process inside the microsandbox VM via ``msb exec``."""
        assert self._sandbox_name, "Sandbox not created"
        cmd = [self._msb_exe, "exec", "--quiet"]

        # Only inject env on the bootstrap call; afterwards, the snapshot file
        # (captured during init_session) carries state across invocations.
        if login:
            cmd.extend(self._init_env_args)

        cmd.append(self._sandbox_name)
        cmd.append("--")

        if login:
            cmd.extend(["bash", "-l", "-c", cmd_string])
        else:
            cmd.extend(["bash", "-c", cmd_string])

        return _popen_bash(cmd, stdin_data)

    def cleanup(self):
        """Stop and remove the microsandbox VM (unless ``persistent_filesystem``)."""
        if not self._sandbox_name:
            return

        name = self._sandbox_name
        if self._persistent:
            logger.info("Leaving microsandbox %s running (persistent=True)", name)
            self._sandbox_name = None
            return

        # Best-effort stop + remove; don't block caller on teardown.
        try:
            subprocess.Popen(
                [self._msb_exe, "stop", name, "--quiet"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as e:
            logger.warning("Failed to stop microsandbox %s: %s", name, e)

        try:
            subprocess.Popen(
                f"sleep 3 && {self._msb_exe} remove {name} --quiet "
                f">/dev/null 2>&1 &",
                shell=True,
            )
        except Exception:
            pass

        self._sandbox_name = None
