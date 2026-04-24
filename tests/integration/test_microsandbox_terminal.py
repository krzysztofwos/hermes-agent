"""Integration tests for the Microsandbox terminal backend.

Requires the ``msb`` CLI on PATH (or MSB_PATH set) and ``/dev/kvm`` readable
on Linux. Run with:

    TERMINAL_ENV=microsandbox pytest tests/integration/test_microsandbox_terminal.py -v

The module skips when either prerequisite is missing so CI and dev machines
without KVM don't see spurious failures.
"""

import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration


def _msb_available() -> bool:
    if os.getenv("MSB_PATH") and os.path.isfile(os.environ["MSB_PATH"]):
        return os.access(os.environ["MSB_PATH"], os.X_OK)
    return shutil.which("msb") is not None


def _kvm_available() -> bool:
    # Non-Linux hosts (macOS Apple Silicon) don't have /dev/kvm but libkrun
    # uses the Hypervisor framework there; skip only when we're on Linux
    # without KVM access.
    if not sys.platform.startswith("linux"):
        return True
    return os.access("/dev/kvm", os.R_OK | os.W_OK)


if not _msb_available():
    pytest.skip("msb not found on PATH or MSB_PATH", allow_module_level=True)

if not _kvm_available():
    pytest.skip(
        "/dev/kvm not readable/writable by this user; add user to 'kvm' group",
        allow_module_level=True,
    )

# Import terminal_tool via importlib to avoid tools/__init__.py side effects.
parent_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(parent_dir))

spec = importlib.util.spec_from_file_location(
    "terminal_tool", parent_dir / "tools" / "terminal_tool.py"
)
terminal_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(terminal_module)

terminal_tool = terminal_module.terminal_tool
cleanup_vm = terminal_module.cleanup_vm


@pytest.fixture(autouse=True)
def _force_microsandbox(monkeypatch):
    monkeypatch.setenv("TERMINAL_ENV", "microsandbox")
    # A minimal Alpine image boots fast and has sh + basic coreutils.
    monkeypatch.setenv("TERMINAL_MICROSANDBOX_IMAGE", "alpine:3")
    monkeypatch.setenv("TERMINAL_CONTAINER_PERSISTENT", "false")


@pytest.fixture()
def task_id(request):
    """Provide a unique task_id and clean up the sandbox after the test."""
    tid = f"msb_test_{request.node.name}"
    yield tid
    cleanup_vm(tid)


def _run(command, task_id, **kwargs):
    result = terminal_tool(command, task_id=task_id, **kwargs)
    return json.loads(result)


class TestMicrosandboxBasic:
    def test_echo(self, task_id):
        r = _run("echo 'Hello from microsandbox!'", task_id)
        assert r["exit_code"] == 0
        assert "Hello from microsandbox!" in r["output"]

    def test_nonzero_exit(self, task_id):
        r = _run("exit 42", task_id)
        assert r["exit_code"] == 42

    def test_kernel_info(self, task_id):
        r = _run("uname -a", task_id)
        assert r["exit_code"] == 0
        assert "Linux" in r["output"]


class TestMicrosandboxFilesystem:
    def test_write_and_read_file(self, task_id):
        _run("echo 'sandboxed' > /tmp/msb_test.txt", task_id)
        r = _run("cat /tmp/msb_test.txt", task_id)
        assert r["exit_code"] == 0
        assert "sandboxed" in r["output"]


class TestMicrosandboxIsolation:
    def test_different_tasks_isolated(self):
        task_a = "msb_test_iso_a"
        task_b = "msb_test_iso_b"
        try:
            _run("echo 'secret' > /tmp/isolated.txt", task_a)
            r = _run("cat /tmp/isolated.txt 2>&1 || echo NOT_FOUND", task_b)
            assert "secret" not in r["output"] or "NOT_FOUND" in r["output"]
        finally:
            cleanup_vm(task_a)
            cleanup_vm(task_b)

    def test_host_env_secrets_not_leaked(self, task_id, monkeypatch):
        """A host env var not in the passthrough allowlist must not reach the VM."""
        monkeypatch.setenv("HERMES_TEST_SECRET_ABC123", "shouldnotleak")
        r = _run("env | grep HERMES_TEST_SECRET_ABC123 || echo ABSENT", task_id)
        assert "ABSENT" in r["output"]
        assert "shouldnotleak" not in r["output"]
