"""Shared per-worker provider, independent of either assets integration style."""

import os
from pathlib import Path
import threading

from .build import ROOT, verify_build
from .model import canonical, fail, load_json, normalize_config
from .provider import MetadataProvider
from .telemetry import event

_providers = {}
_provider_lock = threading.Lock()

def _after_fork():
    global _provider_lock
    _providers.clear()
    _provider_lock = threading.Lock()

if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork)


def get_provider(expected_repo=None):
    path = os.environ.get("QFA_METADATA_CONFIG")
    repo = os.environ.get("QFA_REPO")
    helper = os.environ.get("QFA_MOCK_HELPER", str(ROOT / "build/metadata_helper"))
    if not path or not repo:
        fail("MISSING_CONFIG", "TTK adapter requires QFA_METADATA_CONFIG and QFA_REPO")
    repo, helper = Path(repo).resolve(), Path(helper).resolve()
    if expected_repo is not None and repo != Path(expected_repo).resolve():
        fail("WORKER_CONFIG_CHANGED", f"registered checkout is {expected_repo}, QFA_REPO is {repo}")
    config = normalize_config(load_json(path))
    identity = canonical({"repo": str(repo), "helper": str(helper), "config": config})
    pid = os.getpid()
    with _provider_lock:
        state = _providers.get(pid)
        if state is not None and state[0] != identity:
            fail("WORKER_CONFIG_CHANGED", "worker checkout, normalized configuration or helper path changed; restart the worker")
        manifest = verify_build(helper, repo)
        build_identity = canonical(manifest)
        if state is not None:
            if state[1] != build_identity:
                fail("STALE_BUILD", "worker build/source identity changed; restart the worker")
            event("worker_reused", config_path=str(Path(path).resolve()), source_digest=manifest["source_digest"])
            return state[2]
        provider = MetadataProvider(config, repo, helper)
        _providers[pid] = (identity, build_identity, provider)
        event("worker_initialized", repo=str(repo), config_path=str(Path(path).resolve()),
              config=config, helper=str(helper), build=manifest)
        return provider
