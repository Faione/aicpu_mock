"""Single worker provider: immutable CPU bytes, bounded cache, no native fallback."""

from collections import OrderedDict
from dataclasses import dataclass, replace
import hashlib
import json
import logging
from pathlib import Path
import subprocess
import sys
import tempfile
import threading

from .build import verify_build
from .model import (ALGORITHM_VERSION, canonical, fail, normalize_config, normalize_request,
                    plan_digest, row_size, wire_request)
from .validation import validate_metadata


@dataclass(frozen=True)
class MetadataResult:
    data: bytes
    summary_json: str
    request_json: str
    config_json: str
    cache_hit: bool = False

    @property
    def summary(self):
        return {**json.loads(self.summary_json), "cache_hit": self.cache_hit}

    def cpu_tensor(self):
        import torch

        if sys.byteorder != "little":
            fail("UNSUPPORTED_HOST", "Torch adapter currently requires a little-endian host")
        # Never expose the cache's mutable backing storage.
        return torch.frombuffer(bytearray(self.data), dtype=torch.int32).clone().reshape(self.summary["shape"])

    def device_tensor(self, anchor):
        return self.cpu_tensor().to(device=anchor.device)

    def write(self, directory):
        directory = Path(directory)
        # A bundle must be new so a failed/repeated call cannot masquerade as an old success.
        directory.mkdir(parents=True, exist_ok=False)
        (directory / "metadata.bin").write_bytes(self.data)
        (directory / "result.json").write_text(canonical(self.summary) + "\n", encoding="utf-8")
        (directory / "request.json").write_text(self.request_json + "\n", encoding="utf-8")
        (directory / "config.json").write_text(self.config_json + "\n", encoding="utf-8")


class MetadataProvider:
    def __init__(self, config, repo, helper, timeout=30):
        self._config_json = canonical(normalize_config(config))
        self.repo, self.helper = Path(repo).resolve(), Path(helper).resolve()
        self.timeout = timeout
        self._cache = OrderedDict()
        self._lock = threading.Lock()
        verify_build(self.helper, self.repo)

    @property
    def config(self):
        return json.loads(self._config_json)

    def generate(self, request):
        r, config = normalize_request(request), self.config
        row_size(config, r)
        with self._lock:
            # Cache hits must also reject a stale/replaced helper or changed source.
            manifest = verify_build(self.helper, self.repo)
            key = plan_digest(config, r, manifest["source_digest"])
            if config["cache"]["enabled"] and key in self._cache:
                self._cache.move_to_end(key)
                result = replace(self._cache[key], cache_hit=True)
                self._log(result)
                return result
            with tempfile.TemporaryDirectory(prefix="qfa-metadata-") as directory:
                output = Path(directory) / "metadata.bin"
                try:
                    process = subprocess.run([str(self.helper), str(output)], input=wire_request(config, r),
                                             capture_output=True, text=True, timeout=self.timeout)
                except subprocess.TimeoutExpired:
                    fail("HELPER_TIMEOUT", f"helper exceeded {self.timeout} seconds")
                if process.returncode:
                    fail("HELPER_FAILED", process.stderr.strip() or f"helper exit code {process.returncode}")
                try:
                    summary = json.loads(process.stdout)
                    data = output.read_bytes()
                except (ValueError, OSError) as error:
                    fail("INVALID_METADATA", f"invalid helper output: {error}")
            checks = validate_metadata(data, summary, config, r, self.repo)
            summary.update({"request_digest": key, "metadata_digest": hashlib.sha256(data).hexdigest(),
                            "source_digest": manifest["source_digest"], "target": config["target"],
                            "validation": checks})
            result = MetadataResult(data, canonical(summary), canonical(r), self._config_json)
            if config["cache"]["enabled"]:
                self._cache[key] = result
                while len(self._cache) > config["cache"]["max_entries"]:
                    self._cache.popitem(last=False)
            self._log(result)
            return result

    @staticmethod
    def _log(result):
        s = result.summary
        logging.getLogger("qfa_mock").info("mock_aicpu target=%s request=%s algorithm=%s cache_hit=%s header=%s AIC=%s AIV=%s used_words=%s",
                                           s["target"], s["request_digest"], ALGORITHM_VERSION, result.cache_hit,
                                           s["header"], s["active_aic"], s["active_aiv"], s["used_words"])
