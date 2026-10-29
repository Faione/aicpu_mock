"""Package existing deployment records; never infer the platform from core counts."""

import argparse
import os
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qfa_mock.model import canonical, load_json
from qfa_mock.platform_check import digest_file, verify_record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "platform", "qfa-binary", "build-record", "mapping", "model-version", "cwd", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--tiling")
    parser.add_argument("--run-id", default=str(uuid.uuid4()))
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    mapping = load_json(args.mapping)
    artifacts = {}
    for name in ("model", "platform", "qfa_binary", "build_record", "mapping", "tiling"):
        value = getattr(args, name)
        if value:
            path = Path(value).resolve()
            artifacts[name] = {"path": str(path), "sha256": digest_file(path)}
    artifacts["model"]["version"] = args.model_version
    environment = {key: os.environ[key] for key in
                   ("ASCEND_TOOLKIT_HOME", "ASCEND_OPP_PATH", "ASCEND_CUSTOM_OPP_PATH", "LD_LIBRARY_PATH")
                   if key in os.environ}
    environment.setdefault("LD_LIBRARY_PATH", "")
    record = {"protocol_version": 1, "run_id": args.run_id,
              "launch": {"command": command, "cwd": str(Path(args.cwd).resolve()), "environment": environment,
                         **{key: mapping[key] for key in ("sim_soc", "model_name", "cann_platform")}},
              "artifacts": artifacts}
    output = Path(args.output).resolve()
    # Validate a private staged record before exposing the requested output.
    import tempfile
    with tempfile.TemporaryDirectory(prefix="qfa-evidence-") as directory:
        staged = Path(directory) / "evidence.json"
        staged.write_text(canonical(record) + "\n")
        verify_record(staged)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(canonical(record) + "\n")
    print(output)


if __name__ == "__main__":
    main()
