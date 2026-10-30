"""python3 -m qfa_mock {build,generate,validate}."""

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import sys

from .build import ROOT, build_helper, source_digest
from .model import MockError, canonical, fail, load_json, normalize_config, normalize_request, plan_digest
from .provider import MetadataProvider
from .validation import validate_metadata


def main():
    parser = argparse.ArgumentParser(description="Single-invocation Host QFA metadata prototype")
    parser.add_argument("--verbose", action="store_true")
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="explicitly compile original scheduler headers")
    build.add_argument("--repo", required=True, help="source root containing attention/; Git history is not required")
    build.add_argument("--output", default=str(ROOT / "build/metadata_helper"))
    build.add_argument("--compiler", default="c++")
    generate = commands.add_parser("generate", help="generate and validate a new output bundle")
    generate.add_argument("--repo", required=True)
    generate.add_argument("--helper", default=str(ROOT / "build/metadata_helper"))
    generate.add_argument("--config", default=os.environ.get("QFA_METADATA_CONFIG"))
    generate.add_argument("--request", required=True)
    generate.add_argument("--output", required=True, help="new directory; existing output is never overwritten")
    validate = commands.add_parser("validate", help="revalidate an existing metadata bundle")
    validate.add_argument("--repo", required=True)
    validate.add_argument("--bundle", required=True)
    launch = commands.add_parser("launch", help="launch TTK with bound platform/build evidence")
    launch.add_argument("--evidence", required=True)
    launch.add_argument("argv", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    try:
        if args.command == "build":
            result = build_helper(args.repo, args.output, args.compiler)
        elif args.command == "generate":
            if not args.config:
                fail("MISSING_CONFIG", "provide --config or QFA_METADATA_CONFIG; mock never auto-selects a device")
            provider = MetadataProvider(load_json(args.config), args.repo, args.helper)
            result = provider.generate(load_json(args.request))
            result.write(args.output)
            result = result.summary
        elif args.command == "launch":
            from .platform_check import launch
            command = args.argv[1:] if args.argv[:1] == ["--"] else args.argv
            return launch(args.evidence, command)
        else:
            bundle = Path(args.bundle)
            config = normalize_config(load_json(bundle / "config.json"))
            request = normalize_request(load_json(bundle / "request.json"))
            summary = load_json(bundle / "result.json")
            data = (bundle / "metadata.bin").read_bytes()
            if summary.get("source_digest") != source_digest(args.repo, summary.get("dependency_files")):
                fail("SOURCE_CHANGED", "bundle was generated from a different source snapshot")
            if summary.get("metadata_digest") != hashlib.sha256(data).hexdigest():
                fail("DIGEST_MISMATCH", "metadata bytes differ from bundle digest")
            if summary.get("request_digest") != plan_digest(config, request, summary["source_digest"]):
                fail("DIGEST_MISMATCH", "request/config differs from bundle digest")
            result = validate_metadata(data, summary, config, request, args.repo)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (MockError, OSError, ValueError) as error:
        details = {"code": getattr(error, "code", "IO_OR_FORMAT_ERROR"), "message": str(error)}
        if hasattr(error, "missing"):
            details.update(repo=error.repo, missing=error.missing)
            print(json.dumps({"error": details}, ensure_ascii=False, indent=2), file=sys.stderr)
        else:
            print(canonical({"error": details}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
