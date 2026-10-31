"""Apply/reverse the one maintained assets patch (a Git checkout is not required)."""

import argparse
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
PATCH = ROOT / "patches/qfa_assets_mock.patch"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="ops-transformer source directory; .git is optional")
    parser.add_argument("--check", action="store_true", help="check applicability without editing")
    parser.add_argument("--reverse", action="store_true", help="remove the patch")
    args = parser.parse_args()
    if not args.reverse and (Path(args.repo) / ".qfa_mock_extension_state.json").exists():
        parser.error("extension_cpp is installed; remove it before applying assets_explicit")
    # Match each target file's line endings, including on reverse application.
    # Git's whitespace-tolerant matching alone would restore changed CRLF lines as LF.
    lines = []
    newline = b"\n"
    repo = Path(args.repo).resolve()
    for line in PATCH.read_bytes().splitlines():
        if line.startswith(b"+++ b/"):
            data = (repo / line[len(b"+++ b/"):].decode()).read_bytes()
            newline = b"\r\n" if data.count(b"\r\n") > data.count(b"\n") // 2 else b"\n"
        is_header = line.startswith((b"--- ", b"+++ "))
        ending = newline if line[:1] in (b" ", b"+", b"-") and not is_header else b"\n"
        lines.append(line + ending)
    with tempfile.TemporaryDirectory(prefix="qfa-assets-patch-") as directory:
        localized = Path(directory) / "assets.patch"
        localized.write_bytes(b"".join(lines))
        return apply(args, localized)


def apply(args, patch_path):
    command = ["git", "-C", str(Path(args.repo).resolve()), "apply", "--ignore-space-change"]
    if args.reverse:
        command.append("--reverse")
    check = subprocess.run([*command, "--check", str(patch_path)], capture_output=True, text=True)
    if check.returncode:
        inverse = ["git", "-C", str(Path(args.repo).resolve()), "apply", "--ignore-space-change"]
        if not args.reverse:
            inverse.append("--reverse")
        already = subprocess.run([*inverse, "--check", str(patch_path)], capture_output=True, text=True)
        if already.returncode == 0:
            print("Already removed" if args.reverse else "Already applied")
            return 0
        print(check.stderr.strip(), file=sys.stderr)
        return check.returncode
    if args.check:
        print("Patch can be removed" if args.reverse else "Patch can be applied")
        return 0
    result = subprocess.run([*command, str(patch_path)], capture_output=True, text=True)
    if result.returncode:
        print(result.stderr.strip(), file=sys.stderr)
        return result.returncode
    print("Removed assets mock patch" if args.reverse else "Applied assets mock patch")
    return 0


if __name__ == "__main__":
    sys.exit(main())
