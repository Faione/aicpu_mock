"""Apply/reverse the same-name Torch extension integration; original assets stay intact."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qfa_mock.extension import integrate
from qfa_mock.model import MockError


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', required=True)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--reverse', action='store_true')
    args = parser.parse_args()
    try:
        print(json.dumps(integrate(args.repo, args.check, args.reverse), indent=2))
        return 0
    except (MockError, OSError, ValueError) as error:
        print(json.dumps({'error':{'code':getattr(error,'code','IO_OR_FORMAT_ERROR'),'message':str(error)}}),file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
