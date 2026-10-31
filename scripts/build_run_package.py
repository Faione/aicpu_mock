"""Bundle an already built CANN custom .run and matching extension wheel."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from qfa_mock.package import build_run

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--opp-package',required=True)
    p.add_argument('--opp-manifest',required=True)
    p.add_argument('--wheel',required=True)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    try:
        print(json.dumps(build_run(a.opp_package,a.opp_manifest,a.wheel,a.output),indent=2))
    except Exception as error:
        p.exit(1,str(error)+'\n')
