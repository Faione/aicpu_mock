"""Build the selected vendor wheel and verify all mock sources are actually packaged."""
import argparse
import json
import os
import re
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from qfa_mock.package import inspect_wheel

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--vendor',required=True)
    p.add_argument('--output',type=Path,required=True,help='new wheel output directory')
    a=p.parse_args()
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", a.vendor):
        p.error("invalid vendor name")
    if a.output.exists():
        p.error('output directory must be new')
    a.output=a.output.resolve();a.output.mkdir(parents=True)
    env={**os.environ,'TORCH_EXTENSION_OPS':'quant_flash_attn','TORCH_EXTENSION_VENDOR':a.vendor}
    result=subprocess.run([sys.executable,'setup.py','bdist_wheel','--dist-dir',str(a.output)],cwd=a.repo.resolve()/'torch_extension',env=env)
    if result.returncode:
        p.exit(result.returncode,'wheel build failed; output is not a successful release\n')
    wheels=list(a.output.glob('*.whl'))
    if len(wheels)!=1:
        p.error('expected exactly one newly built wheel')
    info=inspect_wheel(wheels[0])
    (a.output/"wheel_manifest.json").write_text(json.dumps(info,indent=2)+"\n")
    print(json.dumps(info,indent=2))
