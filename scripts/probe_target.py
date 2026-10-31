"""Read-only environment report; does not claim tiling or device execution verification."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from qfa_mock.build import check_sources
from qfa_mock.extension import compatibility

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--repo',type=Path,required=True)
p.add_argument('--output',type=Path)
a=p.parse_args()
report={'host':platform.system(),'machine':platform.machine(),'python':sys.version.split()[0],
        'compiler':shutil.which('c++'),'torch_npu_available':importlib.util.find_spec('torch_npu') is not None,
        'cann_home':os.environ.get('ASCEND_HOME_PATH'),'real_h2d':'not_run','actual_tiling':'not_run','aicore_out_lse':'not_run'}
try:
    check_sources(a.repo);compatibility(a.repo);report['source_compatible']=True
except Exception as error:
    report['source_compatible']=False;report['source_error']=str(error)
try:
    import torch
    report['torch']=torch.__version__
except ImportError:
    report['torch']=None
report['extension_environment_present']=bool(report['host']=='Linux' and report['torch'] and report['torch_npu_available'] and report['compiler'] and report['cann_home'])
report['note']='Presence checks only. Runtime copies, CANN compilation and actual kernel tiling require target acceptance.'
text=json.dumps(report,indent=2)+'\n'
if a.output:
    a.output.write_text(text)
print(text,end='')
