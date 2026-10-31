"""Self-contained release installer. Embedded by qfa_mock.package; not an SDK emulator."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('payload')
    parser.add_argument('--inspect',action='store_true',help='verify the archive without installing or claiming SDK readiness')
    parser.add_argument('--prefix',type=Path)
    args=parser.parse_args()
    payload=Path(args.payload)
    release=json.loads((payload/'release.json').read_text())
    for name,expected in release['payloads'].items():
        relative=Path(name)
        if relative.is_absolute() or '..' in relative.parts or hashlib.sha256((payload/name).read_bytes()).hexdigest()!=expected:
            raise RuntimeError('package checksum mismatch: '+name)
    if args.inspect:
        print(json.dumps(release,indent=2));return
    if args.prefix is None or not args.prefix.is_absolute() or args.prefix.exists():
        raise RuntimeError('--prefix must be an absolute new directory; existing installs are never overwritten')
    machine={'arm64':'aarch64','amd64':'x86_64'}.get(platform.machine(),platform.machine())
    if platform.system()!='Linux' or machine!=release['opp']['host_arch']:
        raise RuntimeError('install on the declared Linux host architecture')
    # Fail before touching the installation if target prerequisites are unavailable.
    subprocess.run([sys.executable,'-c','import torch, torch_npu; import torch.utils.cpp_extension'],check=True)
    subprocess.run([sys.executable,'-m','pip','--version'],check=True,stdout=subprocess.DEVNULL)
    prefix=args.prefix
    prefix.mkdir(parents=True)
    (prefix/'INSTALL_INCOMPLETE').write_text('Installation has not passed import/JIT checks.\n')
    opp=prefix/'opp';opp.mkdir()
    subprocess.run([str(payload/'opp.run'),'--quiet', '--install-path='+str(opp)],check=True)
    vendor=opp/'vendors'/release['opp']['vendor']
    if not (vendor/'op_api').is_dir() or not (vendor/'op_impl').is_dir():
        raise RuntimeError('CANN installer did not create the declared vendor payload')
    for name,expected in release['opp']['installed_files'].items():
        path=vendor/name
        if not path.resolve().is_relative_to(vendor.resolve()) or not path.is_file():
            raise RuntimeError('CANN installation is incomplete: '+name)
        digest=hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(1024*1024),b''):
                digest.update(block)
        if digest.hexdigest()!=expected:
            raise RuntimeError('CANN installed file differs: '+name)
    python=prefix/'python'
    subprocess.run([sys.executable,'-m','pip','install','--no-deps','--no-compile','--target',str(python),str(payload/release['wheel'])],check=True)
    env=dict(os.environ)
    env['PYTHONPATH']=str(python)+os.pathsep+env.get('PYTHONPATH','')
    env['ASCEND_CUSTOM_OPP_PATH']=str(vendor)+os.pathsep+env.get('ASCEND_CUSTOM_OPP_PATH','')
    env['LD_LIBRARY_PATH']=str(vendor/'op_api/lib')+os.pathsep+env.get('LD_LIBRARY_PATH','')
    # A fresh process checks selected provider, complete payload and actual JIT extension.
    package=release['wheel_info']['package']
    probe='''import importlib, importlib.metadata as metadata, pathlib
import cann_ops_transformer.ops as ops
expected, prefix, identity = __import__('sys').argv[1:]
providers=[x for x in metadata.entry_points(group='cann_ops_transformer.ops') if x.name=='quant_flash_attn']
if len(providers)!=1 or not providers[0].value.startswith(expected+'.'):
    raise RuntimeError('ambiguous or incorrect qfa entry point provider')
f=ops.quant_flash_attn_metadata
if not f.__module__.startswith(expected+'.'):
    raise RuntimeError('base package resolved another qfa provider')
m=importlib.import_module(f.__module__)
if not pathlib.Path(m.__file__).resolve().is_relative_to(pathlib.Path(prefix).resolve()):
    raise RuntimeError('qfa imported outside this installation')
backend=importlib.import_module(m.__package__+'._qfa_metadata_backend')
backend.verify_payload()
ext=m.quant_flash_attn_op_builder.load()
info=ext._qfa_mock_info()
if info['build_id']!=identity or info['cpu_acceptance_only']:
    raise RuntimeError('stale or incomplete C++ extension')
'''
    subprocess.run([sys.executable,'-c',probe,package,str(python),release['wheel_info']['build_id']],env=env,check=True)
    bin_dir=prefix/'bin';bin_dir.mkdir()
    # Write an activation script using absolute installation paths, never development paths.
    q=shlex.quote
    activation='\n'.join(['# Source this file before starting TTK workers.',
        'export QFA_MOCK_RELEASE='+q(str(prefix/'release.json')),
        'export PYTHONPATH='+q(str(python))+'${PYTHONPATH:+:$PYTHONPATH}',
        'export ASCEND_CUSTOM_OPP_PATH='+q(str(vendor))+'${ASCEND_CUSTOM_OPP_PATH:+:$ASCEND_CUSTOM_OPP_PATH}',
        'export LD_LIBRARY_PATH='+q(str(vendor/'op_api/lib'))+'${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}',
        '# Mock is off by default. Select QFA_METADATA_BACKEND and QFA_METADATA_CONFIG explicitly.',''])
    (bin_dir/'set_env.bash').write_text(activation)
    (prefix/'release.json').write_text(json.dumps(release,indent=2)+'\n')
    (prefix/'INSTALL_INCOMPLETE').unlink()
    print('Installed and JIT checked. Source '+str(bin_dir/'set_env.bash'))


if __name__=='__main__':
    try:
        main()
    except Exception as error:
        print('QFA package installation failed: '+str(error),file=sys.stderr)
        sys.exit(1)
