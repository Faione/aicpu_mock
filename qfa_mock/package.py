"""Validate real release inputs and bundle a custom CANN .run plus its matching JIT wheel."""
import configparser
import hashlib
import io
import json
from pathlib import Path
import re
import tarfile
import zipfile

from .extension import sha
from .model import canonical, fail


def file_sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def inspect_wheel(path):
    with zipfile.ZipFile(path) as wheel:
        names = wheel.namelist()
        if len(names) != len(set(names)) or any(Path(n).is_absolute() or '..' in Path(n).parts for n in names):
            fail('PACKAGE_INVALID', 'unsafe or duplicate wheel entries')
        suffix = '/csrc/attention/qfa_mock/manifest.json'
        manifests = [n for n in names if n.endswith(suffix)]
        if len(manifests) != 1:
            fail('PACKAGE_INVALID', 'wheel must contain exactly one extension_cpp payload')
        prefix = manifests[0][:-len('manifest.json')]
        package = prefix.split('/')[0]
        manifest = json.loads(wheel.read(manifests[0]))
        if manifest.get('mode') != 'extension_cpp' or manifest.get('format') != 1:
            fail('PACKAGE_INVALID', 'unsupported extension manifest')
        for name, expected in manifest['files'].items():
            if Path(name).is_absolute() or '..' in Path(name).parts or sha(wheel.read(prefix+name)) != expected:
                fail('PACKAGE_INVALID', 'missing/changed wheel payload: '+name)
        for name, expected in manifest['framework_files'].items():
            if Path(name).is_absolute() or '..' in Path(name).parts or sha(wheel.read('cann_ops_transformer/'+name)) != expected:
                fail('PACKAGE_INVALID', 'base loader framework missing/changed: '+name)
        runtime = package+'/ops/attention/quant_flash_attn/_qfa_metadata_backend.py'
        if sha(wheel.read(runtime)) != manifest['runtime_sha256']:
            fail('PACKAGE_INVALID', 'wheel runtime identity differs')
        audit = package+'/ops/attention/quant_flash_attn/_qfa_metadata_audit.py'
        if sha(wheel.read(audit)) != manifest['audit_sha256']:
            fail('PACKAGE_INVALID', 'wheel audit module identity differs')
        cpp = wheel.read(package+'/csrc/attention/quant_flash_attn.cpp').decode()
        if sha(cpp.encode()) != manifest['entry_cpp_sha256']:
            fail('PACKAGE_INVALID', 'wheel binding source identity differs')
        python = wheel.read(package+'/ops/attention/quant_flash_attn/quant_flash_attn.py').decode().replace('\r\n','\n')
        python = re.sub(r'\b(from|import) cann_ops_transformer_[A-Za-z0-9_]+\.op_builder', r'\1 cann_ops_transformer.op_builder', python)
        if sha(python.encode()) != manifest['entry_python_sha256']:
            fail('PACKAGE_INVALID', 'wheel Python entry identity differs')
        if 'MakeMetadataBinding(&quant_flash_attn_metadata)' not in cpp or 'm.def("quant_flash_attn", &quant_flash_attn' not in cpp:
            fail('PACKAGE_INVALID', 'wheel must preserve both real operator exports')
        entries = [n for n in names if n.endswith('.dist-info/entry_points.txt')]
        if package != 'cann_ops_transformer':
            if len(entries) != 1:
                fail('PACKAGE_INVALID', 'vendor wheel requires entry point metadata')
            config = configparser.ConfigParser(interpolation=None)
            config.read_string(wheel.read(entries[0]).decode())
            value = config.get('cann_ops_transformer.ops', 'quant_flash_attn', fallback='')
            if not value.startswith(package+'.ops.'):
                fail('PACKAGE_INVALID', 'vendor wheel does not export qfa through entry points')
        return {'package':package,'build_id':manifest['build_id'],'adapter_version':manifest['adapter_version'],
                'base_framework_embedded':True,'wheel_sha256':file_sha(path),'behavior_digest':sha(wheel.read(prefix+'compatibility.json')),'jit_required':True,'assets_modified':False}


def build_run(opp, opp_manifest, wheel, output):
    opp, wheel, output = Path(opp), Path(wheel), Path(output)
    if output.exists():
        fail('OUTPUT_EXISTS', 'package output must be new')
    info = inspect_wheel(wheel)
    declaration = json.loads(Path(opp_manifest).read_text())
    required = {'format','package_type','sha256','vendor','host_arch','kernel_arch','operators','behavior_digest','installed_files'}
    if not required <= set(declaration) or declaration['format'] != 1 or declaration['package_type'] != 'cann_custom_run':
        fail('PACKAGE_INVALID', 'provide a CANN custom release manifest')
    with opp.open('rb') as stream:
        script_header = stream.read(2)
    if declaration['sha256'] != file_sha(opp) or script_header != b'#!':
        fail('PACKAGE_INVALID', 'CANN .run identity mismatch')
    if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*', declaration['vendor']):
        fail('PACKAGE_INVALID', 'invalid vendor name')
    if declaration['host_arch'] not in ('aarch64','x86_64') or declaration['kernel_arch'] not in ('arch92','arch35'):
        fail('PACKAGE_INVALID', 'unsupported release architecture')
    if not {'quant_flash_attn','quant_flash_attn_metadata'} <= set(declaration['operators']):
        fail('PACKAGE_INVALID', 'CANN payload must declare qfa and native metadata dependencies')
    inventory = declaration['installed_files']
    if not isinstance(inventory, dict) or not inventory or not any(k.startswith('op_api/') for k in inventory) or not any(k.startswith('op_impl/') for k in inventory):
        fail('PACKAGE_INVALID', 'CANN manifest needs a complete installed file inventory')
    for name, digest in inventory.items():
        if Path(name).is_absolute() or '..' in Path(name).parts or not isinstance(digest,str) or not re.fullmatch('[0-9a-f]{64}',digest):
            fail('PACKAGE_INVALID', 'invalid CANN installed file identity')
    if declaration['behavior_digest'] != info['behavior_digest']:
        fail('PACKAGE_INVALID', 'CANN build behavior identity differs from the wheel adapter')
    if not info['package'].endswith('_'+declaration['vendor']):
        fail('PACKAGE_INVALID', 'wheel vendor and CANN vendor differ')
    release = {'format':1,'wheel':wheel.name,'wheel_info':info,'opp':declaration,
               'payloads':{'opp.run':file_sha(opp),wheel.name:file_sha(wheel)}}
    installer = Path(__file__).with_name('run_installer.py').read_bytes()
    release['payloads']['install.py'] = sha(installer)
    small = {'install.py':installer,'release.json':(canonical(release)+'\n').encode()}
    header='''#!/bin/sh
set -eu
qfa_tmp=$(mktemp -d)
trap 'rm -rf "$qfa_tmp"' EXIT HUP INT TERM
qfa_python=${QFA_PACKAGE_PYTHON:-python3}
tail -n +9 "$0" | tar -xz -C "$qfa_tmp"
"$qfa_python" "$qfa_tmp/install.py" "$qfa_tmp" "$@"
exit 0
'''
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('xb') as stream:
        stream.write(header.encode())
        with tarfile.open(fileobj=stream,mode='w|gz') as archive:
            for name,path in (('opp.run',opp),(wheel.name,wheel)):
                item=tarfile.TarInfo(name);item.size=path.stat().st_size;item.mode=0o700 if name=='opp.run' else 0o600
                with path.open('rb') as source:
                    archive.addfile(item,source)
            for name,data in small.items():
                item=tarfile.TarInfo(name);item.size=len(data);item.mode=0o600
                archive.addfile(item,io.BytesIO(data))
    output.chmod(0o755)
    return {'path':str(output.resolve()),'sha256':file_sha(output),'build_id':info['build_id'],
            'jit_required':True,'cann_payload_verified':'digest and declared operator inventory; target execution not run'}
