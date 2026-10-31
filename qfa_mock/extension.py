"""Install a scoped, reversible extension integration and self-contained sources."""
import ast
import base64
import hashlib
import json
from pathlib import Path
import re

from .build import ROOT, COMPILE_SOURCES, check_sources
from .model import canonical, fail

CPP = 'attention/quant_flash_attn/torch_extension/csrc/quant_flash_attn.cpp'
PYTHON = 'attention/quant_flash_attn/torch_extension/quant_flash_attn.py'
SETUP = 'torch_extension/setup.py'
LOCAL = 'attention/quant_flash_attn/torch_extension/'
PAYLOAD = LOCAL + 'csrc/qfa_mock/'
JOURNAL = '.qfa_mock_extension_state.json'
BINDING = 'm.def("quant_flash_attn_metadata", &quant_flash_attn_metadata, "quant_flash_attn_metadata");'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def compatibility(repo):
    known = json.loads((ROOT / 'compatibility/qfa_host_v2.json').read_text())
    changed = [name for name, value in known['files'].items() if sha((Path(repo) / name).read_bytes()) != value]
    if changed:
        fail('UPSTREAM_INCOMPATIBLE', 'adapter qfa-host-v2 needs review for changed behavior files: ' + ', '.join(changed))
    return known


def _node(tree, kind, name):
    nodes = [n for n in tree.body if isinstance(n, kind) and n.name == name]
    if len(nodes) != 1:
        fail('UPSTREAM_INCOMPATIBLE', 'expected one definition: ' + name)
    return nodes[0]


def _edit_python(text):
    tree = ast.parse(text)
    lines = text.splitlines(keepends=True)
    edits = []
    builder = _node(tree, ast.ClassDef, 'QuantFlashAttnOpBuilder')
    sources = _node(builder, ast.FunctionDef, 'sources')
    returns = [n for n in sources.body if isinstance(n, ast.Return)]
    if len(returns) != 1 or ast.literal_eval(returns[0].value) != ['csrc/attention/quant_flash_attn.cpp']:
        fail('UPSTREAM_INCOMPATIBLE', 'unsupported qfa builder sources')
    if any(isinstance(n, ast.FunctionDef) and n.name == 'include_paths' for n in builder.body):
        fail('UPSTREAM_INCOMPATIBLE', 'qfa builder already defines include_paths')
    edits.append((sources.lineno-1, sources.end_lineno, '''    def sources(self):
        return ["csrc/attention/quant_flash_attn.cpp",
                "csrc/attention/qfa_mock/metadata_core.cpp",
                "csrc/attention/qfa_mock/qfa_metadata_binding.cpp"]

    def include_paths(self):
        return super().include_paths() + ["csrc/attention/qfa_mock",
                                        "csrc/attention/qfa_mock/upstream"]
'''))
    function = _node(tree, ast.FunctionDef, 'quant_flash_attn_metadata')
    allocations = [n for n in function.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'max_schedule_size' for t in n.targets)]
    if len(allocations) != 1:
        fail('UPSTREAM_INCOMPATIBLE', 'metadata allocation point is not unique')
    n = allocations[0]
    if ast.unparse(n.value) != '_calculate_max_schedule_size(batch_size, num_heads_q)':
        fail('UPSTREAM_INCOMPATIBLE', 'metadata allocation contract changed')
    edits.append((n.lineno-1, n.end_lineno, '''    from ._qfa_metadata_backend import prepare_metadata
    max_schedule_size = prepare_metadata(quant_flash_attn_op_builder, batch_size, num_heads_q,
                                         _calculate_max_schedule_size)
'''))
    qfa = _node(tree, ast.FunctionDef, 'quant_flash_attn')
    fields = {arg.arg for arg in qfa.args.args + qfa.args.kwonlyargs}
    required = {'q','k','v','quant_mode','sinks','v_tail','block_table_tail','seqused_v_tail','win_left','win_right','max_seqlen_q','max_seqlen_kv','layout_q','layout_q_descale','layout_kv','layout_out','mask_mode'}
    if not required <= fields:
        fail('UPSTREAM_INCOMPATIBLE', 'main qfa signature changed')
    insertion = qfa.body[0].end_lineno if isinstance(qfa.body[0], ast.Expr) and isinstance(qfa.body[0].value, ast.Constant) and isinstance(qfa.body[0].value.value, str) else qfa.lineno
    edits.append((insertion, insertion, '''    from ._qfa_metadata_backend import validate_qfa_call
    validate_qfa_call(locals())
'''))
    for start, end, replacement in sorted(edits, reverse=True):
        lines[start:end] = [replacement]
    result = ''.join(lines)
    ast.parse(result)
    return result


def _edit_setup(text):
    tree = ast.parse(text)
    function = _node(tree, ast.FunctionDef, '_collect_op')
    blocks = [n for n in function.body if isinstance(n, ast.If) and ast.unparse(n.test) == 'os.path.isdir(csrc_dir)']
    if len(blocks) != 1:
        fail('UPSTREAM_INCOMPATIBLE', 'setup csrc collection point is not unique')
    n = blocks[0]
    replacement = '''    if os.path.isdir(csrc_dir):
        # QFA_MOCK_EXTENSION_V1: include nested sources, headers and identity data.
        for root, _, filenames in os.walk(csrc_dir):
            for filename in sorted(filenames):
                if filename.endswith((".cpp", ".h", ".json", ".py")) or filename == "LICENSE":
                    source = os.path.join(root, filename)
                    _op_cpp_files.append((os.path.join("csrc", op_cat,
                        os.path.relpath(source, csrc_dir)), source))
'''
    lines = text.splitlines(keepends=True)
    lines[n.lineno-1:n.end_lineno] = [replacement]
    result = ''.join(lines)
    tree = ast.parse(result)
    build_class = _node(tree, ast.ClassDef, 'BuildPyWithOps')
    run = _node(build_class, ast.FunctionDef, 'run')
    lines = result.splitlines(keepends=True)
    lines[run.end_lineno:run.end_lineno] = ['''        if PACKAGE_NAME != BASE_PACKAGE_NAME:
            from qfa_mock_wheel_support import copy_loader_framework
            copy_loader_framework(_src_path, self.build_lib)
''']
    result = ''.join(lines)
    ast.parse(result)
    return result


def changes(repo):
    repo = Path(repo).resolve()
    check_sources(repo)
    compatibility(repo)
    assets = repo / 'attention/quant_flash_attn/tests/assets'
    if any('QFA_MOCK_ASSETS_V1' in p.read_text() for p in assets.rglob('*.py')):
        fail('INTEGRATION_CONFLICT', 'remove the assets_explicit patch before installing extension_cpp')
    originals = {name: (repo/name).read_bytes() for name in (CPP, PYTHON, SETUP)}
    cpp = originals[CPP].decode('utf-8')
    # Only a known function declaration and one exact, unique binding are accepted.
    signature = (ROOT / 'compatibility/torch_metadata_signature.txt').read_text().strip()
    marker = 'at::Tensor quant_flash_attn_metadata('
    if cpp.count(marker) != 1 or cpp.count(BINDING) != 1:
        fail('UPSTREAM_INCOMPATIBLE', 'metadata definition/binding not unique')
    start = cpp.index(marker)
    end = cpp.index('\n{', start)
    if ' '.join(cpp[start:end].split()) != signature:
        fail('UPSTREAM_INCOMPATIBLE', 'metadata C++ signature changed')
    if cpp.count('#include "aclnn_common.h"') != 1:
        fail('UPSTREAM_INCOMPATIBLE', 'expected scoped C++ include anchor')
    cpp = cpp.replace('#include "aclnn_common.h"', '#include "aclnn_common.h"\n#include "qfa_mock/qfa_metadata_binding.h"', 1)
    cpp = cpp.replace(BINDING, 'm.def("quant_flash_attn_metadata", qfa_mock::MakeMetadataBinding(&quant_flash_attn_metadata), "quant_flash_attn_metadata");\n    qfa_mock::RegisterMetadataControls(m);', 1)
    outputs = {CPP: cpp.encode()}
    for name, transform in ((PYTHON, _edit_python), (SETUP, _edit_setup)):
        old = originals[name]
        value = transform(old.decode().replace('\r\n', '\n'))
        if b'\r\n' in old:
            value = value.replace('\n', '\r\n')
        outputs[name] = value.encode()
    files = {name: (ROOT/'cpp'/name).read_bytes() for name in ('metadata_core.h','metadata_core.cpp','metadata_adapter.h')}
    for name in ('qfa_metadata_binding.h','qfa_metadata_binding.cpp'):
        files[name] = (ROOT/'cpp/torch'/name).read_bytes()
    for name in COMPILE_SOURCES:
        files['upstream/'+name] = (repo/name).read_bytes()
    known = json.loads((ROOT/'compatibility/qfa_host_v2.json').read_text())
    files['LICENSE'] = (repo/'LICENSE').read_bytes()
    files['compatibility.json'] = (canonical(known)+'\n').encode()
    for name in known['files']:
        if name not in COMPILE_SOURCES:
            files['reference/'+name] = (repo/name).read_bytes()
    # The vendor provider alone does not supply cann_ops_transformer.ops.
    framework = {}
    framework_root = repo/'torch_extension/cann_ops_transformer'
    for path in sorted(framework_root.rglob('*')):
        if not path.is_file():
            continue
        relative = path.relative_to(framework_root)
        if any(part.startswith('.') or part == '__pycache__' for part in relative.parts) or path.suffix in ('.pyc','.pyo'):
            continue
        if relative.parts[0] == 'csrc' or (relative.parts[0] == 'ops' and relative.as_posix() != 'ops/__init__.py'):
            continue
        framework[relative.as_posix()] = sha(path.read_bytes())
    runtime = (ROOT/'qfa_mock/extension_backend.py').read_bytes()
    audit = (ROOT/'qfa_mock/attestation.py').read_bytes()
    identity = sha(canonical({'files': {k:sha(v) for k,v in files.items()}, 'runtime':sha(runtime), 'audit':sha(audit), 'framework':framework,
                              'bridge_inputs':{k:sha(v) for k,v in outputs.items()}}).encode())
    files['qfa_mock_build.h'] = ('#pragma once\n#define QFA_MOCK_BUILD_ID "'+identity+'"\n').encode()
    files['qfa_metadata_binding.cpp'] += ('\n// Payload identity forces JIT source invalidation: '+identity+'\n').encode()
    manifest = {'format':1,'mode':'extension_cpp','adapter_version':'qfa-host-v2','build_id':identity,
                'files':{k:sha(v) for k,v in files.items()}, 'framework_files':framework, 'runtime_sha256':sha(runtime), 'audit_sha256':sha(audit), 'entry_cpp_sha256':sha(outputs[CPP]),
                'entry_python_sha256':sha(outputs[PYTHON].decode().replace('\r\n','\n').encode()), 'assets_modified':False}
    files['manifest.json'] = (canonical(manifest)+'\n').encode()
    outputs.update({PAYLOAD+name: value for name,value in files.items()})
    outputs[LOCAL+'_qfa_metadata_backend.py'] = runtime
    outputs[LOCAL+'_qfa_metadata_audit.py'] = audit
    outputs['torch_extension/qfa_mock_wheel_support.py'] = (ROOT/'qfa_mock/wheel_support.py').read_bytes()
    for name in outputs.keys()-originals.keys():
        if (repo/name).exists():
            fail('INTEGRATION_CONFLICT', 'refusing to overwrite existing file: '+name)
        originals[name] = None
    return originals, outputs, manifest


def integrate(repo, check=False, reverse=False):
    repo = Path(repo).resolve()
    journal = repo/JOURNAL
    if journal.exists():
        saved = json.loads(journal.read_text())
        if set(saved['installed']) != set(saved['before']):
            fail('PACKAGE_INVALID', 'invalid integration journal')
        for name in saved['installed']:
            if Path(name).is_absolute() or '..' in Path(name).parts or not (repo/name).resolve().is_relative_to(repo):
                fail('PACKAGE_INVALID', 'unsafe integration journal path')
        for name, expected in saved['installed'].items():
            if not (repo/name).is_file() or sha((repo/name).read_bytes()) != expected:
                fail('SOURCE_CHANGED', 'integration file edited; preserve changes before reversal: '+name)
        if not reverse:
            return {'status':'already_applied','build_id':saved['build_id']}
        if not check:
            for name, before in saved['before'].items():
                path = repo/name
                if before is None:
                    path.unlink()
                else:
                    path.write_bytes(base64.b64decode(before))
            # Remove only empty directories owned by this integration.
            for path in sorted((repo/PAYLOAD).rglob('*'), key=lambda p:len(p.parts), reverse=True):
                if path.is_dir():
                    path.rmdir()
            (repo/PAYLOAD).rmdir()
            journal.unlink()
        return {'status':'can_remove' if check else 'removed'}
    if reverse:
        fail('INTEGRATION_MISSING', 'extension_cpp is not installed')
    before, outputs, manifest = changes(repo)
    if not check:
        written = []
        try:
            for name,value in outputs.items():
                path = repo/name; path.parent.mkdir(parents=True,exist_ok=True)
                path.write_bytes(value); written.append(name)
            record = {'format':1,'build_id':manifest['build_id'],'before':{k:None if v is None else base64.b64encode(v).decode() for k,v in before.items()},
                      'installed':{k:sha(v) for k,v in outputs.items()}}
            journal.write_text(canonical(record)+'\n')
        except BaseException:
            for name in reversed(written):
                if before[name] is None:
                    (repo/name).unlink()
                else:
                    (repo/name).write_bytes(before[name])
            raise
    return {'status':'can_apply' if check else 'applied','build_id':manifest['build_id'],
            'changed_files':[CPP,PYTHON,SETUP],'assets_modified':False}
