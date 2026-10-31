"""Compile the actual bridge against CPU libtorch; no CANN/NPU claims."""
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import torch
    from torch.utils.cpp_extension import load
    from qfa_mock.build import ROOT, load_json
    from qfa_mock.extension import sha
    from qfa_mock.model import normalize_config, normalize_request, wire_request
    import subprocess
    repo = Path(os.environ.get('QFA_REPO', str(ROOT.parent/'ops-transformer')))
    directory = ROOT/'build/cpu_bridge'
    directory.mkdir(parents=True, exist_ok=True)
    signature = (ROOT/'compatibility/torch_metadata_signature.txt').read_text().strip()
    signature = signature.replace('quant_flash_attn_metadata(', 'Native(')
    (directory/'qfa_mock_build.h').write_text('#define QFA_MOCK_BUILD_ID "cpu-acceptance"\n')
    source = '#include "qfa_metadata_binding.h"\n#include <cstdlib>\nstatic int nativeCalls = 0;\n'+signature+' { ++nativeCalls; if (std::getenv("QFA_MOCK_TEST_NATIVE_ERROR")) throw std::runtime_error("native-sentinel"); return output; }\n'
    source += '''PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("quant_flash_attn_metadata", qfa_mock::MakeMetadataBinding(&Native));
    m.def("native_count", [](){ return nativeCalls; });
    qfa_mock::RegisterMetadataControls(m);
}
'''
    (directory/'acceptance.cpp').write_text(source)
    os.environ.setdefault('MAX_JOBS', '1')
    module = load(name='qfa_mock_cpu_bridge', sources=[str(directory/'acceptance.cpp'),
        str(ROOT/'cpp/torch/qfa_metadata_binding.cpp'), str(ROOT/'cpp/metadata_core.cpp')],
        extra_include_paths=[str(ROOT/'cpp/torch'),str(ROOT/'cpp'),str(directory),str(repo)],
        extra_cflags=['-O0','-std=c++17','-DQFA_MOCK_CPU_ACCEPTANCE=1'],
        build_directory=str(directory), verbose=False)
    # Native pointer path first; use a separate process for mock to honor mode immutability.
    if '--mock-child' not in sys.argv:
        env=dict(os.environ);env.pop('QFA_METADATA_CONFIG',None);env['QFA_METADATA_BACKEND']='native'
        child_reports=[]
        for extra in ([], ['--arch35']):
            child=subprocess.run([sys.executable, str(Path(__file__)), '--mock-child', *extra], env={**env,'QFA_METADATA_BACKEND':'mock_aicpu','QFA_METADATA_CONFIG':'cpu-acceptance'},capture_output=True,text=True)
            if child.returncode:
                raise RuntimeError(child.stdout+child.stderr)
            child_reports.append(json.loads(child.stdout.strip().splitlines()[-1]))
        old=dict(os.environ)
        os.environ.pop('QFA_METADATA_CONFIG',None);os.environ['QFA_METADATA_BACKEND']='native'
        output=torch.zeros((2,32),dtype=torch.int32)
        args=(None,None,None,None,1,1,1,1,1,128,128,1,3,-1,-1,'TND','TND','PA_BNBD','TND',False,output)
        assert module.quant_flash_attn_metadata(*args) is output
        assert module.native_count()==1
        os.environ['QFA_MOCK_TEST_NATIVE_ERROR']='1'
        try: module.quant_flash_attn_metadata(*args)
        except RuntimeError as error: assert 'native-sentinel' in str(error)
        else: raise AssertionError('native failure swallowed')
        os.environ.pop('QFA_MOCK_TEST_NATIVE_ERROR')
        os.environ['QFA_METADATA_BACKEND']='mock_aicpu'
        try: module.quant_flash_attn_metadata(*args)
        except RuntimeError: pass
        else: raise AssertionError('mode change accepted')
        report=child_reports[0]
        report['records'] += child_reports[1]['records']
        report.update(date='2026-10-31', native_pointer=True, native_calls=2, native_error_propagated=True, mode_change_rejected=True)
        report['source_sha256']={str(p.relative_to(ROOT)):sha(p.read_bytes()) for p in [ROOT/'cpp/metadata_core.cpp',ROOT/'cpp/metadata_core.h',ROOT/'cpp/metadata_adapter.h',ROOT/'cpp/torch/qfa_metadata_binding.cpp',ROOT/'cpp/torch/qfa_metadata_binding.h']}
        (ROOT/'docs/cpp_bridge_verification.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2))
        os.environ.clear();os.environ.update(old)
        return
    arch35='--arch35' in sys.argv
    module._qfa_mock_configure(True,not arch35,36 if arch35 else 80,72 if arch35 else 80,'cpu-acceptance')
    cases=[('request.json','target_arch92.json'),('request_fd.json','target_arch35.json'),('request_multisection.json','target_arch35.json')]
    records=[]
    lib=torch.library.Library('qfa_mock_acceptance','DEF')
    lib.define('metadata(Tensor? cuq, Tensor? cukv, Tensor? usedq, Tensor? usedkv, int batch, int maxq, int maxkv, int nq, int nk, int dim, int dimv, int quant, int mask, int left, int right, str lq, str ls, str lkv, str lo, bool grad, Tensor output) -> Tensor')
    lib.impl('metadata', module.quant_flash_attn_metadata, 'CPU')
    for request_name, config_name in (cases[1:] if arch35 else cases[:1]):
        r=normalize_request(load_json(ROOT/'examples'/request_name));c=normalize_config(load_json(ROOT/'examples'/config_name))
        def tensor(v): return None if v is None else torch.tensor(v,dtype=torch.int32)
        row=module._qfa_mock_row_size(r['batch_size'],r['num_heads_q'])
        output=torch.full((2,row+4096),-99,dtype=torch.int32)
        args=[tensor(r[k]) for k in ('cu_seqlens_q','cu_seqlens_kv','seqused_q','seqused_kv')]
        args += [r['batch_size'],r['max_seqlen_q'],r['max_seqlen_kv'],r['num_heads_q'],r['num_heads_kv'],128,128,1,r['mask_mode'],-1,-1,'TND','TND',r['layout_kv'],'TND',False,output]
        actual=module.quant_flash_attn_metadata(*args)
        assert actual is output and module.native_count()==0
        assert torch.ops.qfa_mock_acceptance.metadata(*args) is output
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'metadata.bin'
            subprocess.run([str(ROOT/'build/metadata_helper'),str(path),'--row-size',str(row+4096)],input=wire_request(c,r),text=True,check=True,capture_output=True)
            assert output.numpy().tobytes()==path.read_bytes()
        untouched=output.clone()
        for index,value in ((20,torch.zeros((2,16),dtype=torch.int32)),(20,torch.zeros((2,row),dtype=torch.float32)),(11,6)):
            bad=list(args);bad[index]=value
            try: module.quant_flash_attn_metadata(*bad)
            except RuntimeError: pass
            else: raise AssertionError('invalid bridge request accepted')
        assert torch.equal(untouched,output)
        records.append({'request':request_name,'row':row+4096,'sha256':sha(output.numpy().tobytes())})
    assert module._qfa_mock_info()['cpu_acceptance_only'] is True
    print(json.dumps({'scope':'actual C++ bridge/core, CPU libtorch; CPU device boundary compile flag',
                      'records':records,'cpu_dispatcher_route':True,'same_output':True,'larger_capacity':True,'invalid_output_rejected':True,
                      'native_not_called_in_mock':True,'real_cann_extension':'not_run','privateuse1':'not_run','h2d':'not_run'}))


if __name__=='__main__':
    main()
