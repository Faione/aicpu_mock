import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from qfa_mock.build import ROOT, COMPILE_SOURCES, ASSET_SOURCES, AICPU_SOURCES, verify_build
from qfa_mock.extension import CPP, PYTHON, SETUP, PAYLOAD, LOCAL, integrate, sha
from qfa_mock.model import MockError
from qfa_mock.package import inspect_wheel, build_run

REPO=Path(os.environ.get('QFA_REPO',str(ROOT.parent/'ops-transformer')))


def source_copy(destination):
    shutil.copyfile(REPO/'LICENSE',destination/'LICENSE')
    names=(*COMPILE_SOURCES,*ASSET_SOURCES,*AICPU_SOURCES,CPP,PYTHON,SETUP)
    for name in names:
        path=destination/name;path.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(REPO/name,path)
    te=REPO/'attention/quant_flash_attn/torch_extension'
    for p in te.glob('*.py'):
        shutil.copyfile(p,destination/LOCAL/p.name)
    shutil.copytree(REPO/'torch_extension/cann_ops_transformer',destination/'torch_extension/cann_ops_transformer',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))


class ExtensionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='qfa-extension-')
        self.repo=Path(self.temp.name)/'source with spaces'
        self.repo.mkdir()
        source_copy(self.repo)

    def tearDown(self):
        self.temp.cleanup()

    def test_scoped_binding_and_byte_exact_reversal_assets_unchanged(self):
        before={p.relative_to(self.repo):p.read_bytes() for p in self.repo.rglob('*') if p.is_file()}
        info=integrate(self.repo,check=True)
        self.assertEqual(info['status'],'can_apply')
        self.assertFalse((self.repo/PAYLOAD).exists())
        integrate(self.repo)
        self.assertEqual(integrate(self.repo)['status'],'already_applied')
        cpp=(self.repo/CPP).read_text()
        original=before[Path(CPP)].decode()
        start=original.index('at::Tensor quant_flash_attn_metadata(')
        end=original.index('// Bind the C++ function')
        self.assertIn(original[start:end],cpp)  # Both original C++ bodies remain byte-for-byte unchanged.
        for name in ASSET_SOURCES:
            self.assertEqual((self.repo/name).read_bytes(),before[Path(name)])
        manifest=json.loads((self.repo/PAYLOAD/'manifest.json').read_text())
        for name,digest in manifest['files'].items():
            self.assertEqual(sha((self.repo/PAYLOAD/name).read_bytes()),digest)
        integrate(self.repo,reverse=True)
        after={p.relative_to(self.repo):p.read_bytes() for p in self.repo.rglob('*') if p.is_file()}
        self.assertEqual(before,after)

    def test_mutual_exclusion_and_source_edits_not_lost(self):
        golden=self.repo/'attention/quant_flash_attn/tests/assets/impl/golden.py'
        golden.write_text(golden.read_text()+'\n# QFA_MOCK_ASSETS_V1\n')
        with self.assertRaisesRegex(MockError,'remove the assets_explicit'):
            integrate(self.repo)
        golden.write_bytes((REPO/'attention/quant_flash_attn/tests/assets/impl/golden.py').read_bytes())
        integrate(self.repo)
        subprocess_result=subprocess.run([sys.executable,str(ROOT/'scripts/apply_assets_patch.py'),'--repo',str(self.repo)],capture_output=True,text=True)
        self.assertNotEqual(subprocess_result.returncode,0)
        source=self.repo/CPP;source.write_text(source.read_text()+'\n// user change\n')
        with self.assertRaisesRegex(MockError,'integration file edited'):
            integrate(self.repo,reverse=True)
        self.assertIn('user change',source.read_text())

    def test_unknown_metadata_behavior_and_signature_rejected(self):
        aicpu=self.repo/AICPU_SOURCES[0]
        aicpu.write_text(aicpu.read_text()+'\n// changed\n')
        with self.assertRaisesRegex(MockError,'changed behavior files'):
            integrate(self.repo)
        with self.assertRaisesRegex(MockError,'changed behavior files'):
            verify_build(ROOT/'build/metadata_helper',self.repo)
        aicpu.write_bytes((REPO/AICPU_SOURCES[0]).read_bytes())
        source=self.repo/CPP;source.write_text(source.read_text().replace('int64_t head_dim_v, int64_t quant_mode','int64_t quant_mode, int64_t head_dim_v',1))
        with self.assertRaisesRegex(MockError,'signature changed'):
            integrate(self.repo)

    def test_runtime_fixed_target_and_payload_integrity(self):
        integrate(self.repo)
        spec=importlib.util.spec_from_file_location('isolated_backend',self.repo/LOCAL/'_qfa_metadata_backend.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with patch.dict(os.environ,{'QFA_METADATA_CONFIG':str(ROOT/'examples/target_arch92.json'),'QFA_METADATA_BACKEND':'mock_aicpu'}):
            config=module._freeze();self.assertEqual(config['target']['aic_core_num'],80)
            module.verify_payload(self.repo/PAYLOAD)
            with patch.dict(os.environ,QFA_METADATA_CONFIG=str(ROOT/'examples/target_arch35.json')):
                with self.assertRaisesRegex(ValueError,'worker backend/target'):
                    module._freeze()
        core=self.repo/PAYLOAD/'metadata_core.h';core.write_text(core.read_text()+'\n//changed\n')
        with self.assertRaisesRegex(ValueError,'extension payload changed'):
            module.verify_payload(self.repo/PAYLOAD)

    def test_real_vendor_wheel_and_run_bundle_integrity(self):
        integrate(self.repo)
        dist=Path(self.temp.name)/'dist'
        command=[sys.executable,str(ROOT/'scripts/build_extension_wheel.py'),'--repo',str(self.repo),'--vendor','qfamock','--output',str(dist)]
        result=subprocess.run(command,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        wheel=next(dist.glob('*.whl'))
        info=inspect_wheel(wheel)
        self.assertEqual(info['package'],'cann_ops_transformer_qfamock')
        self.assertTrue(info['jit_required'])
        # Test packaging mechanics only. This trap fixture is never a release/SDK payload.
        opp=Path(self.temp.name)/'fixture.run';opp.write_text('#!/bin/sh\nexit 97\n')
        declaration={'format':1,'package_type':'cann_custom_run','sha256':sha(opp.read_bytes()),
                     'vendor':'qfamock','host_arch':'aarch64','kernel_arch':'arch92',
                     'operators':['quant_flash_attn','quant_flash_attn_metadata'], 'behavior_digest':info['behavior_digest'],'installed_files':{'op_api/lib/fixture.so':'0'*64,'op_impl/fixture.bin':'0'*64}}
        metadata=Path(self.temp.name)/'opp.json';metadata.write_text(json.dumps(declaration))
        output=Path(self.temp.name)/'bundle.run'
        build_run(opp,metadata,wheel,output)
        env={**os.environ,'QFA_PACKAGE_PYTHON':sys.executable}
        check=subprocess.run([str(output),'--inspect'],env=env,capture_output=True,text=True)
        self.assertEqual(check.returncode,0,check.stderr)
        release=json.loads(check.stdout)
        self.assertEqual(release['wheel_info']['build_id'],info['build_id'])
        # Integrity-only inspection did not execute the trap payload.
        self.assertEqual(release['opp']['vendor'],'qfamock')
        declaration['operators']=['quant_flash_attn'];metadata.write_text(json.dumps(declaration))
        with self.assertRaisesRegex(MockError,'native metadata dependencies'):
            build_run(opp,metadata,wheel,Path(self.temp.name)/'invalid.run')
