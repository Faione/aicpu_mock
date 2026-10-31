import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from qfa_mock.build import ROOT
from qfa_mock.model import load_json, normalize_config, normalize_request, wire_request


class CoreTests(unittest.TestCase):
    def test_explicit_capacity_uses_real_schedule_not_allocation_upper_bound(self):
        config=normalize_config(load_json(ROOT/'examples/target_arch92.json'))
        request=normalize_request(load_json(ROOT/'examples/request.json'))
        wire=wire_request(config,request)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'metadata.bin'
            command=[str(ROOT/'build/metadata_helper'),str(path)]
            baseline=subprocess.run(command,input=wire,text=True,capture_output=True,check=True)
            used=json.loads(baseline.stdout)['used_words']
            prefix=path.read_bytes()[:used*4]
            for capacity in (used,used+1,200000):
                result=subprocess.run([*command,'--row-size',str(capacity)],input=wire,text=True,capture_output=True)
                self.assertEqual(result.returncode,0,result.stderr)
                self.assertEqual(json.loads(result.stdout)['shape'],[2,capacity])
                data=path.read_bytes()
                self.assertEqual(data[:used*4],prefix)
                self.assertFalse(any(data[used*4:]))
            old=path.read_bytes()
            bad=subprocess.run([*command,'--row-size',str(used-1)],input=wire,text=True,capture_output=True)
            self.assertNotEqual(bad.returncode,0)
            self.assertEqual(path.read_bytes(),old)

    def test_cpp_validator_rejects_corrupt_header_fa_and_padding(self):
        source='''#include "metadata_core.h"
#include <iostream>
int main() {
    qfa_mock::Target t;
    qfa_mock::Request r;
    r.nq=64; r.nk=8; r.maxQ=128; r.maxKv=1024;
    r.cuQ={0,128}; r.usedKv={1024};
    auto good=qfa_mock::Compute(t,r);
    unsigned rejected=0;
    for (auto index : {0U,1U,4U,6U,15U,16U,unsigned(good.words.size()-1)}) {
        auto bad=good; bad.words[index]++;
        try { qfa_mock::Validate(t,r,bad); }
        catch(const std::exception&) { ++rejected; }
    }
    return rejected==7 ? 0 : 1;
}'''
        repo=os.environ.get('QFA_REPO',str(ROOT.parent/'ops-transformer'))
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory);(path/'check.cpp').write_text(source)
            compiled=subprocess.run(['c++','-O1','-std=c++17','-I',str(ROOT/'cpp'),'-I',repo,str(path/'check.cpp'),str(ROOT/'cpp/metadata_core.cpp'),'-o',str(path/'check')],capture_output=True,text=True)
            self.assertEqual(compiled.returncode,0,compiled.stderr)
            self.assertEqual(subprocess.run([str(path/'check')]).returncode,0)
