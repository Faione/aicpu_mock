"""Independent profiling of unmodified golden; never changes its fallback behavior."""
import array
import hashlib
import sys
import threading


class GoldenAudit:
    def __init__(self, module):
        self.codes = {}
        for name in ('get_qfa_section_info', '_decode_qfa_sections', '_cpu_mxfp8_golden_sectioned'):
            self.codes[getattr(module, name).__code__] = name
        self.records = []
        self._thread = threading.get_ident()

    def _profile(self, frame, event, value):
        if threading.get_ident() != self._thread or frame.f_code not in self.codes:
            return
        name = self.codes[frame.f_code]
        if event == 'call' and name == '_decode_qfa_sections':
            words = array.array('I', (int(v) & 0xffffffff for v in frame.f_locals['meta_flat']))
            if sys.byteorder != 'little':
                words.byteswap()
            self.records.append({'event':'decoder_input','metadata_digest':hashlib.sha256(words.tobytes()).hexdigest()})
        if event == 'return':
            valid = isinstance(value, tuple) and len(value) == (2 if name == '_cpu_mxfp8_golden_sectioned' else 5)
            self.records.append({'event':name,'success':valid})

    def __enter__(self):
        if sys.getprofile() is not None:
            raise RuntimeError('golden audit requires an independent worker without another profiler')
        sys.setprofile(self._profile)
        return self

    def __exit__(self, *error):
        sys.setprofile(None)

    def require_consumed(self, metadata_digest=None):
        for name in ('get_qfa_section_info', '_cpu_mxfp8_golden_sectioned'):
            matching = [r for r in self.records if r['event'] == name]
            if not matching or not all(r['success'] for r in matching):
                raise ValueError('golden did not successfully consume sectioned metadata: '+name)
        if metadata_digest is not None and not any(r.get('metadata_digest') == metadata_digest for r in self.records):
            raise ValueError('golden decoder did not consume the expected metadata identity; inspect cache/worker state')
        return list(self.records)
