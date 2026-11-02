import hashlib
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

SCRIPTS=Path(__file__).resolve().parents[1]/'gpu/scripts'
sys.path.insert(0,str(SCRIPTS))
from export_policy import verified_manifest


class WeightExportTests(unittest.TestCase):
    def fixture(self,root,checksum=''):
        raw=struct.pack('<2f',1.,-2.)
        (root/'000.f32').write_bytes(raw)
        manifest={'tensors':{'linear.weight':{'shape':[1,2],'file':'000.f32','sha256':checksum}}}
        (root/'manifest.json').write_text(json.dumps(manifest))
        return raw

    def test_legacy_export_gets_checksums_without_changing_weights(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);raw=self.fixture(root)
            manifest=verified_manifest(root)
            self.assertEqual(manifest['tensors']['linear.weight']['sha256'],hashlib.sha256(raw).hexdigest())
            self.assertEqual(json.loads((root/'manifest.json').read_text()),manifest)
            self.assertEqual((root/'000.f32').read_bytes(),raw)
            before=(root/'manifest.json').read_bytes();verified_manifest(root)
            self.assertEqual((root/'manifest.json').read_bytes(),before)

    def test_existing_checksum_rejects_same_size_corruption(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);self.fixture(root);verified_manifest(root)
            before=(root/'manifest.json').read_bytes()
            (root/'000.f32').write_bytes(struct.pack('<2f',1.,-3.))
            with self.assertRaisesRegex(ValueError,'checksum mismatch'):verified_manifest(root)
            self.assertEqual((root/'manifest.json').read_bytes(),before)

    def test_legacy_export_rejects_truncation(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);self.fixture(root)
            before=(root/'manifest.json').read_bytes();(root/'000.f32').write_bytes(b'1234')
            with self.assertRaisesRegex(ValueError,'byte count'):verified_manifest(root)
            self.assertEqual((root/'manifest.json').read_bytes(),before)


if __name__=='__main__':unittest.main()
