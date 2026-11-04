import hashlib,json,sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'gpu/scripts'))
from train_ggd import refresh_export_source

class ExportSourceTests(unittest.TestCase):
    def test_hash_tracks_checkpoint_after_history_update(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);checkpoint=root/'epoch.pt';checkpoint.write_bytes(b'before evaluation')
            weights=root/'weights';weights.mkdir();manifest=weights/'manifest.json'
            manifest.write_text(json.dumps({'tensors':{'a':{'sha256':'unchanged'}},'source_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest()}))
            checkpoint.write_bytes(b'after evaluation history added')
            refresh_export_source(checkpoint,root)
            result=json.loads(manifest.read_text())
            self.assertEqual(result['source_sha256'],hashlib.sha256(checkpoint.read_bytes()).hexdigest())
            self.assertEqual(result['tensors'],{'a':{'sha256':'unchanged'}})

if __name__=='__main__':unittest.main()
