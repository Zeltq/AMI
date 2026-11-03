import importlib.util
import json
from pathlib import Path
import tempfile
import sys
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('p07_measure',Path(__file__).resolve().parents[1]/'reports/P03/measure_dota.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)

class ProbeIOTests(unittest.TestCase):
    def test_busy_shared_index_does_not_terminate_measurement(self):
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'gpu/scripts'))
        import p07_status
        with patch.object(Path,'read_text',side_effect=PermissionError('busy')),patch.object(p07_status.time,'sleep'):
            self.assertFalse(p07_status.publish('test',{'stage':'running'}))
    def test_transient_windows_reader_does_not_abort_job(self):
        original=Path.replace
        attempts=[]
        def replace(source,target):
            attempts.append(target)
            if len(attempts)==1:raise PermissionError('destination held by reader')
            return original(source,target)
        with tempfile.TemporaryDirectory() as root,patch.object(Path,'replace',replace),patch.object(runner.time,'sleep'):
            path=Path(root)/'status.json'
            runner.save(path,{'stage':'running'})
            self.assertEqual(json.loads(path.read_text()),{'stage':'running'})
            self.assertEqual(len(attempts),2)
    def test_persistent_write_failure_is_reported(self):
        with tempfile.TemporaryDirectory() as root,patch.object(Path,'replace',side_effect=PermissionError('locked')),patch.object(runner.time,'sleep'):
            with self.assertRaises(PermissionError):runner.save(Path(root)/'status.json',{})
