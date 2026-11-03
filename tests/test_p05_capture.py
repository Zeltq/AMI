import importlib.util
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('p05_measure',ROOT/'reports/P03/measure_dota.py')
measure=importlib.util.module_from_spec(spec);spec.loader.exec_module(measure)


class CaptureTests(unittest.TestCase):
    def test_reset_is_only_allowed_after_completed_capture_window(self):
        class Connection:
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def settimeout(self,*args):pass
            def recv(self,*args):raise ConnectionResetError('test connection reset')
        for expected in (False,True):
            with self.subTest(expected=expected),tempfile.TemporaryDirectory() as folder:
                capture=measure.WorldCapture.__new__(measure.WorldCapture)
                capture.stop=threading.Event();capture.errors=[];capture.expect_disconnect=expected
                with patch.object(measure.socket,'create_connection',return_value=Connection()):
                    capture.receive(Path(folder),2,12345)
                self.assertEqual(capture.errors,[] if expected else ['test connection reset'])


if __name__=='__main__':unittest.main()
