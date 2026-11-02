import importlib.util
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
spec=importlib.util.spec_from_file_location('native_runner_recording',ROOT/'scripts/run_native_teacher.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

class RecordingRequestTests(unittest.TestCase):
 def test_only_selected_matches_record_and_attempts_have_unique_names(self):
  cfg={'recordings':{'match-0005':'pilot'}}
  self.assertIsNone(module.requested_recording(Path('match-0004-attempt-01'),cfg))
  self.assertEqual(module.requested_recording(Path('match-0005-attempt-01'),cfg),'pilot_match-0005-attempt-01')
  self.assertEqual(module.requested_recording(Path('match-0005-attempt-02'),cfg),'pilot_match-0005-attempt-02')
 def test_explicit_existing_cli_recording_takes_priority(self):
  self.assertEqual(module.requested_recording(Path('match-0005-attempt-01'),{'recordings':{'match-0005':'pilot'}},'existing'),'existing')
 def test_runtime_name_cannot_inject_console_commands_or_paths(self):
  for name in ('../escape','two words','x;quit','x\nquit','',23):
   with self.subTest(name=name),self.assertRaises(ValueError):module.requested_recording(Path('match-0005-attempt-01'),{'recordings':{'match-0005':name}})
