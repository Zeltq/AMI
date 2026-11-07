import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('install_chio_bot',ROOT/'scripts/install_chio_bot.py')
installer=importlib.util.module_from_spec(spec);spec.loader.exec_module(installer)


class Installation(unittest.TestCase):
    def test_prepare_records_explicit_model_without_installing(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);model=root/'new-model.pt';source=root/'original-weights'
            prepared=installer.prepare(root/'prepared',3,model,source)
            policy=json.loads((prepared/'policy-config.json').read_text())
            self.assertEqual(Path(policy['model']),model)
            self.assertEqual(Path(policy['source']),source)
            self.assertEqual(policy['model_team'],2)
            self.assertEqual(policy['executor_version'],'p17-v2')

    def test_prepare_install_restore_preserves_user_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);target=root/'user-bots';target.mkdir();(target/'mine.lua').write_bytes(b'-- user\r\n')
            before=installer.inventory(target);prepared=installer.prepare(root/'prepared',2)
            self.assertEqual(installer.inventory(target),before)
            from lupa.lua51 import LuaRuntime
            self.assertEqual(LuaRuntime().execute((prepared/'settings.lua').read_text()).human_team,2)
            self.assertIn('chio_executor',(prepared/'bot_generic.lua').read_text())
            policy=json.loads((prepared/'policy-config.json').read_text())
            self.assertEqual(policy['executor_version'],'p17-v2')
            self.assertEqual(Path(policy['source']),Path('C:/AMI/runs/p16/sampling-dagger-2-il/weights'))
            self.assertEqual(Path(policy['model']),Path('C:/AMI/runs/p17/models/sampling-dagger-2.pt'))
            self.assertEqual((policy['device'],policy['action_selection'],policy['model_team']),('cpu','sampling',3))
            artifacts=root/'artifacts';saved=installer.install(target,prepared,artifacts,2)
            self.assertTrue(saved['installed']);self.assertEqual(installer.inventory(Path(saved['backup'])),before)
            result=installer.restore(artifacts)
            self.assertTrue(result['restore_verified']);self.assertEqual(installer.inventory(target),before)
