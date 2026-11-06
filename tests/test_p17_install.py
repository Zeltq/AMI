import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('install_chio_bot',ROOT/'scripts/install_chio_bot.py')
installer=importlib.util.module_from_spec(spec);spec.loader.exec_module(installer)


class Installation(unittest.TestCase):
    def test_prepare_install_restore_preserves_user_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);target=root/'user-bots';target.mkdir();(target/'mine.lua').write_bytes(b'-- user\r\n')
            before=installer.inventory(target);prepared=installer.prepare(root/'prepared',2)
            self.assertEqual(installer.inventory(target),before)
            from lupa.lua51 import LuaRuntime
            self.assertEqual(LuaRuntime().execute((prepared/'settings.lua').read_text()).human_team,2)
            self.assertIn('chio_executor',(prepared/'bot_generic.lua').read_text())
            artifacts=root/'artifacts';saved=installer.install(target,prepared,artifacts,2)
            self.assertTrue(saved['installed']);self.assertEqual(installer.inventory(Path(saved['backup'])),before)
            result=installer.restore(artifacts)
            self.assertTrue(result['restore_verified']);self.assertEqual(installer.inventory(target),before)
