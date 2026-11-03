import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from sf_duel.live.install import isolated_content


class IsolatedContentTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        base = Path(self.temporary.name)
        self.game = base / 'steam'
        self.worker = base / 'worker'
        self.bots = base / 'bots'
        self.bots.mkdir()
        (self.bots / 'agent.lua').write_text('-- test')
        (self.game / 'game/bin/win64').mkdir(parents=True)
        content = self.game / 'game/dota'
        (content / 'cfg').mkdir(parents=True)
        self.vscripts = content / 'scripts/vscripts'
        (self.vscripts / 'shared').mkdir(parents=True)
        (self.vscripts / 'bots').mkdir()
        (self.vscripts / 'bots.before-ami-teacher-v9-test').mkdir()
        (content / 'gameinfo.gi').write_text('test')

    @unittest.skipUnless(sys.platform == 'win32', 'Windows PowerShell junction regression')
    def test_single_pair_stays_a_pair_and_bot_backups_are_skipped(self):
        content, runtime = isolated_content(self.worker, self.bots, self.game)
        link = content / 'scripts/vscripts/shared'
        self.addCleanup(link.rmdir)
        self.assertTrue(link.is_junction())
        self.assertEqual(link.resolve(), self.vscripts / 'shared')
        self.assertFalse((content / 'scripts/vscripts/bots.before-ami-teacher-v9-test').exists())
        self.assertFalse((content / 'scripts/vscripts/bots').exists())
        self.assertEqual((runtime / 'agent.lua').read_text(), '-- test')

    def test_child_stderr_is_visible_in_exception(self):
        error = subprocess.CalledProcessError(1, ['powershell'], stderr='specific junction failure')
        with patch('sf_duel.live.install.subprocess.run', side_effect=error):
            with self.assertRaisesRegex(RuntimeError, 'specific junction failure'):
                isolated_content(self.worker, self.bots, self.game)


if __name__ == '__main__':
    unittest.main()
