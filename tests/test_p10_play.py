import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from install_play_bot import install,restore,inventory
from play_vs_model import summarize


class PlayTests(unittest.TestCase):
    def test_restore_recovers_nested_bytes_empty_directories_and_removes_new_files(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);target=root/'bots';target.mkdir()
            (target/'nested').mkdir();(target/'empty').mkdir()
            (target/'nested/original').write_bytes(bytes(range(256)))
            before=inventory(target)
            result=install(target,root/'artifacts',3)
            self.assertEqual(result['model_team'],2)
            (target/'new').write_text('command buffer')
            restore(root/'artifacts')
            self.assertEqual(before,inventory(target))
            self.assertTrue((target/'empty').is_dir())

    def test_first_death_censors_future_casts_and_unreached_marks(self):
        rows=[dict(kind='hero',team=t,time=300,lh=t,deny=1,alive=True,x=0,y=0) for t in (2,3)]
        rows += [dict(kind='cast',team=3,time=301,ability='nevermore_shadowraze1'),
                 dict(kind='hurt',team=3,time=301,target='npc_dota_hero_nevermore',ability='nevermore_shadowraze1'),
                 dict(kind='death',team=2,killer_team=3,time=302),
                 dict(kind='cast',team=3,time=400,ability='nevermore_shadowraze2')]
        with tempfile.TemporaryDirectory() as folder:
            log=Path(folder)/'log';log.write_text(''.join('SF_PLAY_EVENT|'+json.dumps(r)+'\n' for r in rows))
            s=summarize(log,3)
        self.assertEqual(s['first_blood']['team'],2)
        self.assertEqual(s['sides']['3']['raze_casts'],1)
        self.assertEqual(s['sides']['3']['raze_hero_hits'],1)
        self.assertIsNone(s['sides']['3']['at_seconds']['600'])
        self.assertEqual(s['sides']['3']['at_seconds']['300']['lh'],3)


if __name__=='__main__':unittest.main()
