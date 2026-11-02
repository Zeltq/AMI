import sys
from pathlib import Path
import tempfile
import json
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import run_mia_autonomous as mia


class MiaTests(unittest.TestCase):
    def test_validation_versions(self):
        self.assertEqual(mia.opponents_for(1,False,['zero']),[('teacher',None),('MIA0','zero')])
        self.assertEqual(mia.opponents_for(2,True,['zero','one']),[('MIA1','one'),('MIA0','zero')])
        self.assertEqual(mia.opponents_for(3,True,['zero','one','two']),[('MIA2','two'),('MIA1','one')])

    def test_deadline_grace(self):
        self.assertTrue(mia.may_continue(99,100,10000))
        self.assertTrue(mia.may_continue(101,100,1799))
        self.assertFalse(mia.may_continue(101,100,1800))

    def test_attempts_preserve_existing_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'learner').mkdir();(root/'learner'/'checkpoint.pt').write_bytes(b'preserve')
            self.assertEqual(mia.available_dir(root,'learner').name,'learner-attempt-2')
            self.assertEqual((root/'learner'/'checkpoint.pt').read_bytes(),b'preserve')

    def test_recover_finished_games_without_restarting_them(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            for index,error in [(1,''),(2,'transport failed')]:
                game=root/f'match-{index:04d}-attempt-01';game.mkdir()
                (game/'duel-result.json').write_text(json.dumps({'error':error,'reason':'death','outcome':{'winner':2}}))
                (root/f'route-{index:04d}.json').write_text(json.dumps({'2':'current','3':'0'}))
            recovered=mia.recover_completed_matches(root)
            self.assertEqual(len(recovered),1);self.assertEqual(recovered[0]['index'],1)
            self.assertEqual(recovered[0]['routing']['3'],'0')
