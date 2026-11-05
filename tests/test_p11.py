import json,os,subprocess,sys,tempfile,unittest
from unittest.mock import patch
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'gpu/scripts'))
from p11_run import training_budget,wait_existing
from p11_report import episode_summary

class P11Tests(unittest.TestCase):
    def test_schedule_reserves_every_periodic_evaluation_and_margin(self):
        for remaining in (100,200,300,375,400):
            t=training_budget(remaining)
            self.assertLessEqual(t+36*(t//120)+5,remaining)
            self.assertGreater(t+1+36*((t+1)//120)+5,remaining)
        self.assertEqual(training_budget(375),298)
        self.assertEqual(training_budget(375,10),288)

    @unittest.skipUnless(os.name=='nt','Windows process adoption')
    def test_adoption_preserves_existing_child_and_exit_code(self):
        child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(.3);raise SystemExit(7)'],creationflags=subprocess.CREATE_NO_WINDOW)
        seen=[]
        self.assertEqual(wait_existing(child.pid,lambda:seen.append(child.pid)),7)
        self.assertEqual(child.wait(),7)
        self.assertEqual(set(seen),{child.pid})

    def test_training_statistics_use_learner_side_and_pooled_denominators(self):
        def episode(side,result,farm,spawn,death):
            return dict(reward=2,components=[0,1,1,0,0,0],game=dict(result=result,lane_creeps_spawned=spawn,sides=[dict(controller='opponent',side='dire' if side=='radiant' else 'radiant',lh=999,deny=999,first_death_game_seconds=0),dict(controller='GGD',side=side,lh=farm,deny=0,first_death_game_seconds=death)]))
        r=episode_summary([episode('radiant',1,2,10,None),episode('dire',-1,4,30,-5),episode('dire',1,6,60,None)])
        self.assertEqual((r['wins'],r['draws'],r['losses'],r['matches']),(2,0,1,3))
        self.assertEqual((r['farm'],r['spawn'],r['death_matches']),(12,100,1))
        self.assertEqual(r['reward_sum'],6)

    def test_audit_ignores_training_log_beside_run_directory(self):
        import p11_audit
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);run=root/'train-001';run.mkdir()
            (root/'protected.json').write_text('{}',encoding='utf-8')
            (root/'train-001.log').write_text('native log',encoding='utf-8')
            (run/'config.json').write_text(json.dumps(dict(schedule_minutes=100)),encoding='utf-8')
            row=dict(training_seconds=10,collection_seconds=2,optimization_seconds=3,learning=dict(clip_fraction=.01,kl_coefficient=.03-.02*5/6000))
            (run/'history.jsonl').write_text(json.dumps(row)+'\n',encoding='utf-8')
            with patch.object(p11_audit,'RUN',root),patch.object(p11_audit,'REPORT',root):p11_audit.finish()
            self.assertEqual(json.loads((root/'audit.json').read_text())['training'],[dict(run='train-001',blocks=1,clip_and_anchor_checks=1)])
if __name__=='__main__':unittest.main()
