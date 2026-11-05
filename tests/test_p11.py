import os,subprocess,sys,unittest
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
if __name__=='__main__':unittest.main()
