import json,sys,tempfile,unittest
from pathlib import Path
import numpy as np
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'gpu/scripts'))
from p09b_reward import potentials,rewards,targets,GAMMA

FIELDS=['own_lh','enemy_lh','own_deny','enemy_deny','own_hp_fraction','enemy_hp_fraction','own_mana_fraction','enemy_mana_fraction','terminal','result','game_seconds']
class RewardTests(unittest.TestCase):
    def test_discounted_shaping_telescopes_with_nonzero_initial_potential(self):
        states=np.zeros((4,len(FIELDS)));states[:,0]=[2,3,4,4];states[:,4]=[1,.8,.6,0];states[:,5]=1;states[-1,8]=1;states[-1,9]=-1
        transitions=np.stack([states[:-1],states[1:]],axis=1)
        r=rewards(transitions,FIELDS);phi=potentials(states,FIELDS)
        np.testing.assert_allclose((r[:,1:]*GAMMA**np.arange(3)[:,None]).sum(0),-phi[0],atol=1e-12)
        np.testing.assert_allclose(r[:,1:].sum(0)+(1-GAMMA)*phi[1:].sum(0),-phi[0],atol=1e-12)
        self.assertEqual(r[-1,0],-10)
        self.assertEqual(r[:-1,0].sum(),0)

    def test_mc_returns_cross_blocks_and_discount_invalid_steps_to_native_terminal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);files=[]
            for block in range(2):
                s=np.zeros((1,128,2,len(FIELDS)));v=np.zeros((1,128),bool);v[0,127 if block==0 else 0]=True
                s[:,:,:,0]=1
                s[0,:,:,10]=block
                if block==1:
                    s[0,2,1,8]=1;s[0,2,1,9]=1
                    s[0,3:,:,8]=1;s[0,3:,:,9]=1
                name=f'b{block}.npz';np.savez(root/name,state=s,valid=v,actors=[0],fields=FIELDS,match_outcome=np.ones((1,128),np.int8))
                files.append(dict(file=name,demonstration=f'b{block}.ggd.gz'))
            (root/'reward-metadata.json').write_text(json.dumps(dict(files=files)))
            result=targets(root,root/'out');self.assertEqual(result['actors'],1)
            self.assertEqual(result['terminals_outside_imitation_mask'],1)
            with np.load(root/'out/b1.npz') as z:self.assertAlmostEqual(float(z['returns'][0,0]),-.5+.99**2*10,places=5)
            with np.load(root/'out/b0.npz') as z:self.assertAlmostEqual(float(z['returns'][0,127]),-.5+.99**3*10,places=5)

    def test_episode_curves_exclude_incomplete_frozen_prefix_and_keep_denominators(self):
        import p09b_results as report
        import xml.etree.ElementTree as ET
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);run=root/'rl-main';run.mkdir();out=root/'reports';out.mkdir()
            (run/'config.json').write_text(json.dumps({'learning_rate':3e-5}))
            update=dict(block=1,training_seconds=60,learning=dict(warmup=False,value_loss=1,ev_post0=.2,rollout_kl_m2=.1,entropy=.4,clip_fraction=.3,kl_coefficient=.05,valid_decisions=256))
            (run/'history.jsonl').write_text(json.dumps(update)+'\n')
            episode=dict(training_seconds=60,reward=9.6,components=[10,.1,.2,-.3,-.4],initialization_prefix_decisions=0,game=dict(lane_creeps_spawned=8,sides=[dict(controller='GGD',lh=3,deny=1,first_death_game_seconds=None)]))
            prefix={**episode,'initialization_prefix_decisions':128,'reward':999}
            (run/'episodes.jsonl').write_text('\n'.join(json.dumps(e) for e in [prefix]+[episode]*100)+'\n')
            with patch.object(report,'RUN',root),patch.object(report,'REPORT',out):updates,bins=report.curves()
            self.assertEqual(len(bins),1);self.assertEqual(bins[0]['matches'],100)
            self.assertAlmostEqual(bins[0]['reward_sum'],960);self.assertEqual(bins[0]['farm'],400);self.assertEqual(bins[0]['spawn'],800)
            self.assertEqual(json.loads((out/'rl-curve-coverage.json').read_text())['excluded_frozen_prefix_episodes'],1)
            ET.parse(out/'rl-curves.svg')

if __name__=='__main__':unittest.main()
