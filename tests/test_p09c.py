import json,sys,tempfile,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'gpu/scripts'))
from p09c_reward import rewards,targets,FIELDS

class EventRewardTests(unittest.TestCase):
    def test_exact_losses_pre_step_max_and_terminal_once(self):
        a=[2,1,500,1000,200,400,8,4,600,1000,100,400,0,0]
        b=[3,2,400,1200,150,500,9,5,400,1200,300,500,1,1]
        np.testing.assert_allclose(rewards(np.array([a,b],float),list(FIELDS)),[10,.5,.2,-.3,.6,-.0625])
        np.testing.assert_array_equal(rewards(np.array([b,b],float),list(FIELDS)),np.zeros(6))

    def test_growth_and_enemy_farm_are_zero_even_if_fraction_falls(self):
        a=[0,0,500,1000,200,400,0,0,600,1000,100,400,0,0]
        b=[0,0,550,1200,250,500,3,2,650,1200,150,500,0,0]
        np.testing.assert_array_equal(rewards(np.array([a,b],float),list(FIELDS)),np.zeros(6))

    def test_defeat_draw_and_dead_padding(self):
        s=np.zeros((2,len(FIELDS)));s[:,[3,5,9,11]]=100
        s[1,12]=1;s[1,13]=-1
        self.assertEqual(rewards(s,list(FIELDS))[0],-10)
        s[1,13]=0;self.assertEqual(rewards(s,list(FIELDS)).sum(),0)

    def test_old_fraction_metadata_rejected(self):
        with self.assertRaises(ValueError):rewards(np.zeros((2,2)),['own_hp_fraction','enemy_hp_fraction'])

    def test_mc_event_value_survives_and_invalid_steps_are_discounted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);fields=list(FIELDS)+['game_seconds'];s=np.zeros((1,128,2,len(fields)))
            s[:,:,:,[3,5,9,11]]=100;valid=np.zeros((1,128),bool);valid[0,0]=True
            s[0,0,1,0]=1;s[0,1:,:,0]=1
            s[0,2,1,12]=1;s[0,2,1,13]=1;s[0,3:,:,12]=1;s[0,3:,:,13]=1
            np.savez(root/'b.npz',state=s,valid=valid,actors=[0],fields=fields,match_outcome=np.ones((1,128),np.int8))
            (root/'reward-metadata.json').write_text(json.dumps({'files':[{'file':'b.npz','demonstration':'b.ggd.gz'}]}))
            r=targets(root,root/'out');self.assertEqual(r['version'],3);self.assertEqual(r['terminals_outside_imitation_mask'],1)
            with np.load(root/'out/b.npz') as z:self.assertAlmostEqual(float(z['returns'][0,0]),.5+.99**2*10,places=5)

    def test_pilot_gate_checks_rate_and_early_deaths(self):
        from p09c_rl import healthy
        row=dict(lh=1120,deny=1,lane_creeps_spawned=10058,early_deaths=24,matches=128)
        self.assertTrue(healthy(row))
        self.assertFalse(healthy({**row,'deny':0}))
        self.assertFalse(healthy({**row,'early_deaths':25}))
        self.assertFalse(healthy({**row,'lane_creeps_spawned':11000}))
        self.assertFalse(healthy({**row,'matches':64}))

if __name__=='__main__':unittest.main()
