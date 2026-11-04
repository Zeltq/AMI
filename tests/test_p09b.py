import json,sys,tempfile,unittest
from pathlib import Path
import numpy as np
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

if __name__=='__main__':unittest.main()
