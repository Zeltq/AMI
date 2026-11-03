import json
import sys
import unittest
from pathlib import Path
import numpy as np
from fixtures import snapshot
from sf_duel.enemy_observations import CastMemory,FIELDS
from sf_duel.observations import encode,batch,warmup_batch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'gpu/scripts'))
from ggd_dataset import layout

class PublicSpellTests(unittest.TestCase):
    def test_visible_onsets_cancellation_fog_and_reappearance(self):
        m=CastMemory()
        self.assertNotIn('since_cast_1',m.observe(1,0,0,True))
        r=m.observe(1,.1,1,True);self.assertEqual(r['since_cast_1'],0)
        self.assertAlmostEqual(m.observe(1,.2,1,True)['cast_phase'],.1/.55)
        # A cancelled cast remains an observed onset, never a claimed cooldown.
        self.assertAlmostEqual(m.observe(1,.3,0,True)['since_cast_1'],.2)
        self.assertEqual(m.observe(1,.4,None,False),{})
        r=m.observe(1,.5,2,True)
        self.assertNotIn('cast_phase',r);self.assertNotIn('since_cast_2',r)
        self.assertAlmostEqual(r['since_cast_1'],.4)

    def test_v4_masks_and_legacy_arrays(self):
        s=snapshot();old=encode(s,observation_profile='ggd-v3').arrays
        s['enemy']['v4']={'cast_2':1,'raze_remaining':4,'spell_level_2':3}
        new=encode(s,observation_profile='ggd-v4').arrays
        for name,value in old.items():
            np.testing.assert_array_equal(value,new[name][...,:value.shape[-1]] if name=='hero_combat' else new[name])
        self.assertEqual(new['hero_combat'].shape,(2,76))
        self.assertAlmostEqual(new['hero_combat'][1,44+15],.4)
        s['enemy']['visible']=False
        np.testing.assert_array_equal(encode(s,observation_profile='ggd-v4').arrays['hero_combat'][1,44:],0)

    def test_policy_accepts_v4_batch_and_layout_is_append_only(self):
        import torch
        from sf_duel.model import Policy
        torch.set_num_threads(2)
        model=Policy(combat_inputs=True,timing_inputs=True,enemy_inputs=True)
        for b in (batch([encode(snapshot(),observation_profile='ggd-v4')]),warmup_batch(1,'cpu',True,True,True)):
            with torch.no_grad():fused,_=model.encode_entities(b)
            self.assertEqual(tuple(fused.shape),(1,256))
        old,w3=layout(3);new,w4=layout(4)
        for k in old:self.assertEqual(old[k],new[k])
        self.assertEqual(w4-w3,4*len(FIELDS));self.assertEqual(new['hero_enemy'][0],w3)

    def test_native_requires_measured_supplement_and_preserves_it(self):
        from test_native_data import fixture
        from sf_duel.native_data import NativeAdapter
        r,m=fixture();adapter=NativeAdapter()
        with self.assertRaisesRegex(ValueError,'spell supplement'):adapter.snapshot(r,m,observation_profile='ggd-v4')
        for u in m.units:u.v4_json=json.dumps({'cast_0':1,'spell_level_1':2})
        for u in r.observation.units:
            if u.name=='npc_dota_hero_nevermore' and u.team_id!=r.decision.team:
                m.units.add(handle=u.handle,v4_json=json.dumps({'cast_0':1,'spell_level_1':2}))
        result=adapter.snapshot(r,m,observation_profile='ggd-v4')
        self.assertEqual(result['self']['v4']['spell_level_1'],2)

if __name__=='__main__':unittest.main()
