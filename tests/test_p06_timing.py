import sys
import unittest
import tempfile
from unittest.mock import patch
from pathlib import Path
import numpy as np
from fixtures import snapshot
from sf_duel.observations import encode
from sf_duel.spec import UNIT_FIELDS, COMBAT_UNIT_FIELDS
from sf_duel.timing_observations import TIMING_FIELDS, HONEST_FIELDS, ablate

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'gpu/scripts'))
from ggd_dataset import layout, decode


class TimingTests(unittest.TestCase):
    def test_dota_service_loads_honest_v3_and_rejects_oracle(self):
        import torch
        from sf_duel.model import Policy
        from sf_duel.native_policy_server import Service
        model=Policy(combat_inputs=True,timing_inputs=True)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);checkpoint=root/'model.pt'
            saved=dict(state_dict=model.state_dict(),config=dict(timing_oracle=False),gpu_pipeline=dict(recurrent_reset_interval=128,observation_storage='f16'))
            torch.save(saved,checkpoint)
            with patch('threading.Thread.start'):
                service=Service(checkpoint,root/'service',max_batch_warmup=0,device='cpu')
            self.assertTrue(service.model.timing_inputs)
            saved['config']['timing_oracle']=True;torch.save(saved,checkpoint)
            with self.assertRaisesRegex(ValueError,'oracle model cannot run in Dota'):
                Service(checkpoint,root/'rejected',max_batch_warmup=0,device='cpu')

    def test_native_timing_uses_game_clock_and_unknown_masks(self):
        from test_native_data import fixture
        from sf_duel.native_data import NativeAdapter
        r,m=fixture();u=r.observation.units[0]
        u.last_attack_time=99.8;u.anim_activity=1503;u.anim_cycle=.2
        adapter=NativeAdapter()
        legacy=adapter.snapshot(r,m)
        fresh=adapter.snapshot(r,m,observation_profile='ggd-v3')
        self.assertNotIn('since_attack',legacy['self'])
        self.assertAlmostEqual(fresh['self']['since_attack'],.2,places=4)
        self.assertAlmostEqual(fresh['self']['ready_in'],.625,places=4)
        self.assertTrue(fresh['self']['attack_animation'])
        u.ClearField('last_attack_time')
        unknown=adapter.snapshot(r,m,observation_profile='ggd-v3')
        self.assertNotIn('ready_in',unknown['self'])

    def test_v3_policy_and_batch_have_matching_widths(self):
        import torch
        from sf_duel.model import Policy
        from sf_duel.observations import batch, warmup_batch
        torch.set_num_threads(2)
        model=Policy(combat_inputs=True,timing_inputs=True)
        b=batch([encode(snapshot(),observation_profile='ggd-v3')])
        with torch.no_grad():
            fused,keys=model.encode_entities(b)
            model.encode_entities(warmup_batch(1,'cpu',True,True))
        self.assertEqual(tuple(fused.shape),(1,256))
        self.assertEqual(keys.shape[-1],64)
        self.assertEqual(model.hero[0].in_features,140)

    def test_live_oracle_is_zero_and_legacy_unchanged(self):
        s=snapshot()
        s['self'].update(attack_animation=True, since_attack=.2, ready_in=.4,
                         attack_cycle=.3,oracle_hp=999,oracle_windup_in=.5)
        before=encode(s,combat_inputs=True).arrays
        v3=encode(s,observation_profile='ggd-v3').arrays
        after=encode(s,combat_inputs=True).arrays
        for k in before:np.testing.assert_array_equal(before[k],after[k])
        at=2*len(COMBAT_UNIT_FIELDS);n=len(TIMING_FIELDS)
        self.assertEqual(v3['hero_combat'].shape[-1],at+2*n)
        np.testing.assert_allclose(v3['hero_combat'][0,at:at+3],[1,.2/3,.4/3])
        np.testing.assert_array_equal(v3['hero_combat'][:,at+HONEST_FIELDS:at+n],0)
        np.testing.assert_array_equal(v3['hero_combat'][:,at+n+HONEST_FIELDS:],0)

    def test_ablation_masks_and_cycle(self):
        s=snapshot();s['self'].update(attack_animation=True,attack_cycle=.4,since_attack=.2,ready_in=.4)
        v3=encode(s,observation_profile='ggd-v3').arrays
        at=2*len(COMBAT_UNIT_FIELDS);n=len(TIMING_FIELDS)
        v3['hero_combat'][0,at+3:at+n]=2
        v3['hero_combat'][0,at+n+3:]=1
        honest=ablate(v3,True,False);old=ablate(v3,False,False)
        self.assertEqual(honest['heroes'][0,list(UNIT_FIELDS).index('attack_cycle')],np.float32(.4))
        np.testing.assert_array_equal(old['hero_combat'][:,at:],0)
        self.assertEqual(old['heroes'][0,list(UNIT_FIELDS).index('attack_cycle')],0)
        self.assertEqual(v3['hero_combat'][0,at+3],2)

    def test_v3_layout_appends_to_v2_and_decoder_preserves_buckets(self):
        v2,w2=layout();v3,w3=layout(3)
        for name in v2:self.assertEqual(v2[name],v3[name])
        self.assertEqual(v3['hero_timing'][0],w2)
        pc,pt,pp=16,32,16;columns=[]
        for name,(offset,shape,boolean) in v3.items():
            indices=np.arange(offset,offset+int(np.prod(shape))).reshape(shape)
            if name in ('creeps','creep_valid','creep_ally','creep_combat','creep_timing'):indices=indices[:pc]
            elif name in ('trees','tree_valid'):indices=indices[:pt]
            elif name in ('projectiles','projectile_valid','projectile_combat'):indices=indices[:pp]
            elif name.endswith('_mask') and shape==(519,):indices=np.concatenate((indices[:2+pc],indices[258:262+pt],indices[-1:]))
            columns.extend(indices.flat)
        header=dict(version=3,buckets=[pc,pt,pp],columns=columns)
        raw=np.zeros((1,128,len(columns)),np.float16);signals=np.zeros((1,128,13),np.float32)
        start=columns.index(v3['hero_timing'][0]);raw[:,:,start]=.75
        decoded=decode(header,raw,signals)
        self.assertEqual(decoded['creep_combat'].shape,(128,pc,2*(len(COMBAT_UNIT_FIELDS)+len(TIMING_FIELDS))))
        np.testing.assert_array_equal(decoded['hero_combat'][:,0,2*len(COMBAT_UNIT_FIELDS)],.75)


if __name__=='__main__':unittest.main()
