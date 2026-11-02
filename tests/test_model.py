import copy
import unittest
import torch
from sf_duel.observations import encode,batch,target_ref
from sf_duel.model import Policy
from sf_duel.spec import *
from fixtures import snapshot

torch.set_num_threads(2)

class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        self.model=Policy().eval()

    def test_packed_transfers_preserve_every_value_dtype_and_shape(self):
        from sf_duel.observations import numpy_batch,packed_tensors
        a=snapshot();b=copy.deepcopy(a);b['creeps']=[dict(a['enemy'],id=i+20,type='creep',x=i*20) for i in range(37)]
        arrays=numpy_batch([encode(a).arrays,encode(b).arrays]);packed=packed_tensors(arrays,'cpu')
        for key,value in arrays.items():torch.testing.assert_close(packed[key],torch.as_tensor(value),rtol=0,atol=0)
        with torch.no_grad():
            expected=self.model.decide({k:torch.as_tensor(v) for k,v in arrays.items()},deterministic=True)
            actual=self.model.decide(packed,deterministic=True)
        for key in expected[0]:torch.testing.assert_close(actual[0][key],expected[0][key],rtol=0,atol=0)
        torch.testing.assert_close(actual[1],expected[1],rtol=0,atol=0)

    def test_omitting_unused_inference_entropy_preserves_policy_and_memory(self):
        b=batch([encode(snapshot())])
        with torch.no_grad():
            expected=self.model.decide(b,deterministic=True)
            actual=self.model.decide(b,deterministic=True,need_entropy=False)
        for k in expected[0]:torch.testing.assert_close(actual[0][k],expected[0][k],rtol=0,atol=0)
        for i in (1,3):torch.testing.assert_close(actual[i],expected[i],rtol=0,atol=0)
        for a,e in zip(actual[4],expected[4]):torch.testing.assert_close(a,e,rtol=0,atol=0)

    def test_enemy_private_and_fog_are_not_inputs(self):
        first=snapshot();second=copy.deepcopy(first)
        second['enemy'].update(gold=123456,xp=9912,last_hits=99,souls=30)
        torch.testing.assert_close(batch([encode(first)])['heroes'],batch([encode(second)])['heroes'])
        first['enemy']['visible']=False;second['enemy'].update(visible=False,x=999,y=112,health=2)
        a,b=encode(first),encode(second)
        self.assertFalse(a.arrays['heroes'][1].any())
        self.assertTrue((a.arrays['heroes']==b.arrays['heroes']).all())
        second['enemy_items'][0].update(inventory=123,stash=99,courier=50,cooldown=10)
        self.assertFalse(encode(second).arrays['enemy_items'].any())

    def test_enemy_inventory_has_no_private_item_state(self):
        first=snapshot();second=copy.deepcopy(first)
        second['enemy_items'][0].update(stash=100,courier=50,cooldown=99,castable=True)
        torch.testing.assert_close(batch([encode(first)])['enemy_items'],batch([encode(second)])['enemy_items'])
        first['enemy_items'][3]=[]
        self.assertFalse(encode(first).arrays['enemy_items'][3].any())

    def test_padding_does_not_change_individual_decision(self):
        a=snapshot();b=copy.deepcopy(a)
        b['creeps']=[dict(a['enemy'],id=i+20,type='creep',x=i*20) for i in range(37)]
        first=batch([encode(a)])
        both=batch([encode(a),encode(b)])
        with torch.no_grad():
            h1,*_=self.model(first);h2,*_=self.model(both)
        torch.testing.assert_close(h1[0],h2[0],atol=1e-6,rtol=1e-5)

    def test_shared_encoder_keeps_distinct_targets_and_permutation(self):
        a=snapshot()
        a['creeps']=[dict(a['enemy'],id=20,type='creep',x=30,health=30),
                     dict(a['enemy'],id=21,type='creep',x=90,health=300)]
        b=copy.deepcopy(a);b['creeps'].reverse()
        with torch.no_grad():
            h1,k1,*_=self.model(batch([encode(a)]));h2,k2,*_=self.model(batch([encode(b)]))
        torch.testing.assert_close(h1,h2,atol=1e-6,rtol=1e-5)
        torch.testing.assert_close(k1[:,2],k2[:,3])
        self.assertFalse(torch.equal(k1[:,2],k1[:,3]))

    def test_target_padding_mapping(self):
        s=snapshot();e=encode(s)
        self.assertEqual(target_ref(e,2+5,5,8)['name'],TOWERS[0])
        self.assertEqual(target_ref(e,6+5+8,5,8)['id'],-1)
        self.assertIsNone(target_ref(e,2,5,8))

    def test_masks_log_probs_gradients_and_memory(self):
        s=snapshot();s['legal']['actions']=[False]*11;s['legal']['actions'][3]=True
        # CONTINUE must remain legal by contract, but force ATTACK in evaluation.
        s['legal']['actions'][0]=True
        b=batch([encode(s),encode(s)])
        chosen,lp,entropy,value,state=self.model.decide(b)
        chosen['action']=torch.full((2,),int(Act.ATTACK),dtype=torch.long)
        chosen['target']=torch.ones(2,dtype=torch.long)
        _,lp,entropy,value,state=self.model.decide(b,actions=chosen)
        (-lp.mean()+value.square().mean()-.01*entropy.mean()).backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in self.model.parameters() if p.grad is not None))
        self.assertIsNotNone(self.model.memory.weight_hh_l0.grad)
        with torch.no_grad():
            h1,*_=self.model(b);h2,*_=self.model(b,state)
        self.assertFalse(torch.equal(h1,h2))

    def test_sampled_actions_have_reproducible_log_probability(self):
        b=batch([encode(snapshot()) for _ in range(8)])
        chosen,lp,*_=self.model.decide(b)
        _,evaluated,*_=self.model.decide(b,actions=chosen)
        torch.testing.assert_close(lp,evaluated)
        self.assertTrue(torch.isfinite(lp).all())

    def test_unused_heads_are_canonical_and_do_not_change_probability(self):
        b=batch([encode(snapshot())])
        a={k:torch.tensor([v]) for k,v in blank_action().items()}
        with torch.no_grad():
            chosen,lp,*_=self.model.decide(b,actions=a)
            a.update(x=torch.tensor([7]),y=torch.tensor([109]),item=torch.tensor([7]))
            again,other,*_=self.model.decide(b,actions=a)
        torch.testing.assert_close(lp,other)
        for key,value in blank_action().items():self.assertEqual(again[key].item(),value)

    def test_memory_gradient_crosses_decision_boundaries(self):
        b=batch([encode(snapshot())]);b['heroes'].requires_grad_()
        _,_,_,state=self.model(b)
        next_b={k:v.detach() for k,v in b.items()}
        _,_,value,_=self.model(next_b,state)
        value.sum().backward()
        self.assertGreater(b['heroes'].grad.abs().sum().item(),0.)

if __name__=='__main__':unittest.main()
