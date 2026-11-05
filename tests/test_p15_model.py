import sys
from pathlib import Path
import unittest
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'gpu/scripts'))
from p15_model_fixture import batch
from sf_duel.chio_model import Policy,HEADS


class ConditionalHeads(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_log_probability_and_ppo_gradients_use_only_selected_head(self):
        for kind,active in [(1,None),(0,'direction_head'),(2,'enemy_query'),(3,'ally_query')]:
            with self.subTest(kind=kind):
                torch.manual_seed(3);model=Policy();b,_=batch(model,n=4);b['action_mask'][:]=True
                h=torch.randn(4,256);keys=torch.randn(4,12,64)
                labels={name:torch.zeros(4,dtype=torch.long) for name in HEADS};labels['action'][:]=kind
                if kind==2:labels['enemy_target'][:]=1
                _,logp,_,value,_=model.heads(b,h,keys,actions=labels)
                expected=torch.log_softmax(model.action_head(h),-1)[:,kind]
                condition=torch.cat((h,model.action_embed(labels['action'])),-1)
                if kind==0:expected+=torch.log_softmax(model.direction_head(condition),-1)[:,0]
                elif kind in (2,3):
                    query=model.enemy_query if kind==2 else model.ally_query;mask=b['enemy_target_mask' if kind==2 else 'ally_target_mask']
                    logits=torch.einsum('bu,bnu->bn',query(condition),keys)/8
                    expected+=torch.log_softmax(logits.masked_fill(~mask,-1e30),-1)[:,1 if kind==2 else 0]
                torch.testing.assert_close(logp,expected)
                ratio=(logp-(logp.detach()+.03)).exp();adv=torch.tensor([1.,-.5,2.,-1.])
                loss=-torch.minimum(ratio*adv,ratio.clamp(.8,1.2)*adv).mean()+.5*value.square().mean()
                self.assertTrue(torch.isfinite(loss));loss.backward()
                for name in ('direction_head','enemy_query','ally_query'):
                    norm=getattr(model,name).weight.grad.abs().sum().item()
                    self.assertGreater(norm,0) if name==active else self.assertEqual(norm,0)

    def test_no_purchase_or_item_or_coordinate_heads(self):
        model=Policy()
        self.assertEqual(model.memory.hidden_size,256)
        self.assertEqual(model.action_head.out_features,13);self.assertEqual(model.direction_head.out_features,36)
        for name in ('buy_head','item_head','x_head','y_head'):self.assertFalse(hasattr(model,name))

    def test_unused_labels_do_not_change_log_probability(self):
        model=Policy();b,_=batch(model,n=4);b['action_mask'][:]=True;h=torch.randn(4,256);keys=torch.randn(4,12,64)
        labels={name:torch.zeros(4,dtype=torch.long) for name in HEADS};labels['action'][:]=1
        first=model.heads(b,h,keys,actions=labels)[1]
        labels['direction'][:]=35;labels['enemy_target'][:]=11;labels['ally_target'][:]=10
        torch.testing.assert_close(first,model.heads(b,h,keys,actions=labels)[1],rtol=0,atol=0)


if __name__=='__main__':unittest.main()
