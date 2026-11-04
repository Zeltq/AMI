import sys,unittest
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'gpu/scripts'))
from train_ggd import loss_weights,matching_actions
from sf_duel.model import Policy

class P09Tests(unittest.TestCase):
    def test_joint_agreement_ignores_unused_heads(self):
        y=torch.tensor([[0,0,0,0,64,64],[3,0,0,1,64,64],[1,0,0,0,5,6],[10,0,6,2,64,64]])
        pred={k:y[:,i].clone() for i,k in enumerate(('action','buy','item','target','x','y'))}
        pred['x'][0]=9;pred['target'][0]=7
        self.assertTrue(matching_actions(pred,y).all())
        pred['target'][1]=2;pred['y'][2]=7;pred['item'][3]=5
        self.assertEqual(matching_actions(pred,y).tolist(),[True,False,False,False])

    def test_rare_weights_and_padding(self):
        y=torch.tensor([[0,0,0,0,0,0],[3,0,0,2,0,0],[3,0,0,1,0,0],[6,0,0,0,0,0],[7,0,0,0,0,0],[8,0,0,0,0,0],[3,0,0,1,0,0]])
        valid=torch.tensor([True]*6+[False])
        config=dict(attack_weight=4,hero_attack_weight=7,action_weights={'6':3,'7':5,'8':9})
        torch.testing.assert_close(loss_weights(y,valid,config),torch.tensor([1.,4,7,3,5,9,0]))
        logits=torch.zeros(7,11,requires_grad=True)
        nll=-logits.log_softmax(-1)[torch.arange(7),y[:,0]]
        w=loss_weights(y,valid,config);(nll*w).sum().div(w.sum()).backward()
        self.assertEqual(logits.grad[-1].abs().sum(),0)
        self.assertAlmostEqual(float(logits.grad[4,7]/logits.grad[3,6]),5/3,places=5)

    def test_v4_initialization_and_metadata_contract(self):
        policy=Policy(combat_inputs=True,timing_inputs=True,enemy_inputs=True)
        self.assertEqual(policy.hero[0].in_features,172)
        self.assertEqual(policy.unit[0].in_features,140)
        with self.assertRaises(ValueError):Policy(combat_inputs=True,enemy_inputs=True)

if __name__=='__main__':unittest.main()
