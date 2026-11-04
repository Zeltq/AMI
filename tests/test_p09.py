import sys,unittest
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'gpu/scripts'))
from train_ggd import loss_weights,matching_actions,validate_zog_weights
from p09_data import rare_action_weights
from p09_results import summarize,ratio
from sf_duel.model import Policy

class P09Tests(unittest.TestCase):
    def test_report_keeps_large_integer_denominators_exact(self):
        self.assertEqual(ratio(965262,1025224),'965262/1025224')
        self.assertEqual(ratio(0.,0.),'0/0')

    def test_user_weight_cap_and_hero_fixed_four(self):
        counts=[9749,0,0,0,0,0,1,40,210,0,0,1]
        weights,hero=rare_action_weights(counts)
        self.assertEqual(weights,{'6':4.,'7':2.5,'8':1.})
        self.assertEqual(hero,4.)
        config=dict(model_name='ZOg',attack_weight=4.,hero_attack_weight=hero,action_weights=weights)
        validate_zog_weights(config)
        for changed in (dict(config,hero_attack_weight=5.),dict(config,action_weights={'6':32.}),dict(config,action_weights={'7':float('nan')})):
            with self.assertRaises(ValueError):validate_zog_weights(changed)

    def test_opponent_casts_come_from_the_same_matches(self):
        def hero(side,near):
            row={k:0 for k in ('lh','deny','attack_cycles_completed','hero_kills','hero_attacks_released','attack_hero_commands','damage_from_hero','damage_from_creeps','damage_from_tower','deaths')}
            row.update(side=side,controller='GGD',first_death_game_seconds=None,spell_completions=[near,near+1,near+2,0],raze_hero_hits_by_spell=[0,0,0],death_causes=[dict(count=0,own_t1_distance_sum=0,enemy_t1_distance_sum=0) for _ in range(5)])
            return row
        games=[dict(game=0,result=0,outcome_reason='LH600',lane_creeps_spawned=8,sides=[hero('radiant',2),hero('dire',7)]),dict(game=1,result=0,outcome_reason='LH600',lane_creeps_spawned=8,sides=[hero('radiant',11),hero('dire',5)])]
        rows=summarize('ZOg-e1',[(20261103,g) for g in games],None)
        self.assertEqual([(r['near_casts'],r['opponent_near_casts'],r['matches']) for r in rows],[(2,7,1),(5,11,1),(7,18,2)])
        self.assertEqual(rows[-1]['opponent_mid_casts'],20)
        self.assertEqual(rows[-1]['opponent_far_casts'],22)

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
