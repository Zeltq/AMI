"""CPU-only checks for recovery routing and pre-RL model selection."""
import unittest
from pathlib import Path
from unittest.mock import patch
import p16_run as runner
class RecoveryWorkflow(unittest.TestCase):
    def test_recovery_engine_is_selected_explicitly(self):
        with patch.object(runner,'read',return_value={'active':True,'engine':'C:/AMI/runs/p16/environment-script-navigation/engine.json'}):
            self.assertTrue(runner.current_engine().endswith('environment-script-navigation/engine.json'))
    def test_pre_rl_selection_cannot_change_to_an_rl_checkpoint_on_controller_resume(self):
        entries=[dict(label=label,opponent=enemy,scripted_base_to_mid=True,complete=True,summary=f'{label}-{enemy}.json',weights=f'C:/AMI/runs/p16/{label}/weights') for label in ['nav-imitation','ppo-120m'] for enemy in ['v10','v9']]
        def read(path,default=None):
            if Path(path).name=='evaluations.json':return entries
            if Path(path).name=='navigation-mode.json':return {'active':True}
            wins=120 if Path(path).name.startswith('ppo') else 10
            c=dict(actors=128,outcomes={'win':wins},deaths={'numerator':128-wins},lh={'numerator':wins},deny={'numerator':0},mid_arrivals={'numerator':128},mid_arrival_elapsed_seconds={'numerator':3200},raze_hero_hit_rates={'near':{'numerator':1,'denominator':2}},death_causes={'tower':0})
            return {'controllers':{'CHiO':c}}
        with patch.object(runner,'read',side_effect=read),patch.object(runner,'save'):
            self.assertEqual(runner.choose_best()['label'],'ppo-120m')
            self.assertEqual(runner.choose_best({'nav-imitation','nav-dagger-1','nav-dagger-2'})['label'],'nav-imitation')
if __name__=='__main__':unittest.main()
