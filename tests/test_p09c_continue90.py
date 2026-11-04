import json,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'gpu/scripts'))
from p09c_continue90 import make_config

class ContinuationTests(unittest.TestCase):
    def test_reduced_limit_is_fifty_total_minutes_with_same_hyperparameters(self):
        base=json.loads((ROOT/'reports/P09c/rl-main-001.json').read_text())
        progress=json.loads((ROOT/'reports/P09c/final-checkpoint-audit.json').read_text())['progress']
        original=make_config(base,progress,'checkpoint-000314',90)
        amended=make_config(base,progress,'checkpoint-000314',50)
        self.assertAlmostEqual(amended['minutes']*60-progress['training_seconds'],3000,places=8)
        self.assertEqual({k:v for k,v in original.items() if k!='minutes'},{k:v for k,v in amended.items() if k!='minutes'})
        self.assertFalse(amended['evaluate'])

    def test_real_checkpoint_metadata_keeps_schedule_and_all_learning_settings(self):
        base=json.loads((ROOT/'reports/P09c/rl-main-001.json').read_text())
        progress=json.loads((ROOT/'reports/P09c/final-checkpoint-audit.json').read_text())['progress']
        cfg=make_config(base,progress,'checkpoint-000314')
        allowed={'minutes','resume_checkpoint','initialize_weights','evaluate','skip_warmup','critic_only','begin_decay_on_resume','stop_at_minutes','max_blocks','schedule_minutes','kl_decay_start_minutes','kl_start','kl_end'}
        self.assertEqual({k:v for k,v in base.items() if k not in allowed},{k:v for k,v in cfg.items() if k not in allowed})
        self.assertAlmostEqual(cfg['minutes']*60-progress['training_seconds'],5400,places=8)
        self.assertEqual(cfg['schedule_minutes'],progress['schedule_minutes'])
        self.assertFalse(cfg['evaluate']);self.assertTrue(cfg['skip_warmup']);self.assertFalse(cfg['begin_decay_on_resume'])
        for seconds in (progress['training_seconds'],cfg['minutes']*60):
            fraction=max(0,min(1,(seconds/60-cfg['kl_decay_start_minutes'])/(cfg['schedule_minutes']-cfg['kl_decay_start_minutes'])))
            self.assertAlmostEqual(cfg['kl_start']+(cfg['kl_end']-cfg['kl_start'])*fraction,.05,places=6)

if __name__=='__main__':unittest.main()
