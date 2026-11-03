"""P08 trial selection counts matches with deaths, not the death-event sum."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('p08_results',Path(__file__).resolve().parents[1]/'gpu/scripts/p08_results.py')
results=importlib.util.module_from_spec(spec);spec.loader.exec_module(results)

def fixture(death_games,early_games):
    games=[]
    for i in range(32):
        side='radiant'if i%2==0 else'dire'
        games.append({'sides':[{'controller':'GGD','side':side,'deaths':(2 if i==0 else 1)if i in death_games else 0,'first_death_game_seconds':100 if i in early_games else 200 if i in death_games else None}]})
    summary=[]
    for side in ['radiant','dire']:
        rows=[g['sides'][0]for g in games if g['sides'][0]['side']==side]
        summary.append(dict(controller='GGD',side=side,lh=500,deny=100,deaths=sum(x['deaths']for x in rows),attack_cycles_completed=1000,attack_cancellations=10,actions=[0]*11))
    return dict(complete=True,matches=32,wall_seconds=1,per_match=games,summary={'sides':summary})

class P08SelectionTests(unittest.TestCase):
    def check(self,deaths,early):
        with patch.object(results,'read',return_value=fixture(deaths,early)):
            return results.evaluation(Path('run/evaluation/report.json'))
    def test_four_death_matches_are_eligible_even_with_repeat_death_events(self):
        v=self.check({0,1,2,3},{0})
        self.assertEqual(sum(s['deaths']for s in v['sides']),5)
        self.assertEqual(v['death_matches'],4)
        self.assertTrue(v['trial_eligible'])
    def test_each_survival_limit_is_required(self):
        self.assertFalse(self.check({0,1,2,3,4},{0})['trial_eligible'])
        self.assertFalse(self.check({0,1,2,3},{0,1})['trial_eligible'])

if __name__=='__main__':unittest.main()
