import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'gpu/scripts'))
from evaluate_dota_teacher import prepare_bots, digest, CALLBACK
from p05_results import summarize


class P05TransferTests(unittest.TestCase):
    def test_censored_times_and_missing_distance_keep_denominators(self):
        rows=[dict(team=2,deaths=1,first_death=-3,lh=1,deny=2,actions=[1]+[0]*10,
                   mid_game_time=None,mid_elapsed=None,t1_mean=None,game_seconds=0),
              dict(team=2,deaths=0,first_death=None,lh=3,deny=1,actions=[0,2]+[0]*9,
                   mid_game_time=20,mid_elapsed=110,t1_mean=500,t1_samples=2,game_seconds=600)]
        side=summarize(rows)['2']
        self.assertEqual((side['pregame_deaths'],side['matches']), (1,2))
        self.assertEqual((side['first_death']['count'],side['first_death']['denominator']), (1,2))
        self.assertEqual(side['mid_elapsed']['mean'],110)
        self.assertEqual((side['t1_mean']['mean'],side['t1_samples']), (500,2))
        self.assertEqual(side['action_fractions'][1],dict(numerator=2,denominator=3))

    def test_cpu_serving_uses_rounded_input_sampling_and_issued_previous(self):
        import torch
        from unittest.mock import Mock,patch
        from test_native_data import fixture
        from sf_duel.native_policy_server import Service
        from sf_duel.native_data import NativeAdapter
        from sf_duel.protos.duel_pb2 import Decision
        r,m=fixture();r.model_supplement.CopyFrom(m);r.episode='P05-contract'
        r.decision.sequence=129
        service=Service.__new__(Service);service.device='cpu';service.stochastic=True;service.writer=None
        service.contracts={'current':(128,'f16')};service.reset_origin=None
        service.stats={'episodes':{},'requests':0,'batches':{}}
        model=Mock();model.combat_inputs=True
        model.initial_state.return_value=(torch.zeros(1,1,2),torch.zeros(1,1,2))
        def decide(b,memory,**kwargs):
            self.assertFalse(kwargs['deterministic'])
            self.assertEqual(b['heroes'].device.type,'cpu')
            torch.testing.assert_close(b['heroes'],b['heroes'].half().float(),rtol=0,atol=0)
            a={k:torch.tensor([v]) for k,v in dict(action=1,buy=0,item=0,target=0,x=65,y=62).items()}
            return a,None,None,None,memory
        model.decide.side_effect=decide;service.models={'current':model}
        actor=dict(seq=128,tick=2996,deaths=0,previous=None,adapter=NativeAdapter(),
                   state=(torch.ones(1,1,2),torch.ones(1,1,2)),shots=set(),cooldowns={})
        service.actors={('P05-contract',2):actor}
        reply=Decision.FromString(service.decide_group([r],'current')[0])
        model.initial_state.assert_called_once_with(1)
        self.assertEqual((reply.action,reply.x,reply.y),(1,116,168))
        self.assertEqual((actor['previous']['action'],actor['previous']['x'],actor['previous']['y']),(1,65,62))

    def test_each_side_routes_to_exactly_one_controller_and_pins_teacher(self):
        engine={'combat':[1], 'prototypes':[[0]*17 for _ in range(9)], 'hero':[0]*26}
        for side in (2,3):
            with self.subTest(side=side),tempfile.TemporaryDirectory() as folder:
                source=Path(folder)/'bots'
                manifest=prepare_bots(source,side,engine,123)
                for name in ('scripted_teacher_v9.lua','scripted_teacher_v9_live.lua'):
                    self.assertEqual(manifest[name],digest(ROOT/'bots'/name))
                settings=(source/'p05_settings.lua').read_text()
                self.assertIn(f'["model_team"]={side}',settings)
                self.assertIn(f'["teacher_team"]={5-side}',settings)
                self.assertIn('if GetTeam()==settings.model_team then controller=require("bots/native_executor")\nelse controller=require("bots/scripted_teacher_v9_live") end',CALLBACK)
                self.assertEqual(CALLBACK.count('pcall(controller.Think)'),1)
                self.assertEqual((source/'native_commands_2.lua').read_text(),'return {}\n')
                self.assertIn('ItemPurchaseThink()', (source/'item_purchase_generic.lua').read_text())


if __name__=='__main__':unittest.main()
