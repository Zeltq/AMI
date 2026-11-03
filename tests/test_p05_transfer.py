import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'gpu/scripts'))
from evaluate_dota_teacher import prepare_bots, digest, CALLBACK


class P05TransferTests(unittest.TestCase):
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
