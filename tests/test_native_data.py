import gzip,json,struct,tempfile,unittest
from pathlib import Path
import numpy as np
from sf_duel.protos.duel_pb2 import Record,ModelSupplement,Execution
from sf_duel.native_data import NativeAdapter,action_label,padded_label,decode_action,examples,ITEM_IDS,SPELL_IDS
from sf_duel.observations import encode
from sf_duel.spec import ITEMS,SPELLS,UNIT_FIELDS,ENEMY_PRIVATE

def fixture():
 r=Record();s=r.observation;s.team_id=2;s.game_time=100;s.dota_time=10;s.game_state=5
 h=s.units.add(handle=10,name='npc_dota_hero_nevermore',team_id=2,unit_type=1,is_alive=True,health=500,health_max=600,mana=200,mana_max=300,facing=90,attack_damage=55,attack_speed=1.4,attack_range=525,attack_anim_point=.5,attack_projectile_speed=1200,reliable_gold=0,unreliable_gold=300,level=2,last_hits=3,denies=1)
 h.location.x=100;h.location.y=200
 for name in SPELLS:h.abilities.add(ability_id=SPELL_IDS[name],level=1,is_fully_castable=True)
 for name,slot in [('item_wraith_band',0),('item_tango',1),('item_tpscroll',15)]:h.items.add(ability_id=ITEM_IDS[name],slot=slot,is_fully_castable=True,charges=1)
 enemy=s.units.add(handle=11,name='npc_dota_hero_nevermore',team_id=3,unit_type=1,is_alive=True,health=400,health_max=600,mana=99,mana_max=300,reliable_gold=9999,unreliable_gold=9999,last_hits=99,denies=99,xp_needed_to_level=999);enemy.location.x=500;enemy.location.y=200
 creep=s.units.add(handle=21,name='npc_dota_creep_badguys_melee',team_id=3,unit_type=3,is_alive=True,health=35,health_max=550);creep.location.x=300;creep.location.y=200
 fountain=s.units.add(handle=90,name='dota_fountain',team_id=2,unit_type=10,is_alive=True);fountain.location.x=-7456;fountain.location.y=-6938
 r.decision.sequence=1;r.decision.team=2;r.decision.observed_tick=3000;r.decision.execute_tick=3004;r.decision.observed_dota_time=10;r.decision.action=3;r.decision.target=21
 r.supplement.team=2;r.supplement.tick=2998
 for name in ['item_boots','item_bracer','item_tpscroll','item_enchanted_mango','item_flask','item_tango','item_clarity']:r.supplement.shop.add(name=name,price=50,stock=5)
 r.supplement.trees.add(id=500,x=150,y=200,z=128)
 m=ModelSupplement(team=2,tick=3000,x=100,y=200,start_bought=True,next_cost=500)
 m.terrain.extend([.2]*64+[1]*128);m.units.add(handle=0,attack_interval=1.2)
 for name in SPELLS:m.skills.add(name=name,level=1,mana_cost=75,cast_point=.55)
 for name in ITEMS:m.items.add(name=name)
 for i,(x,y) in enumerate([(-1500,-1400),(-3190,-2926),(524,652),(2496,2112)]):m.towers.add(id=i+1,x=x,y=y,z=128)
 m.base.x=-7542;m.base.y=-6171
 return r,m

class NativeInputsTests(unittest.TestCase):
 def test_values_facing_and_enemy_privacy(self):
  r,m=fixture();e=encode(NativeAdapter().snapshot(r,m));fields=list(UNIT_FIELDS)
  self.assertAlmostEqual(e.arrays['heroes'][0,fields.index('health')],.25)
  self.assertAlmostEqual(e.arrays['heroes'][0,fields.index('facing_sin')],1)
  self.assertFalse(ENEMY_PRIVATE.intersection(e.refs[1]))
  before=e.arrays['heroes'].copy();r.observation.units[1].reliable_gold=1;r.observation.units[1].last_hits=0
  np.testing.assert_array_equal(before,encode(NativeAdapter().snapshot(r,m)).arrays['heroes'])
 def test_attack_and_tree_targets_survive_padding(self):
  r,m=fixture();e=encode(NativeAdapter().snapshot(r,m))
  for action,item,target in [(3,'',21),(10,'item_tango',500)]:
   r.decision.action=action;r.decision.item=item;r.decision.target=target
   a=action_label(r.decision,e);p=padded_label(a,e,64,64);d=decode_action(p,e,64,64,1,3000)
   self.assertEqual(d.target,target);self.assertEqual(d.action,action)
 def test_tp_fountain_is_exact_without_changing_teacher(self):
  r,m=fixture();e=encode(NativeAdapter().snapshot(r,m));r.decision.action=10;r.decision.item='item_tpscroll';r.decision.target=0;r.decision.x=-7456;r.decision.y=-6938
  a=action_label(r.decision,e);self.assertEqual((a['x'],a['y']),(64,64))
  d=decode_action(padded_label(a,e,32,32),e,32,32,1,3000);self.assertEqual((d.x,d.y),(-7456,-6938))
 def test_unavailable_hold_is_masked(self):
  r,m=fixture();e=encode(NativeAdapter().snapshot(r,m));self.assertFalse(e.arrays['action_mask'][5])
  with self.assertRaises(ValueError):decode_action({'action':5,'buy':0,'item':0,'target':0,'x':64,'y':64},e,16,32,1,3000)
 def test_native_current_flags_override_earlier_auxiliary_flags(self):
  r,m=fixture();r.observation.units[0].is_using_ability=True;r.observation.units[0].action_type=4
  e=encode(NativeAdapter().snapshot(r,m));self.assertTrue(e.snapshot['self']['is_using_ability']);self.assertTrue(e.snapshot['self']['is_attacking'])
 def test_missing_or_future_auxiliary_data_fails(self):
  r,m=fixture()
  with self.assertRaises(ValueError):NativeAdapter().snapshot(r,None)
  m.tick=3004
  with self.assertRaises(ValueError):NativeAdapter().snapshot(r,m)
 def test_missed_command_keeps_recurrent_observation(self):
  r,m=fixture()
  with tempfile.TemporaryDirectory() as directory:
   root=Path(directory);(root/'duel-result.json').write_text(json.dumps({'clock':'game_time','time':20,'period_ticks':4,'offset_ticks':4}))
   with gzip.open(root/'native-decisions.pb.gz','wb') as f:
    for value in (r,Record(model_supplement=m)):
     b=value.SerializeToString();f.write(struct.pack('<I',len(b))+b)
   rows=list(examples(root));self.assertEqual(len(rows),1);self.assertFalse(rows[0]['action_loss_mask']);self.assertEqual(rows[0]['reason'],'no_execution_ack')
