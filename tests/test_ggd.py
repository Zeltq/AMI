import copy,unittest
import torch
from fixtures import snapshot
from sf_duel.spec import *
from sf_duel.model import Policy
from sf_duel.observations import encode,batch
from sf_duel.combat_observations import CombatMemory

class GgdTests(unittest.TestCase):
 def scene(self):
  s=snapshot();s['self'].update(armor=2,attack_interval=1.2)
  s['creeps']=[dict(id=20,type='creep',name='npc_dota_creep_badguys_melee',visible=True,team=3,x=200,y=0,health=100,health_max=550,armor=2,attack_damage=21,attack_interval=1),dict(id=21,type='creep',name='npc_dota_creep_goodguys_ranged',visible=True,team=2,x=-300,y=0,health=300,health_max=300,armor=0,attack_damage=23,attack_interval=1,attack_target_id=20)]
  s['projectiles']=[dict(x=-150,y=0,speed=900,is_attack=True,target_self=False,caster_enemy=False,target_id=20,caster_id=21,damage=99999)]
  return s
 def test_relations_and_last_seen_caster(self):
  s=self.scene();memory=CombatMemory();a=encode(s,combat_inputs=True,combat_memory=memory)
  f=list(COMBAT_UNIT_FIELDS);r=a.arrays['creep_combat'][0]
  self.assertAlmostEqual(r[f.index('incoming_attackers')]*16,1)
  self.assertAlmostEqual(r[f.index('incoming_projectile_damage')]*500,31,places=4)
  self.assertAlmostEqual(a.arrays['creep_combat'][1,f.index('attack_target_dx')]*2048,200)
  b=copy.deepcopy(s);b['projectiles'][0]['damage']=-99999
  self.assertTrue((a.arrays['creep_combat']==encode(b,combat_inputs=True,combat_memory=memory).arrays['creep_combat']).all())
  b['creeps'].pop();cached=encode(b,combat_inputs=True,combat_memory=memory);fresh=encode(b,combat_inputs=True)
  self.assertEqual(cached.arrays['creep_combat'][0,len(f)+f.index('incoming_projectile_damage')],1)
  self.assertEqual(fresh.arrays['creep_combat'][0,len(f)+f.index('incoming_projectile_damage')],0)
 def test_legacy_dimensions_and_ggd_padding(self):
  torch.set_num_threads(2);s=self.scene();other=copy.deepcopy(s);other['creeps']+= [dict(s['creeps'][0],id=100+i,x=1000+i*40) for i in range(18)]
  self.assertEqual(Policy().hero[0].in_features,96)
  torch.manual_seed(1);model=Policy(combat_inputs=True).eval()
  self.assertEqual(model.hero[0].in_features,118);self.assertEqual(model.unit[0].in_features,118);self.assertEqual(model.projectile[0].in_features,28)
  self.assertEqual(sum(p.numel() for p in model.parameters())-sum(p.numel() for p in Policy().parameters()),4736)
  a=encode(s,combat_inputs=True);b=encode(other,combat_inputs=True)
  with torch.no_grad():
   first=model(batch([a]))[0];padded=model(batch([a,b]))[0]
  torch.testing.assert_close(first[0],padded[0],rtol=1e-5,atol=1e-6)
 def test_hidden_source_does_not_enter_relations(self):
  s=self.scene();a=encode(s,combat_inputs=True)
  b=copy.deepcopy(s);b['creeps'].append(dict(s['creeps'][1],id=999,visible=False,x=7000,attack_damage=99999))
  c=encode(b,combat_inputs=True)
  for key in ['hero_combat','creep_combat','tower_combat','projectile_combat']:self.assertTrue((a.arrays[key]==c.arrays[key]).all())
