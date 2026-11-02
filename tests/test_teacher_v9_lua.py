import copy,hashlib,unittest
from pathlib import Path
try:
 from lupa.lua51 import LuaRuntime
except ImportError:
 LuaRuntime=None

ROOT=Path(__file__).resolve().parents[1]

@unittest.skipIf(LuaRuntime is None,'CPU Lua 5.1 runtime required (lupa)')
class TeacherV9LuaTests(unittest.TestCase):
 def setUp(self):
  self.lua=LuaRuntime(unpack_returned_tuples=True)
  self.core=self.lua.execute((ROOT/'bots/scripted_teacher_v9.lua').read_text())
 def table(self,v):
  if isinstance(v,dict):return self.lua.table_from({k:self.table(x) for k,x in v.items()})
  if isinstance(v,list):return self.lua.table_from([self.table(x) for x in v])
  return v
 def unit(self,id,kind=2,team=3,x=300,hp=30):
  return dict(id=id,kind=kind,team=team,x=x,y=0,hp=hp,max_hp=550,damage=21,armor=2,regen=0,range=100,speed=305,interval=1,point=.467,projectile_speed=0,facing=0,ready=0,windup=0,target=0,order=0)
 def scene(self,side=2):
  me=self.unit(1,kind=0,team=side,x=0,hp=580);me.update(max_hp=580,damage=50,range=525,point=.5,interval=1.6,projectile_speed=1200,facing=0 if side==2 else 3.141592653589793)
  creep=self.unit(10,team=5-side,x=300 if side==2 else -300)
  return dict(own=me,units=[me,creep],shots=[],time=100,dota_time=10,can_act=True,can_buy={},can_item={},trees=[],execute_ticks=7)
 def decide(self,s,m=None):return self.core.decide(self.table(s),m or self.core.new())
 def test_pins_the_selected_gpu_teacher(self):
  self.assertEqual(self.core.source_sha256,hashlib.sha256((ROOT/'gpu/src/scripted_teacher.rs').read_bytes()).hexdigest())
 def test_last_hit_both_sides_and_seven_tick_delay(self):
  for side in [2,3]:
   with self.subTest(side=side):
    s=self.scene(side);a,d=self.decide(s)
    self.assertEqual(a.action,3);self.assertEqual(a.target_id,10)
    self.assertGreaterEqual(d.forecasts[10].hit-s['time'],7/30+s['own']['point'])
 def test_deny_both_sides_and_half_health_gate(self):
  for side in [2,3]:
   with self.subTest(side=side):
    s=self.scene(side);s['units'][1]['team']=side
    self.assertEqual(self.decide(s)[0].action,3)
    s['units'][1]['hp']=300;self.assertNotEqual(self.decide(s)[0].action,3)
 def test_three_attackers_predict_the_own_landing_health(self):
  s=self.scene();s['units'][1]['hp']=94
  for i in range(3):
   u=self.unit(20+i,team=2,x=320+i*4,hp=550);u.update(target=10,windup=100.4,windup_target=10)
   s['units'].append(u)
  a,d=self.decide(s);self.assertEqual(a.action,3);self.assertAlmostEqual(d.forecasts[10].predicted_hp,37)
 def test_competing_targets_are_deterministic_and_current_attack_is_preserved(self):
  s=self.scene();m=self.core.new();first=self.decide(s,m)[0];self.assertEqual(first.target_id,10)
  s['time']+=4/30;s['units'].append(self.unit(11,x=280,hp=15))
  a,d=self.decide(s,m);self.assertTrue(d.protected);self.assertEqual(a.action,0)
  x=self.decide(s)[0];y=self.decide(copy.deepcopy(s))[0];self.assertEqual((x.action,x.target_id),(y.action,y.target_id))
 def test_dead_target_releases_the_pending_attack(self):
  s=self.scene();m=self.core.new();self.decide(s,m)
  s['time']+=4/30;s['units'][1]['hp']=0;s['units'].append(self.unit(11,hp=20))
  a,_=self.decide(s,m);self.assertEqual((a.action,a.target_id),(3,11))
 def test_projectile_damage_is_causal_and_survives_the_caster(self):
  s=self.scene();s['units'][1]['hp']=70
  source=self.unit(4,kind=3,team=2,x=600,hp=300);source['damage']=23;s['units'].append(source)
  s['shots']=[dict(caster=4,target=10,x=450,y=0,speed=900,damage=99999)]
  m=self.core.new();a,d=self.decide(s,m);self.assertEqual(a.action,3);self.assertAlmostEqual(d.forecasts[10].predicted_hp,39)
  s['units'].pop();s['shots'][0]['damage']=-99999
  a,d=self.decide(s,m);self.assertEqual(a.action,0);self.assertAlmostEqual(d.forecasts[10].predicted_hp,39)
 def test_existing_own_projectile_prevents_a_second_attack(self):
  s=self.scene();s['units'][1]['hp']=30;s['shots']=[dict(caster=1,target=10,x=0,y=0,speed=1200)]
  self.assertNotEqual(self.decide(s)[0].action,3)
 def test_departure_does_not_wait_for_creeps(self):
  for side in [2,3]:
   s=self.scene(side);s['dota_time']=-83;s['units']=s['units'][:1];s['own'].update(x=-6700 if side==2 else 6800,y=-6700 if side==2 else 6500)
   a,d=self.decide(s);self.assertEqual(a.action,1);self.assertEqual(d.candidates,0)
 def test_sustain_keeps_the_same_thresholds(self):
  s=self.scene();s['units']=s['units'][:1];s['own']['hp']=150;s['own_tower']=dict(x=-1000,y=-900);s['can_item']={5:True}
  a,_=self.decide(s);self.assertEqual((a.action,a.item),(10,5))

@unittest.skipIf(LuaRuntime is None,'CPU Lua 5.1 runtime required (lupa)')
class TeacherV9LiveClockTests(unittest.TestCase):
 def setUp(self):
  self.lua=LuaRuntime(unpack_returned_tuples=True)
  self.lua.execute('''
clock=100;player_is_bot=true;team=3;orders={};snapshots=0
GAME_STATE_PRE_GAME=4;GAME_STATE_GAME_IN_PROGRESS=5
function GameTime() return clock end
function DotaTime() return -10 end
function GetGameState() return 4 end
function IsPlayerBot() return player_is_bot end
local bot={}
function bot:GetUnitName() return "npc_dota_hero_nevermore" end
function bot:GetTeam() return team end
function bot:GetPlayerID() return 1 end
function bot:IsAlive() return true end
function bot:GetLastHits() return 0 end
function bot:GetDenies() return 0 end
function bot:GetHealth() return 580 end
function GetBot() return bot end
package.loaded["bots/settings"]={teacher_team=3}
package.loaded["game/dkjson"]={encode=function()return "{}"end}
package.loaded["bots/scripted_teacher_v9"]={source_sha256="test",new=function()return {}end,decide=function(s,m)return {action=1,buy=0,item=0,x=100,y=100,target_id=0},{forecasts={},candidates=0}end}
package.loaded["bots/agent"]={
 infrastructure=function()end,resolve=function()return bot end,
 execute=function(b,st,a) orders[#orders+1]={actual=math.floor(clock*30+.5),planned=a.execute_tick,observed=a.observed_tick};st.execution={accepted=true}end,
 snapshot=function()
  snapshots=snapshots+1
  local items={};for i=1,8 do items[i]={inventory=0,castable=false}end
  return {self={id=1,type="hero",team=3,visible=true,x=0,y=0,health=580,health_max=580,attack_damage=50,attack_interval=1.6,attack_point=.5,projectile_speed=1200,move_speed=305,attack_range=525},time=-10,team=3,creeps={},towers={},trees={},projectiles={},items=items,legal={actions={true,true},buys={}}}
 end}
''')
  self.live=self.lua.execute((ROOT/'bots/scripted_teacher_v9_live.lua').read_text())
 def tick(self,n):
  self.lua.globals().clock=(3000+n)/30;self.live.Think()
 def test_command_waits_seven_real_ticks_and_duplicate_callback_does_not_execute_twice(self):
  for n in range(7):self.tick(n);self.live.Think()
  self.assertEqual(len(self.lua.globals().orders),0)
  self.tick(7);self.live.Think();orders=self.lua.globals().orders
  self.assertEqual(len(orders),1);self.assertEqual(orders[1].actual-orders[1].observed,7);self.assertEqual(orders[1].planned,orders[1].actual)
 def test_late_callback_never_executes_early(self):
  self.tick(0);self.tick(10);order=self.lua.globals().orders[1]
  self.assertEqual(order.actual-order.observed,10);self.assertEqual(order.planned-order.observed,7)
 def test_human_and_other_team_are_never_controlled(self):
  self.lua.globals().player_is_bot=False;self.tick(0);self.assertEqual(self.lua.globals().snapshots,0)
  self.lua.globals().player_is_bot=True;self.lua.globals().team=2;self.tick(4);self.assertEqual(self.lua.globals().snapshots,0)
 def test_local_lobby_bundle_can_control_either_side_without_an_external_gate(self):
  self.lua.execute('package.loaded["bots/settings"].teacher_team=nil;team=2')
  for n in range(8):self.tick(n)
  self.assertEqual(len(self.lua.globals().orders),1)
  self.assertEqual(self.lua.globals().orders[1].actual-self.lua.globals().orders[1].observed,7)
 def test_native_lobby_waits_for_human_sf_and_then_runs_without_a_file_gate(self):
  self.lua.execute('''package.loaded["bots/settings"].human_team=2;package.loaded["bots/settings"].native_lobby=true
function GetTeamPlayers()return {0}end
function GetSelectedHeroName()return human_hero end
function IsPlayerBot(pid)return pid~=0 end
human_hero=""''')
  self.tick(0);self.assertEqual(self.lua.globals().snapshots,0)
  self.lua.globals().human_hero='npc_dota_hero_nevermore'
  for n in range(4,12):self.tick(n)
  self.assertEqual(len(self.lua.globals().orders),1)

if __name__=='__main__':unittest.main()
