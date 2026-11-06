import copy
import math
from pathlib import Path
import sys
import unittest
import torch
import numpy as np
from fixtures import snapshot
from sf_duel.chio_observations import Observer,command
from sf_duel.chio_cpu import Session
from sf_duel.chio_model import Policy


def scene(tick=0):
    s=snapshot();s['tick']=tick;s['deaths']=0
    for u in [s['self'],s['enemy'],*s['towers']]:u.update(facing=0,hull=24,attack_range=500)
    return s


class P17(unittest.TestCase):
    def test_history_fog_gap_and_generation(self):
        o=Observer();s=scene();b=o.encode(s)
        s['tick']=6;s['enemy']['visible']=False;b=o.encode(s)
        row=b['hero_combat'][0,1,-50:]
        self.assertEqual(row[0],0);self.assertEqual(row[25],0)
        self.assertAlmostEqual(float(row[1]),500/600,places=6);self.assertEqual(row[26],1)
        s['tick']=18;s['enemy'].update(visible=True,generation=1,health=100);b=o.encode(s)
        row=b['hero_combat'][0,1,-50:]
        self.assertEqual(float(row[26:].sum()),5) # four geometry known plus angle
        self.assertAlmostEqual(float(row[0]),1/6,places=6)
        s['tick']=24;s.pop('enemy');b=o.encode(s)
        row=b['hero_combat'][0,1,-50:]
        self.assertEqual(row[25],0);self.assertEqual(row[26],1)
        self.assertAlmostEqual(float(row[1]),1/6,places=6)

    def test_geometry_and_pending_cast_mask(self):
        s=scene();s['enemy']['x']=700
        b=Observer().encode(s,pending_cast=8)
        self.assertEqual(torch.where(b['action_mask'][0])[0].tolist(),[1,8])
        row=b['hero_combat'][0,1,-50:]
        self.assertEqual(row[12:15].tolist(),[0,1,1]);self.assertEqual(row[15],0)
        self.assertEqual(row[16],0)
        self.assertEqual(b['items'].shape,(1,6,16));self.assertEqual(b['previous'][0,1],1)
        self.assertTrue((b['heroes'][0,:,[38,86]]==0).all())
        s['raze_rules']=[dict(range=0,radius=0) for _ in range(3)]
        unknown=Observer().encode(s)
        self.assertEqual(float(unknown['hero_combat'][0,1,-50:][37:40].sum()),0)
        self.assertEqual(float(unknown['hero_combat'][0,0,-50:][42:45].sum()),0)

    def test_target_sort_tree_tie_move_bounds(self):
        s=scene(60);s['self'].update(x=8100,y=8100)
        d=command(s,dict(action=0,direction=0),1)
        self.assertEqual((d['x'],d['execute_tick']),(8192,67))
        s['trees']=[dict(id=4,x=8099,y=8100),dict(id=2,x=8101,y=8100)]
        self.assertEqual(command(s,dict(action=12),2)['target'],2)

    def test_cpu_reset_and_fp16_boundary(self):
        class Fake:
            def decide(self,b,state,deterministic):
                self.input=b['x'];self.before=state
                return {'action':torch.tensor([1])},None,None,torch.tensor([0.]),('memory',)
        m=Fake();session=Session(m)
        resets=[]
        for i in range(130):
            _,_,r=session.decide({'x':torch.tensor([.1234567])},i*6,'a');resets.append(r)
        self.assertEqual([i for i,r in enumerate(resets) if r],[0,128])
        self.assertEqual(m.input.item(),torch.tensor(.1234567).half().float().item())
        session.decide({'x':torch.tensor([1.])},780,'a',death=True);self.assertIsNone(m.before)
        with self.assertRaises(ValueError):session.decide({'x':torch.tensor([1.])},780,'a')

    def test_all_lua_compiles(self):
        from lupa.lua51 import LuaRuntime
        lua=LuaRuntime();root=Path(__file__).resolve().parents[1]
        for name in ('bots/chio_executor.lua','bots/chio_economy.lua','gpu/bots/p17_bridge.lua','gpu/bots/p17_setup.lua'):
            lua.execute('assert(loadstring(...))',(root/name).read_text())

    def test_empty_entities_model_forward(self):
        torch.set_num_threads(1);model=Policy().eval();b=Observer().encode(scene())
        with torch.inference_mode():result=model.decide(b,deterministic=True)
        self.assertTrue(torch.isfinite(result[3]).all())

    def test_lua_continuation_after_cast_end_does_not_cast_again(self):
        from lupa.lua51 import LuaRuntime
        lua=LuaRuntime();root=Path(__file__).resolve().parents[1]
        lua.execute('''tick=7;casts=0;attacks=0;commands={}
GAME_STATE_PRE_GAME=1;GAME_STATE_GAME_IN_PROGRESS=2
BOT_ACTION_TYPE_ATTACK=3;ACTIVITY_ATTACK=10;ACTIVITY_ATTACK2=11
function GetGameState()return 1 end
function GameTime()return tick/30 end
function GetTeam()return 2 end
function print()end
target={IsNull=function()return false end,CanBeSeen=function()return true end,
 IsAlive=function()return true end,GetTeam=function()return 3 end}
bot={GetUnitName=function()return 'npc_dota_hero_nevermore' end,IsAlive=function()return true end,
 IsCastingAbility=function()return false end,IsUsingAbility=function()return false end,
 GetAbilityByName=function()return {IsFullyCastable=function()return true end} end,
 Action_UseAbility=function()casts=casts+1 end,Action_AttackUnit=function()attacks=attacks+1 end,
 GetAttackTarget=function()return nil end,GetAnimActivity=function()return 10 end,
 GetCurrentActionType=function()return 3 end}
function GetBot()return bot end
package.preload['bots/agent']=function()return {resolve=function()return target end}end
package.preload['bots/chio_economy']=function()return {}end
package.preload['bots/settings']=function()return {}end
package.preload['game/dkjson']=function()return {encode=function()return '{}'end}end
package.preload['bots/chio_commands_2']=function()return commands end
''')
        executor=lua.execute((root/'bots/chio_executor.lua').read_text())
        lua.execute('commands={{sequence=1,observed_tick=0,execute_tick=7,action=6}}');executor.Think()
        lua.execute('tick=13;commands={{sequence=2,observed_tick=6,execute_tick=13,action=6,continuation=true}}');executor.Think()
        self.assertEqual(lua.globals().casts,1)
        lua.execute('tick=19;commands={{sequence=3,observed_tick=12,execute_tick=19,action=5,target=1}}');executor.Think()
        lua.execute('tick=25;commands={{sequence=4,observed_tick=18,execute_tick=25,action=5,target=1}}');executor.Think()
        self.assertEqual(lua.globals().attacks,1)

    def test_economy_permanent_observation_includes_delivery(self):
        from lupa.lua51 import LuaRuntime
        lua=LuaRuntime();root=Path(__file__).resolve().parents[1]
        lua.execute('''owned={item_boots=1,item_tango=3,item_flask=1}
function DotaTime()return 100 end
function GameTime()return 100 end
function GetTeam()return 2 end
function GetCourier()return nil end
function GetItemStockCount()return 0 end
function print()end
package.preload['game/dkjson']=function()return {encode=function()return '{}'end}end
package.preload['bots/agent']=function()return {
 owned=function(_,name)return owned[name] or 0 end,infrastructure=function()end,
 missing_cost=function()return 100 end}end
bot={IsAlive=function()return true end,GetHealth=function()return 100 end,GetMaxHealth=function()return 100 end,
 GetMana=function()return 100 end,GetMaxMana=function()return 100 end,GetLevel=function()return 1 end,
 GetGold=function()return 0 end,GetModifierByName=function()return -1 end,
 GetPlayerID=function()return 0 end,GetItemInSlot=function()return nil end,GetCurrentMovementSpeed=function()return 305 end}
state={permanent=0}
''')
        economy=lua.execute((root/'bots/chio_economy.lua').read_text())
        economy.Think(lua.globals().bot,lua.globals().state);self.assertEqual(lua.globals().state.permanent,1)
        lua.execute('owned.item_bracer=1');economy.Think(lua.globals().bot,lua.globals().state)
        self.assertEqual(lua.globals().state.permanent,2)


if __name__=='__main__':unittest.main()
