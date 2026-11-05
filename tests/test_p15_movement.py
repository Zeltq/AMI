"""P15 native diagnostic executor: real Lua with mocked Bot API."""
import sys
from pathlib import Path
import unittest

from lupa import LuaRuntime

ROOT = Path(__file__).resolve().parents[1]


class DirectExecutorTest(unittest.TestCase):
    def runtime(self, direction=0, x=80, y=-816):
        lua = LuaRuntime(unpack_returned_tuples=True)
        lua.execute('''
          tick=0; calls={}; orders={}; position={x=80,y=-816,z=0}
          package.preload['game/dkjson']=function() return {encode=function(v) return '' end} end
          package.preload['bots/p15_config']=function() return {
            cases={{direction=0},{direction=9}},case_ticks=180,first_decision=30,decision_ticks=6,offset_ticks=6} end
          package.preload['bots/p15_clock']=function() return {start_tick=0} end
          function GameTime() return tick/30 end
          function GetTeam() return 2 end
          function Vector(x,y,z) return {x=x,y=y,z=z} end
          function print(s) end
          bot={GetUnitName=function() return 'npc_dota_hero_nevermore' end,
            GetLocation=function() return position end,GetFacing=function() return 0 end,
            GetCurrentMovementSpeed=function() return 305 end,GetCurrentActionType=function() return 3 end,
            Action_ClearActions=function() end,
            Action_MoveDirectly=function(self,v) calls[#calls+1]={tick=tick,x=v.x,y=v.y} end}
          function GetBot() return bot end
        ''')
        lua.globals().position.x = x
        lua.globals().position.y = y
        lua.execute(f"require('bots/p15_config').cases[1].direction={direction}")
        lua.execute((ROOT / 'gpu/bots/p15_direct_actor.lua').read_text())
        return lua

    def test_every_six_ticks_delayed_seven_and_duplicate_callback(self):
        lua = self.runtime()
        for tick in range(180):
            lua.globals().tick = tick
            lua.globals().Think()
            lua.globals().Think()
        calls = list(lua.globals().calls.values())
        self.assertEqual([c.tick for c in calls], list(range(37, 158, 6)))
        self.assertTrue(all(c.x == 2080 and c.y == -816 for c in calls))

    def test_all_36_absolute_angles_and_map_clamp(self):
        import math
        for k in range(36):
            with self.subTest(direction=k):
                lua = self.runtime(k, 8100, -8100)
                for tick in range(38):
                    lua.globals().tick = tick
                    lua.globals().Think()
                call = lua.globals().calls[1]
                self.assertAlmostEqual(call.x, max(-8192, min(8192, 8100+2000*math.cos(k*math.pi/18))))
                self.assertAlmostEqual(call.y, max(-8192, min(8192, -8100+2000*math.sin(k*math.pi/18))))

    def test_destination_uses_observed_position_before_latency(self):
        lua = self.runtime()
        for tick in range(38):
            lua.globals().tick = tick
            lua.globals().position.x = tick
            lua.globals().Think()
        self.assertEqual(lua.globals().calls[1].x, 2030)


if __name__ == '__main__':
    unittest.main()
