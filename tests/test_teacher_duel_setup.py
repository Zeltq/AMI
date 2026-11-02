import unittest
from pathlib import Path
try:
 from lupa.lua51 import LuaRuntime
except ImportError:
 LuaRuntime=None
ROOT=Path(__file__).resolve().parents[1]


@unittest.skipIf(LuaRuntime is None,'CPU Lua 5.1 runtime required (lupa)')
class TeacherDuelSetupTests(unittest.TestCase):
 def setUp(self):
  self.lua=LuaRuntime(unpack_returned_tuples=True)
  self.lua.execute('''players={};reset={};disconnected={}
for i=0,9 do
 local pid=i;local team=i<5 and 2 or 3
 local name=i==0 and "human" or ("SF_DUEL_"..team.."_"..(i<5 and i or i-4))
 local unit={};function unit:GetTeamNumber()return team end
 function unit:GetUnitName()return (pid==0 or pid==1 or pid==5) and "npc_dota_hero_nevermore" or "npc_dota_hero_lina" end
 players[i]={fake=i~=0,name=name,hero=unit}
end
PlayerResource={}
function PlayerResource:IsValidPlayerID(i)return players[i]~=nil end
function PlayerResource:IsFakeClient(i)return players[i].fake end
function PlayerResource:GetPlayerName(i)return players[i].name end
function PlayerResource:GetSelectedHeroEntity(i)return players[i].hero end
GameRules={};function GameRules:ResetPlayer(i)reset[#reset+1]=i end
function DisconnectClient(i)disconnected[#disconnected+1]=i;players[i]=nil end
''')
  self.setup=self.lua.execute((ROOT/'bots/teacher_v9_duel_setup.lua').read_text())
 def test_removes_extra_bot_sf_and_linas_while_preserving_human_and_teacher(self):
  result=self.setup.run(2);self.assertEqual(result.teacher_player,5);self.assertEqual(result.removed,8)
  self.assertIsNotNone(self.lua.globals().players[0]);self.assertIsNotNone(self.lua.globals().players[5])
  self.assertEqual([self.lua.globals().reset[i] for i in range(1,9)],[1,2,3,4,6,7,8,9])
  self.assertEqual(self.setup.run(2).removed,0)
 def test_missing_teacher_changes_nothing(self):
  self.lua.execute('players[5]=nil')
  with self.assertRaises(Exception):self.setup.run(2)
  self.assertEqual(len(self.lua.globals().reset),0)
 def test_additional_human_changes_nothing(self):
  self.lua.execute('players[2].fake=false')
  with self.assertRaises(Exception):self.setup.run(2)
  self.assertEqual(len(self.lua.globals().reset),0)


if __name__=='__main__':unittest.main()
