import importlib.util,json,shutil,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
try:
 from lupa.lua51 import LuaRuntime
except ImportError:
 LuaRuntime=None
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('prepare_teacher_lobby',ROOT/'scripts/prepare_human_teacher.py')
prepare=importlib.util.module_from_spec(spec);spec.loader.exec_module(prepare)


@unittest.skipIf(LuaRuntime is None,'CPU Lua 5.1 runtime required (lupa)')
class TeacherLobbyBundleTests(unittest.TestCase):
 def test_native_bundle_selects_own_bot_sf_and_never_selects_a_human_or_opponent(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);(root/'bots').mkdir();engine=root/'runs/gpu/dota-calibration-003/engine.json';engine.parent.mkdir(parents=True)
   engine.write_text(json.dumps({'combat':[],'prototypes':[[0]*17 for _ in range(9)],'hero':[0]*26}))
   (root/'gpu/src').mkdir(parents=True);shutil.copy2(ROOT/'gpu/src/scripted_teacher.rs',root/'gpu/src/scripted_teacher.rs')
   for name in ['agent.lua','scripted_teacher_v9.lua','scripted_teacher_v9_live.lua','teacher_v9_duel_setup.lua']:shutil.copy2(ROOT/'bots'/name,root/'bots'/name)
   with patch.object(prepare,'ROOT',root):output=prepare.prepare(root/'package',game=root/'fake-game',lobby=True)
   self.assertFalse((output/'runtime').exists());self.assertTrue((output/'Open-Dota.cmd').exists())
   lua=LuaRuntime(unpack_returned_tuples=True);settings=lua.execute((output/'bots/settings.lua').read_text())
   self.assertEqual(settings.human_team,2);self.assertEqual(settings.teacher_team,3);self.assertTrue(settings.native_lobby)
   lua.globals().package.loaded['bots/settings']=settings
   lua.execute('''team=3;picks={};names={}
function GetTeam() return team end
function GetTeamPlayers(t) assert(t==team,"opponent team queried");return team==2 and {0,1} or {5} end
function IsPlayerBot(pid) return pid~=0 end
function SelectHero(pid,hero) picks[#picks+1]={pid=pid,hero=hero} end
''')
   lua.execute((output/'bots/hero_selection.lua').read_text());lua.globals().Think();lua.globals().Think()
   picks=lua.globals().picks;self.assertEqual(len(picks),1);self.assertEqual((picks[1].pid,picks[1].hero),(5,'npc_dota_hero_nevermore'))
   lua.globals().team=2;lua.globals().Think();self.assertEqual(len(picks),2);self.assertEqual((picks[2].pid,picks[2].hero),(1,'npc_dota_hero_lina'))


if __name__=='__main__':unittest.main()
