import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from install_play_bot import install,restore,inventory,bundle
from play_vs_model import summarize


class PlayTests(unittest.TestCase):
    def test_status_replace_retries_a_transient_windows_reader(self):
        from unittest.mock import patch
        sys.path.insert(0,str(ROOT/'gpu/scripts'))
        from evaluate_dota_teacher import save
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'status.json';original=Path.replace;calls=[]
            def replace(source,target):
                calls.append(target)
                if len(calls)==1:raise PermissionError('temporary reader')
                return original(source,target)
            with patch.object(Path,'replace',replace):save(path,{'stage':'running'})
            self.assertEqual(json.loads(path.read_text()),{'stage':'running'})
            self.assertEqual(len(calls),2)

    def test_lobby_selects_only_own_bots_and_one_opposing_sf(self):
        from lupa.lua51 import LuaRuntime
        for human in (2,3):
            with tempfile.TemporaryDirectory() as folder:
                root=Path(folder)/'bots';bundle(root,human)
                lua=LuaRuntime(unpack_returned_tuples=True)
                lua.globals().package.loaded['bots/settings']=lua.execute((root/'settings.lua').read_text())
                lua.execute('''team=2;human=2;picks={}
function GetTeam() return team end
function GetTeamPlayers(t) assert(t==team);return team==human and {1,0,2} or {5,6,7} end
function IsPlayerBot(pid) return pid~=0 end
function SelectHero(pid,name) picks[#picks+1]={pid=pid,name=name} end
''')
                lua.globals().human=human
                lua.execute((root/'hero_selection.lua').read_text())
                for team in (2,3):
                    lua.globals().team=team;lua.globals().Think();lua.globals().Think()
                picks=list(lua.globals().picks.values())
                self.assertEqual(len(picks),5)
                self.assertEqual(sum(p.name=='npc_dota_hero_nevermore' for p in picks),1)
                self.assertFalse(any(p.pid==0 for p in picks))

    def test_restore_recovers_nested_bytes_empty_directories_and_removes_new_files(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);target=root/'bots';target.mkdir()
            (target/'nested').mkdir();(target/'empty').mkdir()
            (target/'nested/original').write_bytes(bytes(range(256)))
            before=inventory(target)
            result=install(target,root/'artifacts',3)
            self.assertEqual(result['model_team'],2)
            (target/'new').write_text('command buffer')
            restore(root/'artifacts')
            self.assertEqual(before,inventory(target))
            self.assertTrue((target/'empty').is_dir())

    def test_first_death_censors_future_casts_and_unreached_marks(self):
        rows=[dict(kind='hero',team=t,time=300,lh=t,deny=1,alive=True,x=0,y=0) for t in (2,3)]
        rows += [dict(kind='cast',team=3,time=301,ability='nevermore_shadowraze1'),
                 dict(kind='hurt',team=3,time=301,target='npc_dota_hero_nevermore',ability='nevermore_shadowraze1'),
                 dict(kind='death',team=2,killer_team=3,killer='npc_dota_hero_nevermore',time=302),
                 dict(kind='cast',team=3,time=400,ability='nevermore_shadowraze2')]
        with tempfile.TemporaryDirectory() as folder:
            log=Path(folder)/'log';log.write_text(''.join('SF_PLAY_EVENT|'+json.dumps(r)+'\n' for r in rows))
            s=summarize(log,3)
        self.assertEqual(s['first_blood']['team'],2)
        self.assertEqual(s['sides']['3']['raze_casts'],1)
        self.assertEqual(s['sides']['3']['raze_hero_hits'],1)
        self.assertIsNone(s['sides']['3']['at_seconds']['600'])
        self.assertEqual(s['sides']['3']['at_seconds']['300']['lh'],3)

    def test_tower_death_is_not_a_hero_kill_or_first_blood(self):
        rows=[dict(kind='hero',team=t,time=100,lh=0,deny=0,kills=0,alive=True,x=0,y=0) for t in (2,3)]
        rows.append(dict(kind='death',team=2,killer_team=3,killer='npc_dota_badguys_tower1_mid',time=100.05))
        with tempfile.TemporaryDirectory() as folder:
            log=Path(folder)/'log';log.write_text(''.join('SF_PLAY_EVENT|'+json.dumps(r)+'\n' for r in rows))
            result=summarize(log,2)
        self.assertIsNone(result['first_blood'])
        self.assertEqual(result['first_death']['time'],100.05)
        self.assertEqual(result['sides']['3']['kills'],0)

    def test_timeout_excludes_later_death_cast_and_score(self):
        rows=[dict(kind='hero',team=t,time=600,lh=0,deny=0,kills=0,alive=True,x=0,y=0) for t in (2,3)]
        rows.extend([dict(kind='death',team=2,killer_team=3,killer='npc_dota_hero_nevermore',time=600.02),
                     dict(kind='cast',team=3,time=600.02,ability='nevermore_shadowraze1'),
                     dict(kind='hero',team=3,time=600.02,lh=0,deny=0,kills=1,alive=True,x=0,y=0)])
        with tempfile.TemporaryDirectory() as folder:
            log=Path(folder)/'log';log.write_text(''.join('SF_PLAY_EVENT|'+json.dumps(r)+'\n' for r in rows))
            result=summarize(log,2)
        self.assertIsNone(result['first_death'])
        self.assertEqual(result['sides']['3']['kills'],0)
        self.assertEqual(result['sides']['3']['raze_casts'],0)


if __name__=='__main__':unittest.main()
