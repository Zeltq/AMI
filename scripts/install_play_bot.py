"""Install ZOg Local Dev Scripts by replacing contents, preserving the bots entry."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
from sf_duel.live.game import DEFAULT_GAME
from sf_duel.live.bridge import literal
from prepare_native_teacher import prepare
from install_teacher_lobby import no_dota

ARTIFACTS = Path('C:/AMI/runs/p10')
DEFAULT_BOTS = DEFAULT_GAME/'game/dota/scripts/vscripts/bots'


def inventory(root):
    result = {}
    for p in sorted(root.rglob('*')):
        if p.is_symlink() or p.is_junction():
            raise ValueError(f'Nested link requires manual preservation: {p}')
        result[p.relative_to(root).as_posix()] = (
            hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else 'directory')
    return result


def replace_contents(target, source):
    target = target.resolve(strict=True)
    source = source.resolve(strict=True)
    if target == source or target in source.parents or source in target.parents:
        raise ValueError('Backup and installation must be disjoint')
    inventory(target)  # Reject nested links before deleting any entry.
    entries = list(target.iterdir())
    for entry in entries:
        if entry.resolve().parent != target:
            raise ValueError(f'Entry escapes installation: {entry}')
    for entry in entries:
        if entry.is_dir():
            shutil.rmtree(entry)
        else:
            entry.unlink()
    shutil.copytree(source, target, dirs_exist_ok=True)


def backup_contents(target, artifacts=ARTIFACTS):
    artifacts.mkdir(parents=True, exist_ok=True)
    backup = artifacts/('bots-backup-'+datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
    before = inventory(target)
    shutil.copytree(target, backup)
    if inventory(backup) != before:
        raise RuntimeError('Backup differs; installation was not modified')
    return backup, before


def bundle(destination, human_team):
    prepare(destination)
    settings = dict(interval=4/30,match='P10-human',observation_profile='ggd-v4',model_team=5-human_team,
                    human_team=human_team,controller_type='native',endpoint='')
    (destination/'settings.lua').write_text('return '+literal(settings)+'\n')
    callback = '''local settings=require('bots/settings')
local controller=require('bots/native_executor')
function Think()
 if GetTeam()~=settings.model_team then return end
 local ok,err=pcall(controller.Think)
 if not ok then print('SF_NATIVE_ERROR|'..tostring(err)) end
end
'''
    (destination/'bot_generic.lua').write_text(callback)
    (destination/'item_purchase_generic.lua').write_text(callback.replace('function Think()', 'function ItemPurchaseThink()'))
    (destination/'hero_selection.lua').write_text('''local settings=require('bots/settings')
local selected={}
function GetBotNames()
 local names={};for i=1,5 do names[i]='SF_DUEL_'..GetTeam()..'_'..i end;return names
end
function Think()
 local first=true
 for _,pid in ipairs(GetTeamPlayers(GetTeam())) do
  if IsPlayerBot(pid) then
   local sf=GetTeam()==settings.model_team and first;first=false
   if not selected[pid] then SelectHero(pid,sf and 'npc_dota_hero_nevermore' or 'npc_dota_hero_lina');selected[pid]=true end
  end
 end
end
function UpdateLaneAssignments()
 local lanes={};for _,pid in ipairs(GetTeamPlayers(GetTeam())) do lanes[pid]=LANE_MID end;return lanes
end
''')
    # Reuse the lobby's names and first bot convention used by the cleanup script.
    for name in ('teacher_v9_duel_setup.lua','play_model_events.lua'):
        shutil.copy2(ROOT/'bots'/name,destination/name)
    (destination/'play_model_setup.lua').write_text('''local M={}
function M.run(human_team)
 local result=require('bots/teacher_v9_duel_setup').run(human_team)
 require('bots/play_model_events').start()
 print('SF_PLAY_DUEL_READY|'..human_team)
 return result
end
M.run(HUMAN_TEAM)
return M
'''.replace('HUMAN_TEAM',str(human_team)))


def install(target=DEFAULT_BOTS, artifacts=ARTIFACTS, human_team=2):
    target=Path(target);artifacts=Path(artifacts)
    record=artifacts/'installation.json'
    if record.exists() and json.loads(record.read_text())['installed']:
        raise RuntimeError('Already installed; restore before installing again')
    prepared=artifacts/('play-bots-'+datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
    bundle(prepared,human_team)
    backup, before=backup_contents(target,artifacts)
    saved=dict(installed=True,path=str(target),resolved=str(target.resolve()),junction=target.is_junction(),
               backup=str(backup),before=before,human_team=human_team,model_team=5-human_team,
               prepared=str(prepared),installed_files=inventory(prepared))
    record.write_text(json.dumps(saved,indent=2))  # Recovery exists before the first mutation.
    print(f'Backup verified: {backup}',flush=True)
    print(f'Restore: python scripts/install_play_bot.py --restore --artifacts "{artifacts}"',flush=True)
    try:
        replace_contents(target,prepared)
        if inventory(target)!=saved['installed_files']:
            raise RuntimeError('Installed files differ from bundle')
    except BaseException:
        restore(artifacts)
        raise
    return saved


def restore(artifacts=ARTIFACTS):
    record=Path(artifacts)/'installation.json';saved=json.loads(record.read_text())
    if not saved['installed']:
        return saved
    target=Path(saved['path']);backup=Path(saved['backup'])
    if str(target.resolve())!=saved['resolved'] or target.is_junction()!=saved['junction']:
        raise RuntimeError('Installation entry changed; refusing restore')
    if inventory(backup)!=saved['before']:
        raise RuntimeError('Backup changed; refusing restore')
    # Preserve the complete current contents before restore, too.
    recovery,_=backup_contents(target,Path(artifacts))
    saved['pre_restore_backup']=str(recovery)
    record.write_text(json.dumps(saved,indent=2))
    replace_contents(target,backup)
    if inventory(target)!=saved['before']:
        raise RuntimeError('Restored bytes differ')
    saved['installed']=False
    saved['restore_verified']=True
    record.write_text(json.dumps(saved,indent=2))
    return saved


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--restore',action='store_true')
    p.add_argument('--bots',type=Path,default=DEFAULT_BOTS)
    p.add_argument('--artifacts',type=Path,default=ARTIFACTS)
    p.add_argument('--human-team',type=int,choices=(2,3),default=2)
    a=p.parse_args();no_dota()
    result=restore(a.artifacts) if a.restore else install(a.bots,a.artifacts,a.human_team)
    print(json.dumps({k:result[k] for k in ('installed','path','backup','human_team','model_team')},indent=2))


if __name__=='__main__':
    main()
