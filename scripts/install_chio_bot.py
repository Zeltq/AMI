"""Prepare CHiO Local Dev Scripts; installation requires explicit --install."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'gpu/scripts')]
from install_play_bot import inventory,backup_contents,replace_contents,restore,DEFAULT_BOTS,bundle as legacy_bundle
from install_teacher_lobby import no_dota
from p17_matches import bundle as dedicated_bundle


def prepare(dest,human_team):
    dedicated_bundle(dest,5-human_team)
    # Reuse the tested human-aware hero selector. Its temporary native bundle is
    # only a source artifact and is never installed into the user's game.
    selection=dest.parent/(dest.name+'-selection')
    legacy_bundle(selection,human_team)
    shutil.copy2(selection/'hero_selection.lua',dest/'hero_selection.lua')
    shutil.copy2(ROOT/'bots/play_model_lobby.lua',dest/'play_model_lobby.lua')
    settings=dict(interval=.2,match='P17-human',observation_profile='ggd-v4',chio=True,
                  human_team=human_team,model_team=5-human_team,controller_type='chio',endpoint='')
    from sf_duel.live.bridge import literal
    (dest/'settings.lua').write_text('return '+literal(settings)+'\n')
    (dest/'chio_play_setup.lua').write_text("local ok,err=pcall(function()\nrequire('bots/play_model_lobby').run("+str(human_team)+")\nrequire('bots/p17_setup')\nend)\nif not ok then print('SF_P17_ERROR|'..tostring(err)) end\n")
    return dest


def install(target,prepared,artifacts,human_team):
    record=artifacts/'installation.json'
    if record.exists() and json.loads(record.read_text())['installed']:raise RuntimeError('Restore previous CHiO installation first')
    backup,before=backup_contents(target,artifacts)
    saved=dict(installed=True,path=str(target),resolved=str(target.resolve()),junction=target.is_junction(),
               backup=str(backup),before=before,human_team=human_team,model_team=5-human_team,
               prepared=str(prepared),installed_files=inventory(prepared))
    record.write_text(json.dumps(saved,indent=2))
    try:
        replace_contents(target,prepared)
        if inventory(target)!=saved['installed_files']:raise RuntimeError('Installed bytes differ')
    except BaseException:restore(artifacts);raise
    return saved


def main():
    p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group()
    g.add_argument('--install',action='store_true');g.add_argument('--restore',action='store_true')
    p.add_argument('--bots',type=Path,default=DEFAULT_BOTS);p.add_argument('--artifacts',type=Path,default=Path('C:/AMI/runs/p17/human'))
    p.add_argument('--human-team',type=int,choices=(2,3),default=2);a=p.parse_args()
    if a.restore:no_dota();result=restore(a.artifacts)
    else:
        a.artifacts.mkdir(parents=True,exist_ok=True)
        prepared=prepare(a.artifacts/('prepared-'+datetime.now().strftime('%Y%m%d-%H%M%S-%f')),a.human_team)
        result=dict(prepared=str(prepared),installed=False)
        if a.install:no_dota();result=install(a.bots,prepared,a.artifacts,a.human_team)
    print(json.dumps(result,indent=2))
    print('Restore: python scripts/install_chio_bot.py --restore --artifacts "'+str(a.artifacts)+'"')


if __name__=='__main__':main()
