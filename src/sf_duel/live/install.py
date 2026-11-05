"""Separate writable game content per server, sharing immutable Steam assets."""
import json
import os
from pathlib import Path
import shutil
import subprocess
from .game import DEFAULT_GAME


def isolated_content(root,bot_source,game=DEFAULT_GAME):
    # Resolve only after creation: concurrent creation of missing parents on
    # Windows can otherwise leave a \\?\ prefix that Source 2 cannot launch from.
    root=Path(root).absolute();root.mkdir(parents=True,exist_ok=True)
    root=root.resolve(strict=True);game=Path(game).resolve()
    source=game/'game/dota';content=root/'game/dota';runtime=root/'teacher_bots'
    content.mkdir(parents=True,exist_ok=True)
    links=[]
    def mirror(src,dst,skip):
        dst.mkdir(parents=True,exist_ok=True)
        for path in src.iterdir():
            if path.name in skip:continue
            target=dst/path.name
            if target.exists():
                if not os.path.samefile(path,target):raise RuntimeError(f'Unexpected isolation entry: {target}')
            elif path.is_dir():links.append((target,path))
            else:os.link(path,target)
    # Dota derives its script directory from the executable location, ignoring
    # -game for that purpose. Give each process a real executable path; all
    # binaries are NTFS hardlinks, not duplicate byte copies.
    mirror(game/'game',root/'game',{'dota','bin'})
    mirror(game/'game/bin',root/'game/bin',{'win64'})
    mirror(game/'game/bin/win64',root/'game/bin/win64',set())
    mirror(source,content,{'scripts','cfg','replays','logs','screenshots','gameinfo.gi','console.log'})
    # With the executable under this root, normal relative paths resolve into
    # the private game. Keep an independent, unmodified gameinfo copy.
    shutil.copy2(source/'gameinfo.gi',content/'gameinfo.gi')
    mirror(source/'scripts',content/'scripts',{'vscripts'})
    vscripts=source/'scripts/vscripts'
    backups={p.name for p in vscripts.iterdir() if p.name.lower().startswith('bots.') and p.is_dir()}
    mirror(vscripts,content/'scripts/vscripts',{'bots','bots.sf_duel_backup'}|backups)
    for name in ('cfg','replays','logs'): (content/name).mkdir(exist_ok=True)
    shutil.copytree(source/'cfg',content/'cfg',dirs_exist_ok=True)
    if links:
        def quote(s):return "'"+str(s).replace("'","''")+"'"
        pairs=';'.join(',@('+quote(a)+','+quote(b)+')' for a,b in links)
        try:
            subprocess.run(['powershell','-NoProfile','-Command',
                '$ErrorActionPreference="Stop";$taskLinks=@('+pairs+');foreach($pair in $taskLinks){New-Item -ItemType Junction -Path $pair[0] -Target $pair[1] | Out-Null}'],
                check=True,capture_output=True,text=True)
        except subprocess.CalledProcessError as error:
            raise RuntimeError(f'Junction creation failed (exit {error.returncode}): {error.stderr}') from error
    shutil.copytree(bot_source,runtime,dirs_exist_ok=True)
    (root/'isolation.json').write_text(json.dumps({'assets':str(source),'content':str(content),
        'bots':str(runtime),'asset_copy':'NTFS hardlinks and junctions; private scripts/cfg/replays/logs'},indent=2))
    return content,runtime
