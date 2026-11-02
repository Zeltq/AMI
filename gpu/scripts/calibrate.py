"""Capture public geometry and runtime stats from the installed Dota, in an isolated worker."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from sf_duel.live.game import Game,DEFAULT_GAME
from vpk import Vpk

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--worker',type=int,default=19);a=p.parse_args()
    run=Path(a.run).resolve();run.mkdir(parents=True,exist_ok=False);bots=run/'bots';bots.mkdir()
    shutil.copy2(ROOT/'bots/hero_selection.lua',bots/'hero_selection.lua')
    shutil.copy2(ROOT/'gpu/bots/calibrate.lua',bots/'bot_generic.lua')
    (bots/'ability_item_usage_generic.lua').write_text('function AbilityUsageThink() end\nfunction AbilityLevelUpThink() end\nfunction CourierUsageThink() end\nfunction ItemUsageThink() end\nfunction BuybackUsageThink() end\n')
    (bots/'item_purchase_generic.lua').write_text('function ItemPurchaseThink() end\n')
    content=ROOT/f'workers/{a.worker:02d}/game/dota'
    grid=np.zeros((512*512,2),np.float32);seen=np.zeros(len(grid),bool);trees=[];heroes={};towers={};bases={};creeps={}
    with Game(run,'gpu-calibration',4,15,bot_root=bots,content_root=content,game=content.parents[1],game_port=27900+a.worker,engine_threads=2) as game:
        deadline=time.monotonic()+180
        offset=0;tail=''
        while time.monotonic()<deadline:
            with game.log.open(errors='replace') as f:f.seek(offset);text=tail+f.read();offset=f.tell()
            lines=text.split('\n');tail=lines.pop()
            for line in lines:
                if 'SF_GPU_ERROR|' in line:raise RuntimeError(line)
                for tag in ('GRID','TREES','HERO','TOWER','BASE','CREEP'):
                    marker=f'SF_GPU_{tag}|'
                    if marker not in line:continue
                    value=json.loads(line.split(marker,1)[1])
                    if tag=='GRID':
                        start=value['first'];rows=value['rows'];end=start+len(rows)
                        if seen[start:end].any():raise RuntimeError('Duplicate geometry chunk')
                        grid[start:end]=rows;seen[start:end]=True
                    elif tag=='TREES':trees.extend(value)
                    elif tag=='HERO':heroes[str(value['team'])]=value
                    elif tag=='TOWER':towers[value['name']]=value
                    elif tag=='BASE':bases[str(value['team'])]=value
                    elif tag=='CREEP':creeps.setdefault(value['name'],value)
            if seen.all() and len(heroes)==2 and len(towers)==4 and len(creeps)>=4:break
            time.sleep(.05)
        else:raise TimeoutError(f'Calibration incomplete: {seen.sum()}/{len(seen)} cells, {len(towers)} towers, {len(creeps)} creeps')
    grid.tofile(run/'geometry.f32')
    vpk=Vpk(DEFAULT_GAME/'game/dota/pak01_dir.vpk');raw=run/'rules';raw.mkdir()
    sources={}
    for name in ('npc_heroes.txt','npc_units.txt','npc_abilities.txt','items.txt','heroes/npc_dota_hero_nevermore.txt'):
        data=vpk.read('scripts/npc/'+name);target=raw/Path(name).name;target.write_bytes(data)
        sources[name]=hashlib.sha256(data).hexdigest()
    report={'captured_utc':datetime.now(timezone.utc).isoformat(),'build':(DEFAULT_GAME/'game/dota/steam.inf').read_text(),
        'grid':{'file':'geometry.f32','shape':[512,512,2],'cell_size':32,'origin':[-8192,-8192],'complete':bool(seen.all())},
        'trees':trees,'heroes':heroes,'towers':towers,'bases':bases,'creeps':creeps,'rules_sha256':sources,
        'behavior_calibrated':False,'training_eligible':False}
    (run/'calibration.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({'complete_geometry':bool(seen.all()),'trees':len(trees),'towers':len(towers),'creeps':list(creeps),'output':str(run)}),flush=True)
if __name__=='__main__':main()
