"""Capture server GridNav/height/objects into one CHiO-only snapshot artifact."""
import argparse,hashlib,json,shutil,time
from pathlib import Path
import numpy as np
from p15_probe import ROOT,Game,isolated_content,save

def parse(lines,state):
    for line in lines:
        if 'SF_P15_GRID_ERROR|' in line:raise RuntimeError(line)
        for tag,key in [('META','metadata'),('TREE','trees'),('BUILDING','buildings')]:
            marker='SF_P15_GRID_'+tag+'|'
            if marker in line:
                value=json.loads(line.split(marker,1)[1])
                if key=='metadata':state[key]=value
                else:state[key].append(value)
        if 'SF_P15_GRID_ROW|' in line:
            v=json.loads(line.split('SF_P15_GRID_ROW|',1)[1]);step=v['step'];row=v['row'];chunk=v['chunk']
            state['grids'][step][row,chunk*64:(chunk+1)*64]=np.asarray(v['values'],np.float32).reshape(64,3);state['chunks'].add((step,row,chunk))
        if 'SF_P15_GRID_DONE|' in line:state['done']=True

def capture(run,port,after_start=False):
    run.mkdir(parents=True,exist_ok=False);bots=run/'bots';bots.mkdir();shutil.copy2(ROOT/'bots/hero_selection.lua',bots/'hero_selection.lua');shutil.copy2(ROOT/'gpu/bots/p15_live_grid.lua',bots/'p15_live_grid.lua')
    if after_start:
        script=bots/'p15_live_grid.lua';code=script.read_text();script.write_text("local export_now=function()\n"+code+"\nend\nlocal hero=Entities:FindAllByClassname('npc_dota_hero_nevermore')[1]\nhero:SetContextThink('wait_grid_start',function()if GameRules:GetDOTATime(false,true)<0 then return .1 end export_now();return nil end,0)\n")
    for name,text in [('bot_generic.lua','function Think() end'),('item_purchase_generic.lua','function ItemPurchaseThink() end'),('ability_item_usage_generic.lua','function AbilityUsageThink() end\nfunction AbilityLevelUpThink() end\nfunction CourierUsageThink() end\nfunction ItemUsageThink() end\nfunction BuybackUsageThink() end')]: (bots/name).write_text(text+'\n')
    state={'grids':{step:np.full((16384//step,16384//step,3),np.nan,np.float32) for step in (64,32)},'metadata':{},'trees':[],'buildings':[],'chunks':set(),'done':False}
    started=time.monotonic();content,_=isolated_content(run/'runtime',bots)
    with Game(run,'P15-live-grid',4,30,bot_root=bots,content_root=content,game=content.parents[1],game_port=port,engine_threads=2) as game:
        save(run/'status.json',dict(stage='gridnav-export',dota_pid=game.pid));game.command('script_reload_code bots/p15_live_grid');offset=0;tail=''
        while time.monotonic()-started<180:
            if game.process.poll() is not None:raise RuntimeError('Dedicated server exited')
            with game.log.open(errors='replace') as f:f.seek(offset);lines=tail+f.read();offset=f.tell()
            lines=lines.split('\n');tail=lines.pop();parse(lines,state)
            if state['done']:break
            time.sleep(.1)
    expected=sum((16384//s)**2//64 for s in (64,32))
    if not state['done'] or len(state['chunks'])!=expected or any(not np.isfinite(v).all() for v in state['grids'].values()):raise RuntimeError('Incomplete GridNav snapshot')
    if not state['trees'] or not state['buildings']:raise RuntimeError('Missing live objects')
    meta={k:state[k] for k in ('metadata','trees','buildings')};meta.update(surface='dedicated server; same server Lua bridge, Local Host not tested',bounds=[-8192,8192],sampling='cell centers',channels=['traversable','blocked','ground_height'],lua_sha256=hashlib.sha256((bots/'p15_live_grid.lua').read_bytes()).hexdigest())
    np.savez_compressed(run/'live-grid.npz',grid64=state['grids'][64],grid32=state['grids'][32],metadata=np.array(json.dumps(meta)))
    save(run/'objects.json',meta)
    report={'file':str(run/'live-grid.npz'),'sha256':hashlib.sha256((run/'live-grid.npz').read_bytes()).hexdigest(),'chunks':len(state['chunks']),'expected_chunks':expected,'trees':len(state['trees']),'buildings':len(state['buildings']),'metadata':state['metadata'],'grids':{str(s):{'cells':v.shape[0]*v.shape[1],'blocked':int((v[:,:,1]>.5).sum()),'traversable':int((v[:,:,0]>.5).sum())} for s,v in state['grids'].items()},'wall_minutes':(time.monotonic()-started)/60,'surface':meta['surface']}
    save(run/'report.json',report);save(run/'status.json',dict(stage='complete',**report));return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('--port',type=int,default=27923);p.add_argument('--after-start',action='store_true');a=p.parse_args();print(json.dumps(capture(a.run.absolute(),a.port,a.after_start)))
