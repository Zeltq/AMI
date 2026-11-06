"""Measure immutable ground height through a dedicated server; no graphical client."""
import argparse,hashlib,json,shutil,sys,time
from pathlib import Path
import numpy as np
from p15_probe import ROOT,Game,isolated_content,save
p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--port',type=int,default=27921);a=p.parse_args();run=a.run.absolute();run.mkdir(parents=True,exist_ok=False);bots=run/'bots';bots.mkdir()
shutil.copy2(ROOT/'bots/hero_selection.lua',bots/'hero_selection.lua')
for name,text in [('bot_generic.lua','function Think() end'),('item_purchase_generic.lua','function ItemPurchaseThink() end'),('ability_item_usage_generic.lua','function AbilityUsageThink() end\nfunction AbilityLevelUpThink() end\nfunction CourierUsageThink() end\nfunction ItemUsageThink() end\nfunction BuybackUsageThink() end')]: (bots/name).write_text(text+'\n')
(bots/'p15_height.lua').write_text("""local json=require('game/dkjson')
local hero=Entities:FindAllByClassname('npc_dota_hero_nevermore')[1]
local row=0
hero:SetContextThink('p15_ground',function()
 local ok,err=pcall(function()
  for chunk=0,3 do local values={}
   for i=0,127 do local x=chunk*128+i
    values[#values+1]=GetGroundPosition(Vector(-8192+(x+.5)*32,-8192+(row+.5)*32,0),hero).z
   end
   print('SF_P15_HEIGHT|'..json.encode({row=row,chunk=chunk,values=values}))
  end
 end)
 if not ok then print('SF_P15_HEIGHT_ERROR|'..tostring(err));return nil end
 row=row+1;if row==512 then print('SF_P15_HEIGHT_DONE|true');return nil end
 return 0.03
end,0)
""")
started=time.monotonic();height=np.full((512,512),np.nan,np.float32);chunks=set();content,_=isolated_content(run/'runtime',bots)
with Game(run,'P15-ground',4,30,bot_root=bots,content_root=content,game=content.parents[1],game_port=a.port,engine_threads=2) as game:
    save(run/'status.json',dict(stage='measuring-ground',pid=game.pid));game.command('script_reload_code bots/p15_height');offset=0;tail='';done=False
    while time.monotonic()-started<180:
        if game.process.poll() is not None:raise RuntimeError('Dedicated server exited')
        with game.log.open(errors='replace') as f:f.seek(offset);lines=tail+f.read();offset=f.tell()
        lines=lines.split('\n');tail=lines.pop()
        for line in lines:
            if 'SF_P15_HEIGHT_ERROR|' in line:raise RuntimeError(line)
            if 'SF_P15_HEIGHT|' in line:
                value=json.loads(line.split('SF_P15_HEIGHT|',1)[1]);row=value['row'];chunk=value['chunk'];height[row,chunk*128:(chunk+1)*128]=value['values'];chunks.add((row,chunk))
            if 'SF_P15_HEIGHT_DONE|' in line:done=True
        if done:break
        time.sleep(.1)
if not done or len(chunks)!=2048 or not np.isfinite(height).all():raise RuntimeError('Incomplete ground height capture')
height.astype('<f4').tofile(run/'ground32.f32');result=dict(samples=int(height.size),expected=512*512,chunks=len(chunks),minimum=float(height.min()),maximum=float(height.max()),
    sha256=hashlib.sha256((run/'ground32.f32').read_bytes()).hexdigest(),source='Dedicated-server GetGroundPosition at 32-unit cell centers',wall_minutes=(time.monotonic()-started)/60)
save(run/'report.json',result);save(run/'status.json',dict(stage='complete',**result));print(json.dumps(result))
