"""Reproducible combat timing captures in the real game. Never eligible as demonstrations."""
import argparse,json,shutil,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from sf_duel.live.game import Game

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--worker',type=int,default=19);p.add_argument('--suite',choices=['combat','movement','turn-cast','attack-turn'],default='combat');a=p.parse_args()
    run=Path(a.run).resolve();run.mkdir(parents=True,exist_ok=False);bots=run/'bots';bots.mkdir()
    shutil.copy2(ROOT/'bots/hero_selection.lua',bots/'hero_selection.lua');shutil.copy2(ROOT/'gpu/bots/scenario.lua',bots/'bot_generic.lua')
    shutil.copy2(ROOT/'bots/agent.lua',bots/'agent.lua')
    (bots/'gpu_commands.lua').write_text('return {}\n');(bots/'item_purchase_generic.lua').write_text('function ItemPurchaseThink() end\n')
    (bots/'ability_item_usage_generic.lua').write_text('function AbilityUsageThink() end\nfunction AbilityLevelUpThink() end\nfunction CourierUsageThink() end\nfunction ItemUsageThink() end\nfunction BuybackUsageThink() end\n')
    content=ROOT/f'workers/{a.worker:02d}/game/dota';frames=[];orders=[];setups=[];offset=0;tail='';serial=0;setup_serial=0
    grid=np.fromfile(ROOT/'runs/gpu/dota-calibration-003/geometry.f32',np.float32).reshape(512,512,2)
    arena=None
    for y in range(230,280):
        for x in range(230,280):
            strip=grid[y,x:x+24]
            if len(strip)==24 and (strip[:,1]>0).all() and (strip[:,0]==strip[0,0]).all():
                arena=(-8192+(x+.5)*32,-8192+(y+.5)*32);break
        if arena:break
    if arena is None:raise RuntimeError('No calibrated level arena')
    with Game(run,'gpu-combat',1,30,bot_root=bots,content_root=content,game=content.parents[1],game_port=27900+a.worker,engine_threads=2) as game:
        def read():
            nonlocal offset,tail
            with game.log.open(errors='replace') as f:f.seek(offset);text=tail+f.read();offset=f.tell()
            lines=text.split('\n');tail=lines.pop()
            for line in lines:
                if 'SF_GPU_ERROR|' in line:raise RuntimeError(line)
                for tag,target in [('FRAME',frames),('ORDER',orders),('SETUP',setups)]:
                    marker=f'SF_GPU_{tag}|'
                    if marker in line:target.append(json.loads(line.split(marker,1)[1]))
            return text
        def reset(distance=384,facing=0):
            nonlocal setup_serial
            x,y=arena
            setup_serial+=1
            body=f'return {{setup={{serial={setup_serial},x0={x},y0={y},x1={x+distance},y1={y}}}}}\n'
            temporary=bots/'gpu_commands.tmp';temporary.write_text(body);temporary.replace(bots/'gpu_commands.lua')
            deadline=time.monotonic()+70
            while time.monotonic()<deadline:
                read()
                if len({s['team'] for s in setups if s['serial']==setup_serial})==2:break
                time.sleep(.02)
            else:raise TimeoutError('Both heroes did not reach the measured arena')
            end=time.monotonic()+1.3
            while time.monotonic()<end:read();time.sleep(.02)
        results=[]
        combat=[('attack',3,None,384),('cancel_before_release',3,6,384),('cancel_after_release',3,20,384),('reissue_same_target',3,-6,384),('reissue_near_raze',6,-6,224),('raze',8,None,704),('cancel_raze',7,6,448)]
        cases=[(*case,None)for case in combat]if a.suite=='combat'else[
            ('move_straight',2,None,384,(256,0)),('turn_90',2,None,384,(0,256)),
            ('turn_180',2,None,384,(-256,0)),('reissue_move_direct',2,-6,384,(256,0)),('stop_moving',2,6,384,(256,0)),
            ('direct_45',2,None,384,(256,256)),('direct_135',2,None,384,(-256,256)),
            ('path_90',1,None,384,(0,256)),('path_180',1,None,384,(-256,0)),
            ('stop_turn_90',2,1,384,(0,256)),('stop_turn_180',2,1,384,(-256,0))]
        if a.suite=='turn-cast':cases=[('turn_one_tick_far_raze',2,1,700,(0,256)),('turn_two_ticks_far_raze',2,2,700,(0,256)),('baseline_far_raze',8,None,700,None)]
        if a.suite=='attack-turn':cases=[('attack_turn_side',3,None,384,None),('attack_turn_back',3,None,384,None)]
        for name,action,cancel,dist,move in cases:
            reset(dist)
            if a.suite=='attack-turn':
                read();own=max((f for f in frames if f['team']==2),key=lambda f:f['tick']);dx,dy=(0,64)if name=='attack_turn_side'else(-64,0)
                serial+=1;tick=max(f['tick']for f in frames)+12
                temporary=bots/'gpu_commands.tmp';temporary.write_text(f'return {{{{team=2,serial={serial},tick={tick},action=2,x={own["x"]+dx},y={own["y"]+dy}}}}}\n');temporary.replace(bots/'gpu_commands.lua')
                deadline=time.monotonic()+1.2
                while time.monotonic()<deadline:read();time.sleep(.02)
                serial+=1;tick=max(f['tick']for f in frames)+6
                temporary.write_text(f'return {{{{team=2,serial={serial},tick={tick},action=4}}}}\n');temporary.replace(bots/'gpu_commands.lua')
                deadline=time.monotonic()+.6
                while time.monotonic()<deadline:read();time.sleep(.02)
            if a.suite=='turn-cast'and results:
                deadline=time.monotonic()+10
                while time.monotonic()<deadline:read();time.sleep(.02)
            read();start=len(frames);order_start=len(orders);latest=max(f['tick'] for f in frames);tick=latest+12;commands=[]
            serial+=1;commands.append({'team':2,'serial':serial,'tick':tick,'action':action})
            if cancel is not None:
                serial+=1;commands.append({'team':2,'serial':serial,'tick':tick+abs(cancel),'action':8 if a.suite=='turn-cast'else action if cancel<0 else 4})
            if move:
                for c in commands:
                    if c['action']in(1,2):c.update(x=arena[0]+move[0],y=arena[1]+move[1])
            body='return {'+','.join('{'+','.join(f'{k}={v}' for k,v in c.items())+'}' for c in commands)+'}\n'
            temporary=bots/'gpu_commands.tmp';temporary.write_text(body);temporary.replace(bots/'gpu_commands.lua')
            end=time.monotonic()+2.5
            while time.monotonic()<end:read();time.sleep(.02)
            case_frames=frames[start:];case_orders=orders[order_start:]
            if not case_orders or not all(o['accepted'] for o in case_orders):raise RuntimeError(f'Unaccepted execution: {name}')
            executed=case_orders[0]['tick'];victim=[f for f in case_frames if f['team']==3 and f['tick']>=executed]
            peak=max(f['health'] for f in victim);hits=[f['tick'] for f in victim if f['health']<peak-5]
            release=[f['tick'] for f in victim if f['projectiles']]
            if len({f['team'] for f in case_frames})!=2:raise RuntimeError(f'Missing own-side frames: {name}')
            row={'name':name,'orders':case_orders,'first_projectile_tick':min(release) if release else None,
                'first_damage_tick':min(hits) if hits else None,'damage_delay_ticks':min(hits)-executed if hits else None,
                'initial_health':peak,'minimum_health':min(f['health'] for f in victim),'frames':case_frames}
            if move:
                own=[f for f in case_frames if f['team']==2 and f['tick']>=executed];origin=own[0];moving=[f['tick']for f in own if (f['x']-origin['x'])**2+(f['y']-origin['y'])**2>1]
                row.update(first_movement_tick=min(moving)if moving else None,final_position=[own[-1]['x'],own[-1]['y']],target=[arena[0]+move[0],arena[1]+move[1]])
            results.append(row);print(json.dumps({k:v for k,v in row.items() if k!='frames'}),flush=True)
            (run/'partial.json').write_text(json.dumps({'capture_passed':False,'diagnostic_cheats':True,'training_eligible':False,'arena':arena,'scenarios':results},indent=2))
        by_name={r['name']:r for r in results}
        if a.suite=='combat':passed=all(by_name[n]['first_damage_tick'] is not None for n in ('attack','cancel_after_release','reissue_near_raze','raze')) and all(by_name[n]['first_damage_tick'] is None for n in ('cancel_before_release','cancel_raze'))
        elif a.suite=='attack-turn':passed=all(r['first_projectile_tick']is not None and r['first_damage_tick']is not None for r in results)
        elif a.suite=='movement':passed=all((r['first_movement_tick']is not None or r['name']=='stop_turn_180')and(r['name'].startswith('stop_')or sum((x-y)**2 for x,y in zip(r['final_position'],r['target']))<=32**2)for r in results)
        else:passed=all(any(f['team']==2 and(f['casting']or f['using'])for f in r['frames'])for r in results)and by_name['baseline_far_raze']['first_damage_tick']is not None
        (run/'scenarios.json').write_text(json.dumps({'capture_passed':passed,'suite':a.suite,'diagnostic_cheats':True,'training_eligible':False,'arena':arena,'scenarios':results},indent=2))
        if not passed:raise RuntimeError('Combat capture assertions failed; data is not a validated reference')
if __name__=='__main__':main()
