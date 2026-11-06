"""Serial isolated CPU CHiO versus Lua v9; fixed executable/firewall identity."""
import argparse
import json
import msvcrt
from pathlib import Path
import shutil
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from play_vs_chio import below_normal,save
from sf_duel.live.game import Game
from sf_duel.live.bridge import literal
from p17_p16_guard import check as p16_check

CALLBACK='''local settings=require('bots/settings')
for k,v in pairs(require('bots/p17_settings')) do settings[k]=v end
local controller
if GetTeam()==settings.model_team then controller=require('bots/chio_executor')
else controller=require('bots/scripted_teacher_v9_live') end
local json=require('game/dkjson');local last=nil;local second=nil
function Think()
 local bot=GetBot();if not bot or bot:GetUnitName()~='npc_dota_hero_nevermore' then return end
 local phase=GetGameState();if phase~=GAME_STATE_PRE_GAME and phase~=GAME_STATE_GAME_IN_PROGRESS then return end
 local tick=math.floor(GameTime()*30+.5);if tick==last then return end;last=tick
 local ok,err=true,nil
 if settings.audit and GetTeam()~=settings.model_team then bot:Action_ClearActions(true)
 else ok,err=pcall(controller.Think) end
 if not ok then print('SF_CHIO_ERROR|'..tostring(err));return end
 local t=DotaTime()
 if second~=math.floor(t) then
  second=math.floor(t);local p=bot:GetLocation();local tower=GetTower(GetTeam(),TOWER_MID_1)
  print('SF_CHIO_METRIC|'..json.encode({team=GetTeam(),tick=tick,time=t,x=p.x,y=p.y,hp=bot:GetHealth(),
   lh=bot:GetLastHits(),deny=bot:GetDenies(),deaths=GetHeroDeaths(bot:GetPlayerID()),alive=bot:IsAlive(),tower_alive=tower and tower:IsAlive() or false}))
 end
end
'''


def bundle(dest,team,audit=False):
    dest.mkdir(parents=True,exist_ok=False)
    for name in ('agent.lua','chio_executor.lua','chio_economy.lua','hero_selection.lua','scripted_teacher_v9_live.lua','scripted_teacher_v9.lua','play_model_events.lua'):
        shutil.copy2(ROOT/'bots'/name,dest/name)
    for name in ('p17_setup.lua','p17_bridge.lua','p17_audit.lua'):shutil.copy2(ROOT/'gpu/bots'/name,dest/name)
    (dest/'bot_generic.lua').write_text(CALLBACK)
    (dest/'item_purchase_generic.lua').write_text('function ItemPurchaseThink() end\n')
    (dest/'ability_item_usage_generic.lua').write_text('require("bots/bot_generic")\nfunction AbilityUsageThink() Think() end\nfunction AbilityLevelUpThink() end\nfunction CourierUsageThink() end\nfunction ItemUsageThink() end\nfunction BuybackUsageThink() end\n')
    (dest/'p17_settings.lua').write_text('return '+literal(dict(model_team=team,chio=True,audit=audit,observation_profile='ggd-v4'))+'\n')
    for side in (2,3):(dest/f'chio_commands_{side}.lua').write_text('return {}\n')


def match(run,model,team,seconds,runtime,audit=False):
    run.mkdir(parents=True,exist_ok=False);bots=run/'bots';bundle(bots,team,audit)
    log=(run/'policy.log').open('wb')
    args=[sys.executable,str(ROOT/'scripts/play_vs_chio.py'),'--model',str(model),'--log',str(run/'dota.log'),
          '--bots',str(bots),'--output',str(run/'policy'),'--team',str(team)]
    if audit:args.append('--audit')
    policy=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW|subprocess.BELOW_NORMAL_PRIORITY_CLASS)
    record=dict(stage='starting',pid=policy.pid,command=subprocess.list2cmdline(args),background_work_remaining=True)
    def status(**kw):record.update(kw);save(run/'status.json',record);save(ROOT/'runs/pilot-status.json',{'P17':record})
    status();metrics={};acks=[];started=time.monotonic();result=None
    try:
        while not (run/'policy/ready').exists():
            if policy.poll() is not None:raise RuntimeError('Policy startup failed: '+(run/'policy.log').read_text())
            if time.monotonic()-started>60:raise TimeoutError('CPU policy readiness')
            time.sleep(.1)
        content=runtime/'game/dota'
        with Game(run,'P17-'+run.name,1,30,bot_root=bots,content_root=content,game=runtime,
                  game_port=27927,engine_threads=1,below_normal=True) as game:
            status(stage='match',dota_pid=game.pid)
            game.command('script_reload_code bots/p17_setup')
            if audit:
                game.command('dota_creeps_no_spawning 1');game.command('script_reload_code bots/p17_audit')
            offset=0;tail='';deadline=time.monotonic()+seconds+240
            while time.monotonic()<deadline:
                p16_check()
                if policy.poll() is not None:raise RuntimeError('CPU policy failed: '+(run/'policy.log').read_text())
                if game.process.poll() is not None:raise RuntimeError('Dota exited')
                with game.log.open(errors='replace') as f:f.seek(offset);data=tail+f.read();offset=f.tell()
                lines=data.split('\n');tail=lines.pop()
                for line in lines:
                    if audit and 'SF_P17_AUDIT_DONE|' in line:result=dict(audit=True,acks=acks)
                    if 'SF_CHIO_ERROR|' in line or 'SF_P17_ERROR|' in line or 'Script Runtime Error:' in line:raise RuntimeError(line)
                    if 'SF_CHIO_ACK|' in line:acks.append(json.loads(line.split('SF_CHIO_ACK|',1)[1]))
                    if 'SF_CHIO_METRIC|' in line:
                        row=json.loads(line.split('SF_CHIO_METRIC|',1)[1]);metrics[row['team']]=row
                if audit and result is not None:break
                if not audit and len(metrics)==2:
                    reason=next(('death' for r in metrics.values() if r['deaths']>0),None)
                    reason=reason or next(('tower' for r in metrics.values() if not r['tower_alive']),None)
                    reason=reason or ('timeout' if min(r['time'] for r in metrics.values())>=seconds else None)
                    if reason:
                        if reason=='death':loser=next(k for k,r in metrics.items() if r['deaths']>0);winner=5-loser
                        elif reason=='tower':loser=next(k for k,r in metrics.items() if not r['tower_alive']);winner=5-loser
                        else:winner=2 if metrics[2]['lh']>metrics[3]['lh'] else 3 if metrics[3]['lh']>metrics[2]['lh'] else 0
                        result=dict(reason=reason,winner=winner,model_team=team,sides=metrics,censored=seconds<600,acks=acks);break
                status(stage='match',metrics=metrics,acks=len(acks),elapsed_seconds=time.monotonic()-started)
                time.sleep(.1)
            if result is None:raise TimeoutError('Match deadline')
        save(run/'result.json',result);status(stage='complete',background_work_remaining=False,result=str(run/'result.json'))
    except BaseException as e:status(stage='failed',background_work_remaining=False,error=str(e));raise
    finally:
        (run/'policy/stop').touch()
        try:policy.wait(timeout=10)
        except subprocess.TimeoutExpired:policy.terminate();policy.wait(timeout=5)
        log.close()
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--games',type=int,default=8);p.add_argument('--seconds',type=int,default=600)
    p.add_argument('--runtime',type=Path,default=Path('C:/AMI/runs/p17/bridge-001/runtime'));p.add_argument('--audit',action='store_true');a=p.parse_args()
    below_normal();a.run.mkdir(parents=True,exist_ok=False)
    with Path('C:/AMI/runs/p17/runtime.lock').open('a+b') as lock:
        lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        results=[]
        for i in range(a.games):
            p16_check()
            results.append(match(a.run/f'match-{i:03d}',a.model,2+i%2,a.seconds,a.runtime,a.audit))
            save(a.run/'series.json',dict(completed=len(results),requested=a.games,results=results))


if __name__=='__main__':main()
