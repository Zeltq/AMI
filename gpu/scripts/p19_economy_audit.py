"""CPU-only, serial dedicated economy measurements; never installs user bots."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from p17_matches import bundle
from play_vs_chio import below_normal,save
from sf_duel.live.game import Game
from sf_duel.live.install import isolated_content


class AuditGame(Game):
    def wait_log(self,needle,timeout=100):
        result=super().wait_log(needle,timeout)
        if needle=='DOTA_GAMERULES_STATE_PRE_GAME':
            self.command('script_reload_code bots/p19_economy_audit')
        return result


def run_match(run,kind,seconds,model,runtime,siege_probe=False):
    run.mkdir(parents=True,exist_ok=False)
    bots=run/'bots';bundle(bots,2 if kind=='chio-v9' else 0)
    shutil.copy2(ROOT/'gpu/bots/p19_economy_audit.lua',bots/'p19_economy_audit.lua')
    if siege_probe:shutil.copy2(ROOT/'gpu/bots/p19_siege_probe.lua',bots/'p19_siege_probe.lua')
    # Observe the first pregame frame before any purchase or setup grant.
    callback=bots/'bot_generic.lua'
    callback.write_text(callback.read_text().replace('local ok,err=true,nil',"""if not _G.p19_initial then
 _G.p19_initial=true
 local difficulty={};local inventory={}
 for _,name in ipairs({'GetDifficulty','GetBotDifficulty'}) do
  if type(_G[name])=='function' then local ok,v=pcall(_G[name]);if ok then difficulty[name]=v end end
  if type(bot[name])=='function' then local ok,v=pcall(bot[name],bot);if ok then difficulty['bot_'..name]=v end end
 end
 for slot=0,20 do local it=bot:GetItemInSlot(slot);if it then inventory[#inventory+1]={slot=slot,name=it:GetName(),charges=it:GetCurrentCharges()} end end
 print('SF_P19_BOT_INITIAL|'..json.encode({team=GetTeam(),time=DotaTime(),gold=bot:GetGold(),xp_to_next=bot:GetXPNeededToLevel(),player=bot:GetPlayerID(),difficulty=difficulty,inventory=inventory,
 difficulty_constants={easy=BOT_DIFFICULTY_EASY,medium=BOT_DIFFICULTY_MEDIUM,hard=BOT_DIFFICULTY_HARD,unfair=BOT_DIFFICULTY_UNFAIR}}))
 end
 local ok,err=true,nil"""))
    content=runtime/'game/dota'
    if not (runtime/'isolation.json').exists():content,_=isolated_content(runtime,bots)
    save(run/'provenance.json',dict(runtime=str(runtime),game_mode=21,host_timescale=1,below_normal=True,siege_probe=siege_probe,
        model=str(model) if kind=='chio-v9' else None,
        model_sha256=hashlib.sha256(model.read_bytes()).hexdigest() if kind=='chio-v9' else None,
        build=(content/'steam.inf').read_text(),
        bots_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in bots.glob('*.lua')}))
    policy=None;policy_log=None
    record=dict(stage='starting',pid=os.getpid(),command=subprocess.list2cmdline(sys.argv),
                artifacts=[str(run)],background_work_remaining=True,kind=kind,errors=0,retries=0)
    def status(**kw):
        record.update(kw);save(run/'status.json',record)
        pilot=ROOT/'runs/pilot-status.json';pilot.parent.mkdir(exist_ok=True);save(pilot,{'P19':record})
    status();started=time.monotonic()
    try:
        if kind=='chio-v9':
            policy_log=(run/'policy.log').open('wb')
            args=[sys.executable,str(ROOT/'scripts/play_vs_chio.py'),'--model',str(model),
                  '--log',str(run/'dota.log'),'--bots',str(bots),'--output',str(run/'policy'),'--team','2','--seed','20261103']
            policy=subprocess.Popen(args,stdout=policy_log,stderr=subprocess.STDOUT,
                                    creationflags=subprocess.CREATE_NO_WINDOW|subprocess.BELOW_NORMAL_PRIORITY_CLASS,
                                    env={**os.environ,'CUDA_VISIBLE_DEVICES':''})
            status(policy_pid=policy.pid,policy_command=subprocess.list2cmdline(args))
            while not (run/'policy/ready').exists():
                if policy.poll() is not None:raise RuntimeError('CPU policy failed: '+(run/'policy.log').read_text())
                if time.monotonic()-started>90:raise TimeoutError('CPU policy readiness')
                time.sleep(.2)
        with AuditGame(run,'P19-'+run.name,1,30,bot_root=bots,content_root=content,
                       game=content.parents[1],game_port=27929,engine_threads=1,below_normal=True) as game:
            status(stage='measuring',dota_pid=game.pid)
            game.command('script_reload_code bots/p17_setup')
            if siege_probe:game.command('script_reload_code bots/p19_siege_probe')
            for command in ('dota_bot_level','host_timescale','sv_cheats','dota_default_gold','find gold','find difficulty'):
                game.command(command)
            offset=0;tail='';count=0;last_time=-100;last_status=0
            deadline=time.monotonic()+seconds+300
            with (run/'events.jsonl').open('w',encoding='utf-8') as output:
                while time.monotonic()<deadline:
                    if game.process.poll() is not None:raise RuntimeError('Dedicated server exited')
                    if policy and policy.poll() is not None:raise RuntimeError('CPU policy exited')
                    with game.log.open(errors='replace') as f:
                        f.seek(offset);data=tail+f.read();offset=f.tell()
                    lines=data.split('\n');tail=lines.pop()
                    for line in lines:
                        if 'SF_P19|' in line:
                            row=json.loads(line.split('SF_P19|',1)[1]);output.write(json.dumps(row)+'\n');count+=1
                            if row['kind']=='error':raise RuntimeError(row['message'])
                            if row['kind']=='sample':last_time=max(last_time,row['time'])
                        if any(tag in line for tag in ('SF_CHIO_ERROR|','SF_P17_ERROR|','SF_P19_LOAD_ERROR|','SF_P19_PROBE_ERROR|','Script Runtime Error:')):
                            raise RuntimeError(line)
                    output.flush()
                    if last_time>=seconds:break
                    if time.monotonic()-last_status>5:
                        status(game_time=last_time,events=count,elapsed_seconds=time.monotonic()-started,
                               game_seconds_per_wall_second=(last_time+90)/max(1,time.monotonic()-started))
                        last_status=time.monotonic()
                    time.sleep(.1)
                else:raise TimeoutError('Measurement deadline')
        save(run/'result.json',dict(kind=kind,through_time=last_time,events=count))
        status(stage='complete',game_time=last_time,events=count,background_work_remaining=False)
    except BaseException as e:
        status(stage='failed',error=str(e),errors=1,background_work_remaining=False);raise
    finally:
        if policy:
            (run/'policy/stop').touch()
            try:policy.wait(timeout=10)
            except subprocess.TimeoutExpired:policy.terminate();policy.wait(timeout=5)
        if policy_log:policy_log.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--kind',choices=['chio-v9','v9-v9'],required=True);p.add_argument('--seconds',type=int,default=600)
    p.add_argument('--model',type=Path,default=Path('C:/AMI/runs/p17/models/sampling-dagger-2.pt'))
    p.add_argument('--runtime',type=Path,default=Path('C:/AMI/runs/p19/baseline-chio-v9-r2/runtime'))
    p.add_argument('--siege-probe',action='store_true',help='Controlled siege LH/deny after t=60, diagnostic only')
    a=p.parse_args();below_normal()
    # P19 only runs one job at a time. Refuse to coexist with an unknown Dota process.
    tasklist=subprocess.run(['tasklist','/FI','IMAGENAME eq dota2.exe','/FO','CSV'],capture_output=True,text=True,errors='replace',check=True)
    if 'dota2.exe' in tasklist.stdout.lower():raise RuntimeError('Existing Dota process; refusing a duplicate job')
    if a.siege_probe and a.seconds<70:p.error('--siege-probe needs --seconds >=70')
    run_match(a.run,a.kind,a.seconds,a.model,a.runtime,a.siege_probe)


if __name__=='__main__':main()
