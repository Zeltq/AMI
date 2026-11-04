"""Evaluate a frozen GGD policy against the unchanged Lua teacher-v9 on CPU.

The Go collector transports the model's commands. Its other team's legacy
teacher commands are recorded but NEVER executed: that Lua VM runs v9 only.
Only the model side's native ACKs are meaningful for transport admission.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import msvcrt
import os
from pathlib import Path
from queue import Queue, Empty
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'src')]
from prepare_native_teacher import prepare
from selfplay_cycle import Config, policy_service
from sf_duel.live.bridge import literal
from sf_duel.live.game import Game
from sf_duel.live.install import isolated_content


CALLBACK = '''local settings=require("bots/settings")
for k,v in pairs(require("bots/p05_settings")) do settings[k]=v end
local json=require("game/dkjson")
local controller
if GetTeam()==settings.model_team then controller=require("bots/native_executor")
else controller=require("bots/scripted_teacher_v9_live") end
local last_tick=nil
local last_second=nil
local deaths=0
local arrived=false
local timeout_announced=false
local M={}
function M.Think()
 local bot=GetBot()
 if not bot or bot:GetUnitName()~="npc_dota_hero_nevermore" then return end
 local phase=GetGameState()
 if phase~=GAME_STATE_PRE_GAME and phase~=GAME_STATE_GAME_IN_PROGRESS then return end
 local tick=math.floor(GameTime()*30+.5)
 if tick==last_tick then return end
 last_tick=tick
 local ok,why=pcall(controller.Think)
 if not ok then print("SF_NATIVE_ERROR|"..tostring(why));return end
 local p=bot:GetLocation();local t=DotaTime()
 -- The native collector requires both timeout votes; v9 uses its own marker.
 if GetTeam()~=settings.model_team and t>=600 and not timeout_announced then
  timeout_announced=true;print("SF_NATIVE_END|"..GetTeam().."|"..t.."|timeout|"..bot:GetLastHits())
 end
 if not arrived and bot:IsAlive() and p.x*p.x+p.y*p.y<2000*2000 and math.abs(p.x-p.y)<1200 then
  arrived=true;print("SF_P05_MID|"..json.encode({team=GetTeam(),time=t}))
 end
 local d=GetHeroDeaths(bot:GetPlayerID())
 if d>deaths then print("SF_P05_DEATH|"..json.encode({team=GetTeam(),time=t,deaths=d}));deaths=d end
 if math.floor(t)~=last_second then
  last_second=math.floor(t)
  local tower=GetTower(GetTeam(),TOWER_MID_1);local q=tower and tower:GetLocation()
  print("SF_P05_HERO|"..json.encode({team=GetTeam(),time=t,health=bot:GetHealth(),mana=bot:GetMana(),lh=bot:GetLastHits(),deny=bot:GetDenies(),deaths=d,x=p.x,y=p.y,alive=bot:IsAlive(),t1_distance=q and math.sqrt((p.x-q.x)^2+(p.y-q.y)^2)}))
 end
end
return M
'''


def save(path, value):
    temporary = path.with_name(path.name + f'.{os.getpid()}.pending')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pending_at_terminal(decision, terminal_time):
    # The two clocks have a constant offset. Compare engine ticks, not floats
    # around e.g. 201.06668 + 7/30 versus 201.30002.
    terminal_tick=decision.observed_tick+round((terminal_time-decision.observed_dota_time)*30)
    return decision.execute_tick>=terminal_tick


def admitted_result(run, side, seconds):
    from sf_duel.native_data import records
    raw=json.loads((run/'duel-result.json').read_text())
    result=json.loads(json.dumps(raw));end=min(float(raw['time']),seconds)
    if end<=0 and raw.get('error'):raise RuntimeError(raw['error'])
    if raw['time']>seconds:
        # Preserve raw evidence. Only the requested prefix is an evaluation.
        samples=[]
        for line in (run/'dota.log').read_text(errors='replace').splitlines():
            if line.startswith('SF_P05_HERO|'):
                row=json.loads(line.split('|',1)[1])
                if abs(row['time']-seconds)<=1/30+.001:samples.append(row)
        for team in (2,3):
            row=next((s for s in samples if s['team']==team),None)
            if row is None:raise RuntimeError('Cannot censor: missing terminal hero sample')
            result['sides'][str(team)].update(last_hits=row['lh'],denies=row['deny'])
        result.update(time=seconds,reason='timeout',outcome=None)
        result['censored_raw_terminal']=dict(time=raw['time'],reason=raw['reason'])
    decisions={};acks={}
    for row in records(run/'native-decisions.pb.gz'):
        if row.HasField('decision') and row.decision.team==side:
            decisions[row.decision.sequence]=row.decision
        if row.HasField('execution') and row.execution.team==side:
            acks[row.execution.sequence]=row.execution
    checked=[(seq,d) for seq,d in decisions.items() if not pending_at_terminal(d,end)]
    quality=dict(decisions=len(decisions),checked_before_terminal=len(checked),
                 terminal_censored_commands=len(decisions)-len(checked),
                 acknowledged=sum(seq in acks for seq,_ in checked),
                 unacknowledged=sum(seq not in acks for seq,_ in checked),
                 late_executions=sum(seq in acks and acks[seq].tick>d.execute_tick for seq,d in checked),
                 rejected=sum(seq in acks and not acks[seq].accepted for seq,_ in checked))
    result['raw_model_transport']=raw['sides'][str(side)]
    result['sides'][str(side)].update(quality)
    return result


def prepare_bots(source, model_team, engine, seed, profile='ggd-v2'):
    prepare(source)
    for name in ('scripted_teacher_v9.lua', 'scripted_teacher_v9_live.lua'):
        shutil.copy2(ROOT / 'bots' / name, source / name)
    settings = dict(model_team=model_team, teacher_team=5-model_team, p03_seed=seed, observation_profile=profile,
                    teacher_combat=engine['combat'],
                    teacher_prototypes={i: engine['prototypes'][k][16] for i, k in enumerate((0, 2, 6, 7, 8))},
                    muzzle_forward=engine['hero'][24], muzzle_side=engine['hero'][25])
    (source / 'p05_settings.lua').write_text('return ' + literal(settings), encoding='utf-8')
    (source / 'p05_controller.lua').write_text(CALLBACK, encoding='utf-8')
    shutil.copy2(ROOT/'bots/play_model_events.lua', source/'play_model_events.lua')
    (source/'play_model_start.lua').write_text("require('bots/play_model_events').start()\n")
    callback = 'local controller=require("bots/p05_controller")\nfunction Think() controller.Think() end\n'
    (source / 'bot_generic.lua').write_text(callback)
    (source / 'item_purchase_generic.lua').write_text(callback.replace('function Think()', 'function ItemPurchaseThink()'))
    return {p.name: digest(p) for p in source.glob('*.lua')}


def play(args, index, slot, active):
    run = args.run / f'match-{index:02d}'
    run.mkdir()
    side = 2 + index % 2
    engine = json.loads(args.engine.read_text())
    source = run / 'bots'
    manifest = prepare_bots(source, side, engine, args.seed + index, args.profile)
    save(run / 'source-manifest.json', manifest)
    content, _ = isolated_content(args.run / 'runtime' / f'{slot:02d}', source)
    ports = (12250+slot*2, 12251+slot*2)
    command = [str(args.collector), '--duel', '--run', str(run), '--bots', str(source),
               '--log', str(run/'dota.log'), '--radiant', str(ports[0]), '--dire', str(ports[1]),
               '--period', '4', '--offset', '7', '--until', str(args.seconds), '--timeout', '1200',
               '--policy-address', '127.0.0.1:12360', '--policy-team', str(side)]
    started = time.monotonic()
    with (run/'go.log').open('wb') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                   creationflags=subprocess.CREATE_NO_WINDOW)
        active[slot] = dict(index=index, model_team=side, run=str(run), pid=process.pid,
                            command=subprocess.list2cmdline(command))
        try:
            with Game(run, 'P05', args.timescale, 7.5, bot_root=source, content_root=content,
                      game=content.parents[1], game_port=27155+slot*10, engine_threads=2,
                      worldstate_ports=ports, worldstate_frames=1, hero_seed=args.seed+index) as game:
                active[slot]['dota_pid'] = game.pid
                game.command('script_reload_code bots/play_model_start')
                game.wait_log('SF_PLAY_EVENTS_READY', timeout=10)
                (run/'ready').touch()
                process.wait(timeout=1200)
        finally:
            if process.poll() is None:
                process.kill(); process.wait()
    result = admitted_result(run,side,args.seconds)
    quality = result['sides'][str(side)]
    errors = [line for line in (run/'dota.log').read_text(errors='replace').splitlines()
              if 'SF_NATIVE_ERROR|' in line or 'SF_RULE_ERROR|' in line]
    row = dict(index=index, model_team=side, worker=slot, run=str(run),
               wall_minutes=(time.monotonic()-started)/60, exit_code=process.returncode,
               result=result, lua_errors=errors)
    save(run/'match.json', row)
    active.pop(slot, None)
    if process.returncode or errors or quality['unacknowledged'] or quality['late_executions']:
        raise RuntimeError(f'Match {index} failed transport admission: {run}/match.json')
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True, type=Path)
    p.add_argument('--run', required=True, type=Path)
    p.add_argument('--assets', type=Path, default=Path('C:/AMI'))
    p.add_argument('--collector', type=Path, default=Path('C:/AMI/runs/p05/nativecollect.exe'))
    p.add_argument('--engine', type=Path, default=Path('C:/AMI/runs/gpu/ggd0-il-20261102-202637/dataset/environment/engine.json'))
    p.add_argument('--matches', type=int, default=20)
    p.add_argument('--workers', nargs='+', type=int, default=[16, 17, 18, 19])
    p.add_argument('--seconds', type=float, default=600)
    p.add_argument('--timescale', type=float, default=1)
    p.add_argument('--seed', type=int, default=20261103)
    p.add_argument('--inference-device', choices=['cpu'], default='cpu')
    p.add_argument('--sampling', action='store_true', required=True)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    if args.matches <= 0 or args.matches % 2: p.error('An even positive match count is required')
    if len(set(args.workers)) != len(args.workers) or any(not 0 <= s < 20 for s in args.workers): p.error('Invalid worker slots')
    import torch
    from sf_duel.native_policy_server import policy_contract
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    args.profile = saved['gpu_pipeline']['observation_profile']
    if policy_contract(saved) != (128, 'f16') or args.profile not in ('ggd-v2','ggd-v3','ggd-v4'):
        raise ValueError('Requires ggd-v2/v3/v4 / f16 / reset-128 checkpoint')
    args.run = args.run.resolve()
    args.run.mkdir(parents=True, exist_ok=args.resume)
    identity = {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()}
    identity.update(checkpoint_sha256=digest(args.checkpoint), collector_sha256=digest(args.collector),
                    teacher_sha256=digest(ROOT/'bots/scripted_teacher_v9.lua'),
                    previous='issued command', recurrent_reset='128 decisions and death',
                    opponent='Lua teacher-v9; Go opponent commands are unexecuted diagnostics')
    if args.resume:
        original=json.loads((args.run/'config.json').read_text())
        for key in ('checkpoint_sha256','collector_sha256','teacher_sha256','matches','workers','seconds','timescale','seed','sampling'):
            if identity[key]!=original[key]:raise ValueError(f'Resume changed {key}')
    else:save(args.run/'config.json', identity)
    prior_status=json.loads((args.run/'status.json').read_text()) if args.resume else None
    elapsed_before=prior_status['wall_minutes']*60 if prior_status else 0
    started = time.monotonic()-elapsed_before; results = []; active = {}
    pilot = args.assets/'runs/pilot-status.json'
    def status(stage, error=None):
        jobs = list(active.values())
        record = dict(stage=stage, pid=os.getpid(), command=subprocess.list2cmdline([sys.executable,*sys.argv]),
                      cwd=str(ROOT), run=str(args.run), completed=len(results), requested=args.matches,
                      wall_minutes=(time.monotonic()-started)/60,
                      matches_per_hour=len(results)/max(1,time.monotonic()-started)*3600,
                      errors=int(error is not None), retries=0, error=error, jobs=jobs,
                      background_work_remaining=stage not in ('complete','failed'),
                      updated_utc=datetime.now(timezone.utc).isoformat())
        save(args.run/'status.json',record)
        current=json.loads(pilot.read_text(encoding='utf-8'));current['P10' if args.profile=='ggd-v4' else 'P05']=record;save(pilot,current)
        save(args.run/'matches.json',sorted(results,key=lambda r:r['index']))
    # One supervisor across cohorts: never accidentally share service/worker slots.
    with (args.assets/'runs/p05/evaluation.lock').open('a+b') as lock:
        lock.write(b'1');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        try:
            if args.resume:
                history_path=args.run/'resume-history.json'
                history=json.loads(history_path.read_text()) if history_path.exists() else []
                history.append(dict(prior_status=prior_status,command=subprocess.list2cmdline([sys.executable,*sys.argv]),
                                    reason='Recheck terminal censoring with integer ticks and finish missing indices; preserve raw records'))
                save(history_path,history)
                for path in sorted(args.run.glob('match-*/match.json')):
                    row=json.loads(path.read_text());side=row['model_team']
                    row['result']=admitted_result(path.parent,side,args.seconds)
                    quality=row['result']['sides'][str(side)]
                    if row['exit_code'] or row['lua_errors'] or quality['unacknowledged'] or quality['late_executions']:
                        raise RuntimeError(f'Unresolved existing match: {path}')
                    row['admission']='integer terminal tick, requested game horizon'
                    save(path,row);results.append(row)
            status('warmup')
            cfg=replace(Config(),workers=len(args.workers),seed=args.seed)
            if args.resume and (args.run/'service').exists():
                # The preceding supervisor must have stopped its service before
                # releasing evaluation.lock. Keep every old inference log.
                suffix=1
                while (args.run/f'service-before-resume-{suffix}').exists():suffix+=1
                (args.run/'service').rename(args.run/f'service-before-resume-{suffix}')
            with policy_service(args.checkpoint,args.run,cfg,[],record=False,inference_device='cpu',selection='sampling',inference_host='windows'):
                pending=Queue()
                completed={r['index'] for r in results}
                for index in range(args.matches):
                    if index not in completed:pending.put(index)
                def worker(slot):
                    while True:
                        try:index=pending.get_nowait()
                        except Empty:return
                        results.append(play(args,index,slot,active))
                with ThreadPoolExecutor(max_workers=len(args.workers)) as pool:
                    futures=[pool.submit(worker,slot) for slot in args.workers]
                    while not all(f.done() for f in futures):
                        status('evaluating');time.sleep(2)
                    for f in futures:f.result()
            if digest(args.checkpoint)!=identity['checkpoint_sha256']:raise RuntimeError('Checkpoint changed')
            status('complete')
        except BaseException as error:
            status('failed',repr(error));raise


if __name__ == '__main__':
    main()
