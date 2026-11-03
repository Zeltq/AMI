"""P03: observe two unchanged Lua v9 teachers on an existing isolated worker."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import struct
import subprocess
import sys
import time
import threading
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from sf_duel.live.bridge import literal
from sf_duel.live.game import Game
from sf_duel.live.install import isolated_content


CALLBACK = '''local settings=require("bots/settings")
for k,v in pairs(require("bots/p03_settings")) do settings[k]=v end
local json=require("game/dkjson")
local original_print=print
function print(message,...)
 if type(message)=="string" and string.find(message,"SF_V9_",1,true)==1 then
  original_print("SF_P03_V9|"..GetTeam().."|"..message)
 else original_print(message,...) end
end
local agent=require("bots/agent")
local original_snapshot=agent.snapshot
local first=true
local first_snapshot_time=nil
local creep_decisions=0
local previous={action=0,x=64,y=64,item=0,buy=0}
local core=require("bots/scripted_teacher_v9")
local decide=core.decide
core.decide=function(scene,memory)
 local action,diagnostics=decide(scene,memory)
 previous={action=action.action,x=64,y=64,item=action.item or 0,buy=action.buy or 0}
 if action.action==1 then previous.x=64+(action.x-scene.own.x)/16;previous.y=64+(action.y-scene.own.y)/16 end
 for _,u in ipairs(scene.units) do
  if u.id==action.target_id then previous.target_offset={(u.x-scene.own.x)/2048,(u.y-scene.own.y)/2048};break end
 end
 return action,diagnostics
end
agent.snapshot=function(bot,st)
 local s=original_snapshot(bot,st)
 s.p03_action_type=bot:GetCurrentActionType()
 s.p03_entity_index_available=type(bot.GetEntityIndex)=="function"
 if first_snapshot_time==nil then first_snapshot_time=s.time end
 if settings.capture_start_phases and (s.time-first_snapshot_time<=2 or math.abs(s.time)<=2) then
  original_print("SF_P05_PHASE|"..json.encode({snapshot=s,previous=previous}))
 end
 if first or (#s.creeps>0 and creep_decisions<10) then
  if #s.creeps>0 then creep_decisions=creep_decisions+1 end
  original_print("SF_P03_OBSERVATION|"..json.encode({initial=first,creep_decision=creep_decisions,snapshot=s,previous=previous}))
  first=false
 end
 return s
end
local controller=require("bots/scripted_teacher_v9_live")
local failed=false
local last_tick=nil
local last_second=nil
local last_attack=nil
local last_lh=0
local arrived=false
local dead=false
function Think()
 if failed then return end
 local ok,why=pcall(function()
  local bot=GetBot()
  if not bot or bot:GetUnitName()~="npc_dota_hero_nevermore" then return end
  local tick=math.floor(GameTime()*30+.5)
  if tick==last_tick then return end
  last_tick=tick
  controller.Think()
  local position=bot:GetLocation()
  if not arrived and bot:IsAlive() and position.x^2+position.y^2<2000^2 and math.abs(position.x-position.y)<1200 then
   arrived=true;original_print("SF_P03_MID|"..json.encode({team=bot:GetTeam(),time=DotaTime()}))
  end
  if not dead and GetHeroDeaths(bot:GetPlayerID())>0 then
   dead=true;original_print("SF_P03_HERO_DEATH|"..json.encode({team=bot:GetTeam(),time=DotaTime()}))
  end
  local second=math.floor(DotaTime())
  if second~=last_second then
   last_second=second
   local p=bot:GetLocation()
   original_print("SF_P03_HERO|"..json.encode({team=bot:GetTeam(),time=DotaTime(),health=bot:GetHealth(),mana=bot:GetMana(),lh=bot:GetLastHits(),deny=bot:GetDenies(),deaths=GetHeroDeaths(bot:GetPlayerID()),x=p.x,y=p.y}))
  end
  local attack=bot:GetLastAttackTime()
  if last_attack and attack>last_attack then
   local target=bot:GetAttackTarget()
   original_print("SF_P03_ATTACK_RELEASE|"..json.encode({team=bot:GetTeam(),time=DotaTime(),last_attack_time=attack,target=target and target:GetUnitName() or "unknown"}))
  end
  last_attack=attack
  if bot:GetLastHits()>last_lh then
   local p=bot:GetLocation();local tower=GetTower(bot:GetTeam(),TOWER_MID_1);local q=tower and tower:GetLocation()
   original_print("SF_P03_LH|"..json.encode({team=bot:GetTeam(),time=DotaTime(),count=bot:GetLastHits()-last_lh,x=p.x,y=p.y,t1_distance=q and math.sqrt((p.x-q.x)^2+(p.y-q.y)^2) or nil}))
  end
  last_lh=bot:GetLastHits()
 end)
 if not ok then failed=true;print("SF_V9_ERROR|"..tostring(why)) end
end
'''


def save(path, value):
    temporary = path.with_name(path.name + f'.p03-{os.getpid()}.pending')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


class WorldCapture:
    def __init__(self, run, ports):
        self.stop = threading.Event()
        self.errors = []
        self.threads = [threading.Thread(target=self.receive, args=(run, team, port), daemon=True)
                        for team, port in zip((2, 3), ports)]
        for thread in self.threads:thread.start()

    def receive(self, run, team, port):
        deadline = time.monotonic() + 100
        while not self.stop.is_set():
            try:
                connection = socket.create_connection(('127.0.0.1', port), timeout=1)
                break
            except OSError:
                if time.monotonic() > deadline:self.errors.append(f'world-{team}: connection timeout');return
                time.sleep(.1)
        else:return
        connection.settimeout(1)
        def read(size):
            data = bytearray()
            while len(data) < size and not self.stop.is_set():
                try:chunk = connection.recv(size-len(data))
                except socket.timeout:continue
                if not chunk:raise EOFError()
                data.extend(chunk)
            if len(data) != size:raise EOFError()
            return bytes(data)
        with connection, gzip.open(run / f'world-{team}.pb.gz', 'wb', compresslevel=1) as output:
            try:
                while not self.stop.is_set():
                    header = read(4);size = struct.unpack('<I', header)[0]
                    if not 0 < size < 32_000_000:raise ValueError('Invalid world state size')
                    output.write(header + read(size))
            except EOFError:pass
            except Exception as error:self.errors.append(str(error))

    def close(self):
        self.stop.set()
        for thread in self.threads:thread.join(3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--assets', type=Path, default=Path('C:/AMI'))
    parser.add_argument('--worker', type=int, default=18)
    parser.add_argument('--seconds', type=float, default=600)
    parser.add_argument('--wall-timeout', type=float, default=900)
    parser.add_argument('--timescale', type=float, default=1)
    parser.add_argument('--seed', type=int, default=20261103)
    parser.add_argument('--no-pilot-update', action='store_true')
    parser.add_argument('--capture-world', action='store_true')
    parser.add_argument('--capture-start-phases', action='store_true')
    parser.add_argument('--events-script', type=Path)
    args = parser.parse_args()
    if not 0 <= args.worker < 20:
        parser.error('worker must be 0..19')
    run = args.run.resolve()
    run.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    pilot = args.assets / 'runs/pilot-status.json'
    record = dict(stage='preparing', pid=os.getpid(), command=subprocess.list2cmdline([sys.executable, *sys.argv]),
                  cwd=str(Path.cwd()), run=str(run), worker=args.worker, seconds=args.seconds,
                  timescale=args.timescale, seed=args.seed, started_utc=datetime.now(timezone.utc).isoformat(),
                  background_work_remaining=True, artifacts=[str(run / 'dota.log'), str(run / 'result.json')])

    def status(stage, **extra):
        record.update(stage=stage, elapsed_seconds=time.perf_counter() - started,
                      checked_utc=datetime.now(timezone.utc).isoformat(), **extra)
        save(run / 'status.json', record)
        if not args.no_pilot_update:
            current = json.loads(pilot.read_text(encoding='utf-8'))
            current['P03'] = record.copy()
            save(pilot, current)

    source = run / 'bots'
    source.mkdir()
    names = ('agent.lua', 'hero_selection.lua', 'scripted_teacher_v9.lua', 'scripted_teacher_v9_live.lua')
    for name in names:
        shutil.copy2(ROOT / 'bots' / name, source / name)
    engine = json.loads((args.assets / 'runs/gpu/dota-calibration-003/engine.json').read_text())
    settings = dict(p03_seed=args.seed, capture_start_phases=args.capture_start_phases, teacher_combat=engine['combat'], teacher_prototypes={i: engine['prototypes'][k][16] for i, k in enumerate((0, 2, 6, 7, 8))},
                    muzzle_forward=engine['hero'][24], muzzle_side=engine['hero'][25])
    (source / 'p03_settings.lua').write_text('return ' + literal(settings) + '\n', encoding='utf-8')
    shutil.copy2(args.events_script or ROOT / 'reports/P03/dota_events.lua', source / 'p03_events.lua')
    (source / 'p03_controller.lua').write_text('local M={}\n' + CALLBACK.replace('function Think()', 'function M.Think()') + '\nreturn M\n', encoding='utf-8')
    callback = 'local controller=require("bots/p03_controller")\nfunction Think() controller.Think() end\n'
    (source / 'bot_generic.lua').write_text(callback, encoding='utf-8')
    (source / 'item_purchase_generic.lua').write_text(callback.replace('function Think()', 'function ItemPurchaseThink()'), encoding='utf-8')
    (source / 'ability_item_usage_generic.lua').write_text('function AbilityUsageThink() end\nfunction AbilityLevelUpThink() end\nfunction CourierUsageThink() end\nfunction ItemUsageThink() end\nfunction BuybackUsageThink() end\n')
    hashes = {name: hashlib.sha256((source / name).read_bytes()).hexdigest() for name in names}
    hashes['gpu/src/scripted_teacher.rs'] = hashlib.sha256((ROOT / 'gpu/src/scripted_teacher.rs').read_bytes()).hexdigest()
    save(run / 'source-manifest.json', hashes)
    status('preparing')
    capture = None
    try:
        content, _ = isolated_content(args.assets / 'workers' / f'{args.worker:02d}', source)
        status('starting')
        ports = (12250 + args.worker*2, 12251 + args.worker*2) if args.capture_world else None
        if ports:capture = WorldCapture(run, ports)
        with Game(run, 'P03', args.timescale, 7.5, bot_root=source, content_root=content,
                  game=content.parents[1], game_port=27155 + args.worker * 10, engine_threads=2, hero_seed=args.seed,
                  worldstate_ports=ports, worldstate_frames=1) as game:
            status('diagnostic', dota_pid=game.pid)
            game.command('find seed')
            game.command('script_reload_code bots/p03_events')
            deadline = time.monotonic() + args.wall_timeout
            while time.monotonic() < deadline:
                log = game.log.read_text(errors='replace')
                errors = [line for line in log.splitlines() if 'SF_V9_ERROR|' in line or 'SF_RULE_ERROR' in line or 'SF_P03_PROBE_ERROR|' in line]
                if errors:
                    raise RuntimeError('Lua teacher failed: ' + '\n'.join(errors[:8]))
                samples = [json.loads(line.split('SF_P03_HERO|', 1)[1]) for line in log.splitlines() if 'SF_P03_HERO|' in line]
                if 'SF_V9_END|' in log or (samples and max(row['time'] for row in samples) >= args.seconds):
                    save(run / 'result.json', dict(complete=True, seed=args.seed, requested_seconds=args.seconds, last_samples=samples[-4:],
                                                  terminal='SF_V9_END|' in log, manifest=hashes))
                    break
                if game.process.poll() is not None:
                    raise RuntimeError('Dota process exited before completion')
                status('diagnostic', game_seconds=max((row['time'] for row in samples), default=None), hero_samples=len(samples))
                time.sleep(.25)
            else:
                raise TimeoutError('Dota diagnostic exceeded wall-timeout')
        if capture:
            capture.close()
            if capture.errors:raise RuntimeError(capture.errors)
        status('complete', background_work_remaining=False)
    except Exception as error:
        save(run / 'result.json', dict(complete=False, error=str(error), type=type(error).__name__, manifest=hashes))
        status('blocked', error=str(error), background_work_remaining=False)
        raise
    finally:
        if capture:capture.close()


if __name__ == '__main__':
    main()
