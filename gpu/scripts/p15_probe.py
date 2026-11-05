"""Isolated dedicated-server MOVE_DIRECT fixture; never training data."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from sf_duel.live.game import Game
from sf_duel.live.install import isolated_content
from sf_duel.live.bridge import literal

CASES = [
    dict(name='open', x=80, y=-816, facing=0, direction=0),
    dict(name='turn90', x=80, y=-816, facing=90, direction=0),
    dict(name='turn180', x=80, y=-816, facing=180, direction=0),
    dict(name='tree', x=-192, y=-1392, facing=270, direction=27),
    dict(name='edge', x=80, y=-1040, facing=0, direction=0),
    dict(name='cliff', x=-816, y=-816, facing=0, direction=0),
    dict(name='slope', x=-1200, y=-1008, facing=50, direction=5),
    dict(name='ally', x=80, y=-816, facing=0, direction=0),
    dict(name='wave', x=80, y=-816, facing=0, direction=0),
    dict(name='enemy', x=80, y=-816, facing=0, direction=0),
]


def save(path, value):
    temp = path.with_suffix(f'.{os.getpid()}.pending')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    for attempt in range(20):
        try:
            temp.replace(path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(.05)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--repeats', type=int, default=20)
    parser.add_argument('--timescale', type=float, default=4)
    parser.add_argument('--cases', nargs='+', choices=[c['name'] for c in CASES])
    parser.add_argument('--distance', type=float, default=2000)
    parser.add_argument('--clear-before-move', action='store_true', help='diagnostic contrast only')
    parser.add_argument('--port', type=int, default=27919)
    parser.add_argument('--server-direction', action='store_true', help='privileged diagnostic contrast only')
    args = parser.parse_args()
    if not 1 <= args.repeats <= 100 or not 1 <= args.timescale <= 4:
        parser.error('repeats 1..100, timescale 1..4 required')
    run = args.run.absolute()
    run.mkdir(parents=True, exist_ok=False)
    bots = run / 'bots'
    bots.mkdir()
    for source, dest in [('bots/hero_selection.lua', 'hero_selection.lua'),
                         ('gpu/bots/p15_direct_actor.lua', 'bot_generic.lua'),
                         ('gpu/bots/p15_direct_server.lua', 'p15_server.lua')]:
        shutil.copy2(ROOT / source, bots / dest)
    (bots / 'item_purchase_generic.lua').write_text('function ItemPurchaseThink() end\n')
    (bots / 'ability_item_usage_generic.lua').write_text(
        'local actor=require("bots/bot_generic")\nfunction AbilityUsageThink() Think() end\n'
        'function AbilityLevelUpThink() end\nfunction CourierUsageThink() end\n'
        'function ItemUsageThink() end\nfunction BuybackUsageThink() end\n')
    cases = [dict(case, repeat=i) for case in CASES if not args.cases or case['name'] in args.cases for i in range(args.repeats)]
    config = dict(cases=cases, case_ticks=180, first_decision=30, decision_ticks=6, offset_ticks=6,
                  distance=args.distance, clear_before_move=args.clear_before_move, server_direction=args.server_direction)
    (bots / 'p15_config.lua').write_text('return ' + literal(config) + '\n')
    (bots / 'p15_clock.lua').write_text('return {}\n')
    started = time.monotonic()
    record = dict(stage='preparing', pid=os.getpid(), command=subprocess.list2cmdline([sys.executable, *sys.argv]),
                  cwd=str(ROOT), run=str(run), background_work_remaining=True, artifacts=[str(run / 'dota.log')])

    def status(stage, **kwargs):
        record.update(stage=stage, elapsed_minutes=(time.monotonic()-started)/60, **kwargs)
        save(run / 'status.json', record)
        pilot = Path('C:/AMI/runs/pilot-status.json')
        current = json.loads(pilot.read_text(encoding='utf-8')) if pilot.exists() else {}
        current['P15'] = record
        current.setdefault('P15_jobs', {})[run.name] = record
        save(pilot, current)

    status('preparing')
    frames, orders, setups, bot_frames = [], [], [], []
    try:
        content, _ = isolated_content(run / 'runtime', bots)
        with Game(run, 'P15-direct', args.timescale, 30, bot_root=bots, content_root=content,
                  game=content.parents[1], game_port=args.port, engine_threads=2) as game:
            status('dota-movement', dota_pid=game.pid)
            game.command('dota_creeps_no_spawning 1')
            game.command('script_reload_code bots/p15_server')
            deadline = time.monotonic() + 180 + len(cases)*6/args.timescale*2
            offset, tail, done = 0, '', False
            while time.monotonic() < deadline:
                if game.process.poll() is not None:
                    raise RuntimeError('dedicated server exited')
                with game.log.open(errors='replace') as stream:
                    stream.seek(offset)
                    text = tail + stream.read()
                    offset = stream.tell()
                lines = text.split('\n')
                tail = lines.pop()
                for line in lines:
                    if 'SF_P15_ERROR|' in line or 'Script Runtime Error:' in line:
                        raise RuntimeError(line)
                    if 'SF_P15_START|' in line:
                        clock = json.loads(line.split('SF_P15_START|', 1)[1])
                        temporary = bots / 'p15_clock.tmp'
                        temporary.write_text('return ' + literal(clock) + '\n')
                        temporary.replace(bots / 'p15_clock.lua')
                    for tag, rows in [('FRAME', frames), ('ORDER', orders), ('CASE', setups), ('BOT', bot_frames)]:
                        if f'SF_P15_{tag}|' in line:
                            rows.append(json.loads(line.split(f'SF_P15_{tag}|', 1)[1]))
                    if 'SF_P15_DONE|' in line:
                        done = True
                if done:
                    break
                status('dota-movement', cases_started=len(setups), cases_total=len(cases), samples=len(frames),
                       orders=len(orders), errors=0, retries=0,
                       cases_per_minute=len(setups)/max((time.monotonic()-started)/60, .001))
                time.sleep(.25)
            if not done:
                raise TimeoutError('movement fixture timed out')
        result = dict(complete=done, config=config, cases=setups, frames=frames, orders=orders, bot_frames=bot_frames,
                      training_eligible=False, elapsed_minutes=(time.monotonic()-started)/60,
                      source_sha256={name: hashlib.sha256((bots/name).read_bytes()).hexdigest()
                                     for name in ('bot_generic.lua', 'p15_server.lua', 'p15_config.lua')})
        save(run / 'capture.json', result)
        status('complete', background_work_remaining=False, cases_started=len(setups), samples=len(frames))
    except BaseException as error:
        status('failed', error=str(error), background_work_remaining=False)
        raise


if __name__ == '__main__':
    main()
