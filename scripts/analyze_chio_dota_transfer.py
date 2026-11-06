"""Read saved Dota traces and ready P16 reports; no inference or simulator runs."""
import collections
import gzip
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = Path('C:/AMI/runs/p17')
P16 = Path('C:/AMI/runs/p16')
NAMES = ['MOVE', 'STOP', 'CREEP', 'DENY', 'HERO', 'TOWER', 'NEAR', 'MID', 'FAR', 'REQUIEM', 'FLASK', 'MANGO', 'TREE']


def low_priority():
    if os.name == 'nt':
        import ctypes
        k = ctypes.WinDLL('kernel32', use_last_error=True)
        k.GetCurrentProcess.restype = ctypes.c_void_p
        k.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
        assert k.SetPriorityClass(k.GetCurrentProcess(), 0x4000)


def distance(a, b):
    return math.hypot(a['x'] - b['x'], a['y'] - b['y'])


def analyze(game, stop):
    run = Path(game['run'])
    decisions = {d['observed_tick']: d for d in map(json.loads, (run/'policy/decisions.jsonl').open())}
    acks = {}
    events = []
    for line in (run/'dota.log').open(errors='replace'):
        if 'SF_CHIO_ACK|' in line:
            a = json.loads(line.split('SF_CHIO_ACK|', 1)[1]); acks[a['sequence']] = a
        elif 'SF_PLAY_EVENT|' in line:
            e = json.loads(line.split('SF_PLAY_EVENT|', 1)[1])
            if 0 <= e.get('time', -1) <= stop: events.append(e)
    c = collections.Counter()
    actions = collections.Counter()
    free_actions = collections.Counter()
    ack_reasons = collections.Counter()
    snapshots = []
    examples = []
    previous = None
    run_start = None
    last_target = None
    own_entity_ids = {e['attacker'] for e in events if e['kind'] == 'hurt' and e.get('team') == game['model_team'] and e.get('ability', '').startswith('nevermore_')}
    with gzip.open(run/'policy/observations.jsonl.gz', 'rt') as f:
        for line in f:
            s = json.loads(line); d = decisions[s['tick']]
            assert d['observed_tick'] == s['tick'] and d['action_selection'] == 'sampling'
            if not 0 <= s['time'] <= stop or not s['navigation_ready']: continue
            own = s['self']; enemy = s.get('enemy') or {}; a = d['action']
            c['decisions'] += 1; actions[NAMES[a]] += 1
            if not s.get('pending_cast'): free_actions[NAMES[a]] += 1; c['free_decisions'] += 1
            ack = acks.get(d['sequence'])
            if ack:
                c['acks'] += 1; c['accepted'] += bool(ack['accepted']); ack_reasons[ack['reason']] += 1
                if ack['accepted']: c['delay7'] += ack['tick'] - ack['observed_tick'] == 7
            visible = [u for u in s['creeps'] if u.get('visible') and u.get('health', 0) > 0]
            foes = [u for u in visible if u['team'] != own['team']]
            allies = [u for u in visible if u['team'] == own['team'] and u['health'] <= u['health_max']*.5]
            def in_range(u): return distance(own, u) <= own['attack_range'] + own.get('hull', 0) + u.get('hull', 0)
            for label, units in [('creep', foes), ('deny', allies)]:
                near = [u for u in units if in_range(u)]
                c[label+'_visible'] += bool(units); c[label+'_in_range'] += bool(near)
                if near:
                    c[label+'_in_range_'+NAMES[a]] += 1
                    # A proxy only: ignores armour, incoming projectiles and attack timing.
                    weak = any(u['health'] <= own['attack_damage'] for u in near)
                    c[label+'_raw_damage_hp_proxy'] += weak
                    if weak: c[label+'_raw_damage_hp_proxy_'+NAMES[a]] += 1
            if foes: c['nearest_creep_distance_sum'] += min(distance(own, u) for u in foes); c['nearest_creep_distance_n'] += 1
            if enemy.get('visible') and enemy.get('health', 0) > 0:
                c['enemy_visible'] += 1; c['enemy_distance_sum'] += distance(own, enemy)
                c['enemy_in_range'] += in_range(enemy)
                if in_range(enemy): c['enemy_in_range_'+NAMES[a]] += 1
            c['low_mana75'] += own['mana'] < 75
            c['attack_damage_sum'] += own['attack_damage']
            c['move_speed_sum'] += own['move_speed']
            c['souls_sum'] += own.get('souls', 0)
            c['boots_owned'] += any(u.get('name') == 'item_boots' for u in s['probe']['inventory'])
            c['bracer_owned'] += any(u.get('name') == 'item_bracer' for u in s['probe']['inventory'])
            c['gold_sum'] += own['gold']
            c['attack_order_active'] += bool(own.get('is_attacking'))
            c['attack_animation'] += bool(own.get('attack_animation'))
            c['off_lane'] += abs(own['x']-own['y']) >= 1200
            if previous:
                c['path_distance'] += distance(own, previous['self'])
                c['attack_clock_resets_proxy'] += own.get('since_attack', 3) + .1 < previous['self'].get('since_attack', 3)
            target = d.get('target') if 2 <= a <= 5 else None
            if target is None or target != last_target: run_start = s['time']
            last_target = target
            if ack and ack['reason'] == 'same_attack':
                u = next((u for u in [*visible, enemy, *s['towers']] if u.get('id') == target), None)
                if u and u.get('visible') and u.get('health', 0) > 0 and in_range(u):
                    c['same_attack_target_in_range'] += 1
                    if not own.get('is_attacking') and not own.get('attack_animation'):
                        c['same_attack_idle_in_range'] += 1
                        if s['time'] - run_start >= 2 and own.get('since_attack', 0) >= 2:
                            c['same_attack_idle_over2s'] += 1
                            if len(examples) < 8:
                                examples.append(dict(time=s['time'], sequence=d['sequence'], action=NAMES[a], target=target,
                                    run_seconds=s['time']-run_start, since_attack=own.get('since_attack'),
                                    attack_target=own.get('attack_target_id'), distance=distance(own, u), target_hp=u['health']))
            if not snapshots or s['time'] >= snapshots[-1]['time'] + 60:
                snapshots.append({k:own.get(k) for k in ['last_hits','denies','souls','level','mana','gold','attack_damage','move_speed']} | {'time':s['time']})
            previous = s
    for e in events:
        # play_model_events.lua emits hurt only for Shadow Fiend attackers.
        if e['kind'] != 'hurt' or e.get('team') != game['model_team']: continue
        if not e.get('ability'):
            target = 'hero' if e.get('target') == 'npc_dota_hero_nevermore' else 'creep' if 'creep' in e.get('target', '') else 'other'
            c['physical_hits_'+target] += 1; c['physical_damage_'+target] += e.get('damage', 0)
    return dict(seed=game['seed'], team=game['model_team'], stop_seconds=stop, counters=dict(c), actions=dict(actions),
                free_actions=dict(free_actions), ack_reasons=dict(ack_reasons), idle_examples=examples, minute_snapshots=snapshots,
                physical_attacker_ids=sorted(own_entity_ids), run=str(run))


def combine(rows):
    result = {}
    for key in ['counters', 'actions', 'free_actions', 'ack_reasons']:
        c = collections.Counter()
        for r in rows: c.update(r[key])
        result[key] = dict(c)
    result['seconds'] = sum(r['stop_seconds'] for r in rows)
    return result


def main():
    low_priority()
    series = json.loads((RUN/'series.json').read_text())
    names = ['sampling-dagger-2', 'ppo-final']
    count = min(len(series[n]['games']) for n in names)
    out = dict(matched_games=count, comparison='same seed and side; each pair truncated to the shorter match, game time >=0', models={})
    stops = [min(series[n]['games'][i]['summary']['sides'][str(series[n]['games'][i]['model_team'])]['end_time'] for n in names) for i in range(count)]
    for name in names:
        rows = [analyze(g, stop) for g,stop in zip(series[name]['games'][:count],stops)]
        out['models'][name] = dict(per_game=rows, aggregate=combine(rows))
    out['simulator'] = {}
    for name in ['sampling-dagger-2', 'ppo-120m']:
        source = P16/f'eval-{name}-v9-sampling/report.json'
        raw = json.loads(source.read_text()); rows = [s for g in raw['per_match'] for s in g['sides'] if s['controller']=='CHiO']
        counters = {k:sum(s[k] for s in rows) for k in ['lh','deny','deaths','hero_attacks_released','attack_hero_commands','attack_enemy_creep_commands','deny_commands','attack_cycles_completed','attack_cancellations','distance_travelled','chio_decisions','alive_seconds']}
        summary = json.loads((Path('C:/AMI-wt/P16/reports/P16')/f'eval-{name}-v9-sampling.json').read_text())['controllers']['CHiO']
        out['simulator'][name] = dict(source=str(source), matches=len(rows), counters=counters,
            outcomes=summary['outcomes'], outcome_reasons=summary['outcome_reasons'], raze_hits=summary['raze_hero_hit_rates'],
            purchases=summary['purchases'], consumable_uses=summary['consumable_uses'])
    out['limitations'] = ['Different trajectories, not controlled counterfactuals.', 'Matched prefixes exclude pregame and later survival advantage.',
        'HP<=raw attack damage is an opportunity proxy, not a guaranteed last hit.', 'Server hurt events count landed hits, not projectile releases; diagnostic only, never policy inputs.',
        'same_attack idle flags are Bot API evidence; no new live experiment or simulator evaluation.']
    dest = ROOT/'reports/P17/transfer-diagnosis.json';dest.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'matched_games':count,'stops':stops,'models':{n:v['aggregate'] for n,v in out['models'].items()},'simulator':out['simulator']},ensure_ascii=False))


if __name__ == '__main__': main()
