"""Summarize actual Bot API calls and export observed attack segments."""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path


def kind(u):
    name = u['name']
    if 'nevermore' in name:
        return 'sf_own' if u['own'] else 'sf_enemy'
    if 'tower' in name:
        return 'tower'
    if 'ranged' in name:
        return 'ranged'
    if 'siege' in name:
        return 'siege'
    return 'melee'


def summarize(log, output):
    output.mkdir(parents=True, exist_ok=True)
    methods = defaultdict(lambda: defaultdict(Counter))
    activities = defaultdict(Counter)
    previous = {}
    segments = defaultdict(list)
    ticks = Counter()
    first = {}
    clocks = defaultdict(list)
    events = []
    with log.open(encoding='utf-8', errors='replace') as f:
        for line in f:
            if 'SF_P06_TICK|' not in line:
                continue
            try:
                r = json.loads(line.split('SF_P06_TICK|', 1)[1])
            except json.JSONDecodeError:
                continue  # A live file can end partway through its last line.
            ticks[r['team']] += 1
            first.setdefault(r['team'], r['time'])
            clocks[r['state']].append(r['time'])
            for u in r['units']:
                k = kind(u)
                for name, v in u.items():
                    if name.startswith('Get'):
                        methods[k][name]['calls'] += 1
                        methods[k][name]['ok' if v['ok'] else 'errors'] += 1
                        methods[k][name]['non_null'] += int('value' in v)
                        if not v['ok']:
                            methods[k][name][v['error']] += 1
                value = lambda name: u[name].get('value')
                activity = value('GetAnimActivity')
                activities[k][activity] += 1
                key = (r['team'], u['id'])
                old = previous.get(key)
                row = dict(team=r['team'], id=u['id'], kind=k, name=u['name'], time=r['time'],
                           game_time=r['game_time'], activity=activity, cycle=value('GetAnimCycle'),
                           last_attack=value('GetLastAttackTime'), interval=value('GetSecondsPerAttack'),
                           point=value('GetAttackPoint'), target=value('GetAttackTarget'), hp=u['hp'])
                if activity in (1503, 1504):
                    start = old is None or old['activity'] not in (1503, 1504) or row['cycle'] < old['cycle'] - .05
                    row['start'] = row['time'] if start else old.get('start', row['time'])
                    row['elapsed'] = row['time'] - row['start']
                    segments[k].append(row.copy())
                if old and row['last_attack'] != old['last_attack']:
                    events.append({**row, 'previous_last': old['last_attack'], 'previous_activity': old['activity'],
                                   'previous_cycle': old['cycle']})
                previous[key] = row
    result = dict(ticks=dict(ticks), first_bot_time=first,
                  states={str(k): dict(min=min(v), max=max(v), samples=len(v)) for k,v in clocks.items()},
                  methods={k: dict(v) for k,v in methods.items()}, activities=dict(activities),
                  attack_samples={k: len(v) for k,v in segments.items()}, last_attack_changes=len(events))
    (output/'probe-summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    (output/'release-events.json').write_text(json.dumps(events[:200], indent=2), encoding='utf-8')
    for k, rows in segments.items():
        with (output/f'{k}-attacks.csv').open('w', newline='') as f:
            writer=csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    print(json.dumps({k:v for k,v in result.items() if k != 'methods'}, indent=2))


if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('log', type=Path); p.add_argument('output', type=Path)
    a=p.parse_args(); summarize(a.log, a.output)
