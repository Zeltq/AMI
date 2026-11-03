"""Reproducible aggregate evidence for three selected M2 evaluation traces.

No raw replay is copied into git. Attack-cycle telemetry is distinct from hits;
the traces do not contain creep HP at intended impact, so timing versus target
selection remains a hypothesis rather than a measured causal attribution.
"""
import argparse,collections,hashlib,json
from pathlib import Path


def analyze(root):
    report=json.loads((root/'report.json').read_text())
    columns=json.loads((root/'trace-header.json').read_text())['columns']
    series=collections.defaultdict(list)
    with (root/'trace.jsonl').open() as stream:
        for line in stream:
            entry=json.loads(line);actor=entry['actor'];row=dict(zip(columns,entry['v']))
            if actor//2<3 and row['finished']==0 and row['active']>0 and row['health']>0:
                series[actor].append(row)
    games=[]
    for game in report['per_match'][:3]:
        sides=[];index=game['game']
        for side,metrics in enumerate(game['sides']):
            frames=series[index*2+side];post=[r for r in frames if r['game_tick']>=0]
            recorded=collections.Counter(int(r['action']) for r in frames)
            if [recorded[i] for i in range(11)]!=metrics['actions']:
                raise ValueError(f'Trace and independent native command counters differ: game {index}, side {side}')
            counts=collections.Counter(int(r['action']) for r in post)
            longest=collections.Counter();last=None;run=0;attack_followed_by_move_stop=0
            for i,row in enumerate(post):
                action=int(row['action']);run=run+1 if action==last else 1;last=action
                longest[action]=max(longest[action],run)
                if action==3 and any(r['action'] in (1,4) for r in post[i+1:i+5]):
                    attack_followed_by_move_stop+=1
            selected={k:metrics[k] for k in ('controller','side','lh','deny','deaths','first_death_game_seconds',
                'mid_arrival_game_seconds','attack_cycles_completed','attack_cancellations',
                'attack_enemy_creep_commands','deny_commands','attack_hero_commands','attack_structure_commands',
                'commands_during_windup','distance_travelled','longest_idle_seconds','actions')}
            selected.update(postgame_decisions=len(post),postgame_actions=dict(counts),
                longest_postgame_action_runs=dict(longest),
                attack_followed_by_move_stop_within_4_decisions={'numerator':attack_followed_by_move_stop,'denominator':counts[3]},
                enemy_t1_min_distance=min(r['enemy_t1_distance'] for r in post))
            sides.append(selected)
        games.append({'game':index,'game_seconds':game['game_seconds'],'lane_deaths':game['lane_deaths'],'sides':sides})
    totals={}
    for controller in ('GGD','teacher'):
        sides=[s for g in games for s in g['sides'] if s['controller']==controller]
        total={k:sum(s[k] for s in sides) for k in ('lh','deny','deaths','attack_cycles_completed',
              'attack_cancellations','attack_enemy_creep_commands','deny_commands','attack_hero_commands','attack_structure_commands','distance_travelled')}
        total.update(matches=len(sides),lh_deny=total['lh']+total['deny'],
                     all_actions=[sum(s['actions'][i] for s in sides) for i in range(11)],
                     postgame_decisions=sum(s['postgame_decisions'] for s in sides),
                     postgame_actions={str(i):sum(s['postgame_actions'].get(i,0) for s in sides) for i in range(11)})
        totals[controller]=total
    return {'model':'M2','selection':'sampling','seed':report['seed'],'trace_matches_analyzed':[0,1,2],
        'native_action_counter_checks':{'passed':6,'total':6},
        'source':str(root),'source_sha256':{name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in ('report.json','trace.jsonl','trace-header.json')},
        'games':games,'totals':totals,
        'limitations':['Attack cycles are cooldown transitions, not an impact or damage log.',
                      'Move/stop following an attack is not itself proof of cancellation; actual cancellations are separate counters.',
                      'Creep HP at intended impact is absent. Timing versus target selection cannot be causally separated.',
                      'Four diagnostic matches were evaluated; the first three are analyzed. GPU batch size differs from the 64-match selection evaluation.']}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('trace_directory');parser.add_argument('output');args=parser.parse_args()
    result=analyze(Path(args.trace_directory));Path(args.output).write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result['totals'],indent=2))
