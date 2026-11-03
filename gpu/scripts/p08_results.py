"""Small reviewable metrics and standalone Matplotlib SVG curves."""
import argparse
import csv
import json
from pathlib import Path

ROOT = Path('C:/AMI/runs/p08')
REPO = Path(__file__).resolve().parents[2]
OUT = REPO / 'reports/P08'
NAMES = ['lh','deny','opponent_lh','opponent_deny','death','opponent_death','tower_loss','tower_kill','timeout_result']

def read(path): return json.loads(path.read_text(encoding='utf-8-sig'))
def save(path, data): path.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
def lines(path): return [json.loads(x) for x in path.read_text(encoding='utf-8-sig').splitlines() if x.strip()] if path.exists() else []

def verify(stage):
    cfg=read(ROOT/stage/'config.json')
    original=Path(cfg.get('initialize_weights')or cfg['initial_weights']); result=read(ROOT/stage/'result.json'); output=Path(result['latest'])/'weights'
    a=read(original/'manifest.json')['tensors']; b=read(output/'manifest.json')['tensors']
    same=[]; different=[]
    for key in a:
        (same if (original/a[key]['file']).read_bytes()==(output/b[key]['file']).read_bytes() else different).append(key)
    nonvalue=[k for k in a if not k.startswith('value_')]
    history=lines(ROOT/stage/'history.jsonl')
    if cfg['learning_rate']!=0 and not(history and all(h['learning']['warmup']for h in history)):
        raise SystemExit('Value-only verification requires completed warmup-only updates')
    passed=not different if cfg['learning_rate']==0 else all(k in same for k in nonvalue) and any(k.startswith('value_') for k in different)
    result=dict(stage=stage,passed=passed,identical=len(same),total=len(a),different=different,nonvalue_identical=sum(k in same for k in nonvalue),nonvalue_total=len(nonvalue),training_seconds=result['training_seconds'],history=lines(ROOT/stage/'history.jsonl'))
    save(OUT/f'{stage}-verification.json',result)
    if not passed: raise SystemExit(f'{stage} invariant failed: {different}')
    print(json.dumps(result,ensure_ascii=False))

def evaluation(path):
    data=read(path)
    result=dict(run=path.parent.parent.name,name=path.parent.name,complete=data['complete'],matches=data['matches'],wall_seconds=data['wall_seconds'],sides=[])
    for side in ['radiant','dire']:
        rows=[s for g in data['per_match'] for s in g['sides'] if s['controller']=='GGD' and s['side']==side]
        aggregate=next(s for s in data['summary']['sides'] if s['controller']=='GGD' and s['side']==side)
        row=dict(side=side,matches=len(rows))
        for k in ['lh','deny','deaths','attack_cycles_completed','attack_cancellations']:
            row[k]=aggregate[k]
        row['death_matches']=sum(x['deaths']>0 for x in rows)
        row['pregame_death_matches']=sum(x.get('first_death_game_seconds') is not None and x['first_death_game_seconds']<0 for x in rows)
        row['early_deaths']=sum(x.get('first_death_game_seconds') is not None and x['first_death_game_seconds']<120 for x in rows)
        row['actions']=aggregate['actions']
        result['sides'].append(row)
    result['farm']=sum(s['lh']+s['deny'] for s in result['sides'])
    result['early_deaths']=sum(s['early_deaths'] for s in result['sides'])
    result['death_matches']=sum(s['death_matches']for s in result['sides'])
    result['trial_eligible']=(result['early_deaths']<=1 and result['death_matches']<=4) if result['matches']==32 else None
    result['criterion']=result['farm']>=2009 and result['early_deaths']<=2
    return result

def curves(stage):
    history=lines(ROOT/stage/'history.jsonl'); episodes=lines(ROOT/stage/'episodes.jsonl')
    if not history:return
    rows=[]
    for h in history:
        row=dict(block=h['block'],training_minutes=h['training_seconds']/60,episodes=h['episodes'],**h['learning'])
        row.pop('buckets',None)
        for i,name in enumerate(NAMES):row[f'partial_batch_{name}']=h['components_sum'][i]
        rows.append(row)
    with (OUT/f'{stage}-updates.csv').open('w',newline='',encoding='utf-8')as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    grouped={}
    for e in episodes:grouped.setdefault(e['block'],[]).append(e)
    episode_rows=[]
    for block,group in grouped.items():
        row=dict(block=block,training_minutes=group[0]['training_seconds']/60,episodes=len(group),reward=sum(e['reward'] for e in group)/len(group))
        for i,k in enumerate(NAMES):row[k]=sum(e['components'][i] for e in group)/len(group)
        row['own_lh']=row['lh'];row['own_deny']=row['deny']*2;episode_rows.append(row)
    if episode_rows:
        with (OUT/f'{stage}-episodes.csv').open('w',newline='',encoding='utf-8')as f:
            w=csv.DictWriter(f,fieldnames=list(episode_rows[0]));w.writeheader();w.writerows(episode_rows)
    panels=[('PPO diagnostics',rows,['rollout_kl_m2','old_policy_kl','clip_fraction']),('Critic',rows,['value_loss','explained_variance']),('Entropy / anchor coefficient',rows,['rollout_entropy','kl_coefficient']),('Mean episode reward',episode_rows,['reward']),('Farm per episode',episode_rows,['own_lh','own_deny']),('Reward components per episode',episode_rows,NAMES)]
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if 'ev_post0' in rows[0]:panels.insert(2,('EV by game phase',rows,['ev_post0','ev_pregame','warmup_ev_last5']))
    fig, axes = plt.subplots(len(panels), 1, figsize=(12, 2.7*len(panels)), constrained_layout=True)
    for ax, (title, points, keys) in zip(axes, panels):
        ax.set_title(f'{stage}: {title}', loc='left')
        ax.set_xlabel('Training time, minutes')
        ax.grid(alpha=.2)
        if points:
            for key in keys:
                ax.plot([p['training_minutes'] for p in points], [p[key] for p in points], label=key, linewidth=1, marker='.' if len(points)<10 else None)
            ax.legend(fontsize=8, ncol=min(5,len(keys)))
        else:
            ax.text(.5,.5,'No completed episodes',ha='center',transform=ax.transAxes)
    fig.savefig(OUT/f'{stage}-curves.svg')
    figure = OUT/f'{stage}-curves.svg'
    figure.write_bytes(('\n'.join(line.rstrip() for line in figure.read_text(encoding='utf-8').splitlines())+'\n').encode('utf-8'))
    fig.savefig(ROOT/stage/'curves-preview.png', dpi=100)
    plt.close(fig)
    checks=0
    for episode in episodes:
        game=episode['game']; learner=next(s for s in game['sides'] if s['controller']=='GGD'); opponent=next(s for s in game['sides'] if s['controller']=='teacher')
        expected=[learner['lh'],.5*learner['deny'],-.5*opponent['lh'],-.25*opponent['deny'],-10*learner['deaths'],10*opponent['deaths']]
        for actual, want in zip(episode['components'],expected):
            if actual!=want:raise SystemExit(f'Reward/telemetry mismatch in game {game["game"]}: {actual} != {want}')
            checks+=1
    summary=dict(total_decisions=history[-1].get("total_decisions"),pregame_deaths=history[-1].get("pregame_deaths"),pregame_death_matches=history[-1].get("pregame_death_matches"),reward_telemetry_checks=dict(passed=checks,total=checks),stage=stage,blocks=len(history),completed_episodes=len(episodes),training_minutes=history[-1]['training_seconds']/60,last_learning=history[-1]['learning'],component_totals={k:sum(e['components'][i] for e in episodes)for i,k in enumerate(NAMES)})
    save(OUT/f'{stage}-summary.json',summary)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--verify',choices=['zero-lr','warmup','rev-warmup-memory-check']);a=p.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    if a.verify:verify(a.verify)
    else:
        for stage in ['smoke','main']:curves(stage)
        evaluations=[]
        for report in sorted(ROOT.glob('*/eval-*/report.json')):
            item=evaluation(report);evaluations.append(item)
            (OUT/f'{item["run"]}-{item["name"]}.json').write_bytes((json.dumps(read(report),ensure_ascii=False,separators=(',',':'))+'\n').encode('utf-8'))
        save(OUT/'evaluations.json',evaluations)
        baseline=evaluation(REPO/'reports/P02/round-2-sampling.json');baseline.update(run='M2',name='sampling')
        save(OUT/'M2-reference.json',baseline)
        rows=[]
        for item in [baseline,*evaluations]:
            for side in item['sides']:
                row=dict(run=item['run'],evaluation=item['name'],**{k:v for k,v in side.items() if k!='actions'})
                row.update({f'action_{i}':v for i,v in enumerate(side['actions'])});rows.append(row)
        if rows:
            with (OUT/'evaluation-sides.csv').open('w',newline='',encoding='utf-8')as f:
                w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)

        controller=ROOT/'controller.json'
        if controller.exists():save(OUT/'commands.json',read(controller)['history'])
    for p in OUT.iterdir():
        if p.is_file()and p.stat().st_size>=1_000_000:raise SystemExit(f'Oversized report: {p}')
