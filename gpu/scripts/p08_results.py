"""Small reviewable metrics and standalone Matplotlib SVG curves."""
import argparse
import csv
import hashlib
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
    cfg=read(REPO/f'reports/P08/{stage}.json')
    original=Path(cfg['initial_weights']); result=read(ROOT/stage/'result.json'); output=Path(result['latest'])/'weights'
    a=read(original/'manifest.json')['tensors']; b=read(output/'manifest.json')['tensors']
    same=[]; different=[]
    for key in a:
        (same if (original/a[key]['file']).read_bytes()==(output/b[key]['file']).read_bytes() else different).append(key)
    nonvalue=[k for k in a if not k.startswith('value_')]
    passed=not different if stage=='zero-lr' else all(k in same for k in nonvalue) and any(k.startswith('value_') for k in different)
    result=dict(stage=stage,passed=passed,identical=len(same),total=len(a),different=different,nonvalue_identical=sum(k in same for k in nonvalue),nonvalue_total=len(nonvalue),training_seconds=result['training_seconds'],history=lines(ROOT/stage/'history.jsonl'))
    save(OUT/f'{stage}-verification.json',result)
    if not passed: raise SystemExit(f'{stage} invariant failed: {different}')
    print(json.dumps(result,ensure_ascii=False))

def evaluation(path):
    data=read(path)
    result=dict(name=path.parent.name,complete=data['complete'],matches=data['matches'],wall_seconds=data['wall_seconds'],sides=[])
    for side in ['radiant','dire']:
        rows=[s for g in data['per_match'] for s in g['sides'] if s['controller']=='GGD' and s['side']==side]
        row=dict(side=side,matches=len(rows))
        for k in ['lh','deny','deaths','attack_cycles_completed','attack_cancellations']:
            row[k]=sum(x[k] for x in rows)
        row['early_deaths']=sum(x.get('first_death_game_seconds') is not None and x['first_death_game_seconds']<120 for x in rows)
        row['actions']=[sum(x['actions'][i] for x in rows) for i in range(11)]
        result['sides'].append(row)
    result['farm']=sum(s['lh']+s['deny'] for s in result['sides'])
    result['early_deaths']=sum(s['early_deaths'] for s in result['sides'])
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
    fig, axes = plt.subplots(6, 1, figsize=(12, 16), constrained_layout=True)
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
    plt.close(fig)
    summary=dict(stage=stage,blocks=len(history),completed_episodes=len(episodes),training_minutes=history[-1]['training_seconds']/60,last_learning=history[-1]['learning'],component_totals={k:sum(e['components'][i] for e in episodes)for i,k in enumerate(NAMES)})
    save(OUT/f'{stage}-summary.json',summary)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--verify',choices=['zero-lr','warmup']);a=p.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    if a.verify:verify(a.verify)
    else:
        for stage in ['smoke','main']:curves(stage)
        evaluations=[evaluation(p) for p in ROOT.glob('*/eval-*/report.json')]
        save(OUT/'evaluations.json',evaluations)
        controller=ROOT/'controller.json'
        if controller.exists():save(OUT/'commands.json',read(controller)['history'])
    for p in OUT.iterdir():
        if p.is_file()and p.stat().st_size>=1_000_000:raise SystemExit(f'Oversized report: {p}')
