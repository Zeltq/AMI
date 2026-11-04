"""Verify that passive reward logging preserves the original evaluation trajectories."""
from copy import deepcopy
from pathlib import Path
from p09b_run import RUN,REPORT,read,save

def without_reward(games):
    games=deepcopy(games)
    for game in games:
        for hero in game['sides']:
            for key in ('reward','reward_components','shaping_telescoping_residual'):hero.pop(key,None)
    return games

def main():
    roots={'ZOg-e3':Path('C:/AMI/runs/p09/ZOg/epoch-003'),'ZOg-d1':RUN,'ZOg-d2':RUN,'v10':Path('C:/AMI/runs/p09')}
    rows=[]
    for name,root in roots.items():
        for seed in (20261103,20261104):
            original=root/(f'vs-teacher-{seed}' if name=='ZOg-e3' else f'{name[-2:]}-vs-v10-{seed}' if name.startswith('ZOg-d') else f'teacher-{seed}')/'report.json'
            current=RUN/(f'reward-{name}-{seed}' if name!='v10' else f'reward-v10-{seed}')/'report.json'
            if not current.exists():continue
            old=read(original);new=read(current)
            if not new['complete']:continue
            assert old['complete'] and old['matches']==new['matches']==64
            before=without_reward(old['per_match']);after=without_reward(new['per_match'])
            equal=sum(a==b for a,b in zip(before,after));assert equal==64,(name,seed,equal)
            residual=max(abs(v) for g in new['per_match'] for h in g['sides'] for v in h['shaping_telescoping_residual'])
            antisymmetry=max(abs(sum(h['reward'] for h in g['sides'])) for g in new['per_match'])
            assert all(h['reward_components'][0]==10*g['result']*(1 if side==0 else -1) for g in new['per_match'] for side,h in enumerate(g['sides']))
            rows.append(dict(model=name,seed=seed,equal_match_records=equal,matches=64,max_shaping_residual=residual,max_paired_reward_roundoff=antisymmetry,source_report=str(original),reward_report=str(current)))
    save(REPORT/'passive-reward-parity.json',dict(scope='all per-match fields and minute snapshots, excluding only three added reward fields; no training or evaluation launched',completed_series=len(rows),equal_match_records=sum(r['equal_match_records'] for r in rows),matches=sum(r['matches'] for r in rows),series=rows))

if __name__=='__main__':main()
