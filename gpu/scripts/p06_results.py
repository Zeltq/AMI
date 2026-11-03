"""Export small, auditable P06 metrics after all required experiment stages."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def trace_counts(root,report):
    counts=dict(matches=4,attack_commands=0,non_creep_target=0,forecast_unknown=0,forecast_dead=0,forecast_too_high=0,forecast_hp_window=0)
    for line in (root/'timing-trace.jsonl').open(encoding='utf-8'):
        r=json.loads(line);actor=r['actor'];m=actor//2;v=r['v']
        if actor%2!=m%2 or v[0]>report['per_match'][m]['game_seconds']+1e-3 or v[3]<=0 or v[1]!=3:continue
        counts['attack_commands']+=1
        if 0<=v[2]<6:counts['non_creep_target']+=1
        elif v[16]==0:counts['forecast_unknown']+=1
        elif v[11]<=.5:counts['forecast_dead']+=1
        elif v[11]>v[13]:counts['forecast_too_high']+=1
        else:counts['forecast_hp_window']+=1
    counts['meaning']='HP forecast window only; excludes no own-shot/commitment/competition checks, not teacher label eligibility'
    return counts


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('C:/AMI/runs/p06/experiment'));p.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[2]/'reports/P06');a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    dataset=read(a.root/'dataset/report.json');manifest=read(a.root/'dataset/dataset.json')
    if dataset['partial'] or dataset['games_recorded']!=1024:raise ValueError('Requires the complete single 1024-game dataset')
    result=dict(dataset={k:v for k,v in dataset.items() if k not in ('dagger',)},models={},evaluations=[])
    (a.output/'teacher-summary.json').write_text(json.dumps(read(a.root/'dataset/teacher-summary.json'),indent=2))
    rows=[];validation=[];initial=[];budgets=[]
    for name in ('baseline','honest','oracle'):
        root=a.root/name;train=read(root/'result.json');config=read(root/'config.json');history=read(root/'history.json')
        if not train['complete'] or train['epochs']!=2:raise ValueError('Incomplete two-epoch model: '+name)
        initial.append(config['initial_weights_sha256']);budgets.append(train['updates'])
        model=dict(config=config,training=train,checkpoint_sha256=hashlib.sha256((root/'epoch-002.pt').read_bytes()).hexdigest())
        result['models'][name]=model
        (a.output/(name+'-weights.json')).write_text(json.dumps(read(root/'export/weights/manifest.json'),indent=2))
        for side,v in history[-1]['validation']['attack_by_side'].items():validation.append(dict(model=name,side=side,**v))
        for mode in ('sampling','argmax'):
            folder=a.root/(name+'-'+mode);report=read(folder/'report.json');settings=read(folder/'config.json')
            if not report['complete'] or report['matches']!=64 or report['seed']!=20261103 or settings['argmax']!=(mode=='argmax'):raise ValueError('Evaluation contract mismatch')
            learner=[s for g in report['per_match'] for s in g['sides'] if s['controller']=='GGD']
            if len(learner)!=64:raise ValueError('Wrong learner denominator')
            compact=dict(model=name,mode=mode,settings=settings,trace=trace_counts(folder,report),sides=[])
            for side in ('radiant','dire'):
                group=[s for s in learner if s['side']==side];n=len(group)
                if n!=32:raise ValueError('Unbalanced sides')
                total=lambda k:int(sum(s[k] for s in group))
                row=dict(model=name,mode=mode,side=side,matches=n,lh=total('lh'),deny=total('deny'),deaths=total('deaths'),early_deaths=sum(s['first_death_game_seconds'] is not None and s['first_death_game_seconds']<120 for s in group),cycles=total('attack_cycles_completed'))
                row['conversion_numerator']=row['lh']+row['deny'];row['conversion_denominator']=row['cycles'];rows.append(row);compact['sides'].append(row)
            result['evaluations'].append(compact)
    if len(set(initial))!=1 or len(set(budgets))!=1:raise ValueError('Initialization or update budgets differ')
    result['equal_initialization']=dict(numerator=len(initial),denominator=len(initial),sha256=initial[0])
    result['equal_updates']=dict(numerator=len(budgets),denominator=len(budgets),updates=budgets[0])
    result['teacher_sha256']=manifest['teacher_sha256'];result['observation_profile']=manifest['observation_profile']
    (a.output/'experiment-metrics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    for file,items in [('evaluation.csv',rows),('validation.csv',validation)]:
        with (a.output/file).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(items[0]));w.writeheader();w.writerows(items)
    commands=read(a.root/'commands.json')
    review_commands=[{k:v for k,v in row.items() if k!='minutes' or row['kind'] in ('collection','training','evaluation')} for row in commands]
    (a.output/'experiment-commands.json').write_text(json.dumps(review_commands,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['| Модель | Выбор | Сторона | LH / матчи | Deny / матчи | Смерти / матчи | До 120 с / матчи | (LH+deny) / циклы |','| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for r in rows:lines.append(f"| {r['model']} | {r['mode']} | {r['side']} | {r['lh']}/{r['matches']} | {r['deny']}/{r['matches']} | {r['deaths']}/{r['matches']} | {r['early_deaths']}/{r['matches']} | {r['conversion_numerator']}/{r['cycles']} |")
    lines+=['','| Модель | Сторона validation | ATTACK recall | ATTACK precision | Кадры |','| --- | --- | ---: | ---: | ---: |']
    for r in validation:lines.append(f"| {r['model']} | {r['side']} | {r['tp']}/{r['labels']} | {r['tp']}/{r['predicted']} | {r['frames']} |")
    (a.output/'tables.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    for path in a.output.iterdir():
        if path.is_file() and path.suffix in ('.json','.csv') and path.stat().st_size>=1_000_000:raise ValueError('Oversized review artifact: '+str(path))
    print(json.dumps(dict(models=len(initial),evaluations=len(rows)//2,updates=budgets[0],output=str(a.output))))


if __name__=='__main__':main()
