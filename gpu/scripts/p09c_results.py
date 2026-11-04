"""P09c GitHub evidence: both players, explicit denominators, six reward components."""
import json,math,re
from pathlib import Path
from p09c_run import ROOT,RUN,REPORT,read,save,sha,history,used
from p09b_results import summaries,csvwrite,lines
from p09_results import ratio
from p09c_reward import NAMES

def curves():
    updates=[];bins=[];coverage=[]
    for folder in sorted(RUN.glob('rl-*')):
        if not (folder/'config.json').exists():continue
        cfg=read(folder/'config.json')
        if cfg.get('critic_only') or cfg.get('learning_rate')==0:continue
        chain='pilot2' if 'pilot2' in folder.name else 'pilot1' if 'pilot1' in folder.name else 'main'
        resumed=read(Path(cfg['resume_checkpoint'])/'progress.json')['training_seconds'] if cfg.get('resume_checkpoint') else 0.
        for r in lines(folder/'history.jsonl'):
            if r['learning'].get('warmup'):continue
            updates.append(dict(run=folder.name,chain=chain,block=r['block'],training_minutes=r['training_seconds']/60,new_decisions=r['total_decisions'],new_training_seconds=r['training_seconds']-resumed,decisions_per_second=r['total_decisions']/max(1e-9,r['training_seconds']-resumed),**{k:r['learning'].get(k) for k in ('value_loss','ev_post0','rollout_kl_m2','entropy','clip_fraction','kl_coefficient','valid_decisions')}))
        episodes=lines(folder/'episodes.jsonl');complete=[r for r in episodes if not r.get('initialization_prefix_decisions',0)]
        coverage.append(dict(run=folder.name,completed=len(episodes),complete_reward=len(complete),excluded_frozen_prefix=len(episodes)-len(complete)))
        for start in range(0,len(complete),100):
            group=complete[start:start+100];heroes=[next(h for h in r['game']['sides'] if h['controller']=='GGD') for r in group]
            row=dict(run=folder.name,chain=chain,first=start+1,last=start+len(group),matches=len(group),training_minutes=group[-1]['training_seconds']/60,reward_sum=sum(r['reward'] for r in group),farm=sum(h['lh']+h['deny'] for h in heroes),spawn=sum(r['game']['lane_creeps_spawned'] for r in group),death_matches=sum(h['first_death_game_seconds'] is not None for h in heroes))
            for i,key in enumerate(NAMES):row['reward_'+key]=sum(r['components'][i] for r in group)
            bins.append(row)
    csvwrite(REPORT/'rl-updates.csv',updates);csvwrite(REPORT/'rl-100-matches.csv',bins);save(REPORT/'curve-coverage.json',coverage)
    for chain in sorted(set(r['chain'] for r in updates)):
        u=[r for r in updates if r['chain']==chain];b=[r for r in bins if r['chain']==chain];xmax=max(r['training_minutes'] for r in u)
        panels=[('KL к ZOg-d1',{'KL':[(r['training_minutes'],r['rollout_kl_m2']) for r in u]}),('Энтропия',{'entropy':[(r['training_minutes'],r['entropy']) for r in u]}),('Value loss',{'loss':[(r['training_minutes'],r['value_loss']) for r in u]}),('EV после 0:00 (train GAE)',{'EV':[(r['training_minutes'],r['ev_post0']) for r in u]}),('Clip fraction',{'clip':[(r['training_minutes'],r['clip_fraction']) for r in u]})]
        for title,keys,denom in [('Награда / матч',['reward_sum'],'matches'),('Компоненты / матч',['reward_'+k for k in NAMES],'matches'),('LH+deny / появившиеся крипы',['farm'],'spawn'),('Матчи со смертью / матчи',['death_matches'],'matches')]:
            panels.append((title,{k:[(r['training_minutes'],r[k]/r[denom]) for r in b if r[denom]] for k in keys}))
        svg=['<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="1900"><rect width="100%" height="100%" fill="white"/>']
        colors=['#1769aa','#d97706','#16803c','#b91c1c','#9333ea','#475569']
        for index,(title,series) in enumerate(panels):
            series={k:[(x,y) for x,y in v if y is not None and math.isfinite(y)] for k,v in series.items()};values=[y for v in series.values() for x,y in v]
            if not values:continue
            lo,hi=min(values),max(values);span=max(hi-lo,1e-8);top=25+index*205
            svg.append(f'<text x="55" y="{top}" font-size="16">{title}; min={lo:.4g}, max={hi:.4g}</text>')
            for color,(key,values) in zip(colors,series.items()):
                points=' '.join(f'{55+x/max(xmax,1)*990:.2f},{top+155-(y-lo)/span*130:.2f}' for x,y in values)
                svg.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="1.5"/>')
            for j,(color,key) in enumerate(zip(colors,series)):svg.append(f'<text x="{55+j*170}" y="{top+180}" fill="{color}" font-size="12">{key}</text>')
        svg.append(f'<text x="55" y="1880">X: минуты обучения, 0–{xmax:.2f}; эпизодные точки: до 100 матчей, знаменатели в CSV</text></svg>')
        (REPORT/f'{chain}-curves.svg').write_text('\n'.join(svg),encoding='utf-8',newline='\n')
    return updates,bins

def evaluations():
    groups={};registry=read(RUN/'evaluations.json') if (RUN/'evaluations.json').exists() else []
    # In-process long-run evaluations are discoverable even before controller exits.
    plan=read(RUN/'main-plan.json') if (RUN/'main-plan.json').exists() else {}
    for folder in RUN.glob('rl-main-*'):
        for path in folder.glob('eval-*/report.json'):
            r=read(path)
            if not r.get('complete'):continue
            block=int(path.parent.name.split('-')[1]);progress=read(folder/f'checkpoint-{block:06}/progress.json')
            label='RL-final' if progress['training_seconds']/60>=plan.get('end_training_minutes',float('inf')) else 'RL-1h'
            other='v10' if '-v10-' in path.parent.name else 'ZOg-d1'
            registry.append(dict(model=label,opponent=other,report=str(path)))
            if label=='RL-final' and plan.get('shared_hour_final'):
                registry.append(dict(model='RL-1h',opponent=other,report=str(path)))
    unique={(r['model'],r['opponent'],str(r['report'])):r for r in registry}
    for entry in unique.values():
        path=Path(entry['report']);r=read(path)
        if not r.get('complete'):continue
        assert r['matches']==64 and len(r['per_match'])==64
        label,other=entry['model'],entry['opponent'];config=read(path.parent/'config.json')
        assert not config.get('argmax',False)
        for g in r['per_match']:
            for side,h in enumerate(g['sides']):
                c=h['event_v2_components'];assert len(c)==6
                assert abs(c[0]-10*g['result']*(1 if side==0 else -1))<1e-5
                assert abs(c[1]-.5*h['lh'])<1e-3 and abs(c[2]-.2*h['deny'])<1e-3
                assert c[3]<=1e-6 and c[4]>=-1e-6 and c[5]<=1e-6
                assert abs(sum(c)-h['event_v2_reward'])<1e-3
        groups.setdefault((label,other),{})[r['seed']]=r['per_match']
        compact=[{**g,'sides':[{k:v for k,v in h.items() if k!='minute_snapshots'} for h in g['sides']]} for g in r['per_match']]
        evidence=dict(model=label,opponent=other,seed=r['seed'],config=config,source=str(path),evaluation_minutes=r['wall_seconds']/60,per_match=compact)
        if config.get('weights'):evidence['weights_manifest_sha256']=sha(Path(config['weights'])/'manifest.json')
        if (path.parent/'opponent.json').exists():evidence['opponent_provenance']=read(path.parent/'opponent.json')
        (REPORT/f'{label}-vs-{other}-{r["seed"]}.json').write_text(json.dumps(evidence,ensure_ascii=False,separators=(',',':')),encoding='utf-8')
    rows=[]
    for (label,other),seeds in groups.items():
        games=[g for group in seeds.values() for g in group];rows+=summaries(label,dict(per_match=games),other,'event-v2')
    save(REPORT/'metrics.json',rows);csvwrite(REPORT/'metrics.csv',rows);return rows

def main():
    REPORT.mkdir(parents=True,exist_ok=True);rows=evaluations();updates,bins=curves()
    totals=[r for r in rows if r['side']=='total' and r['perspective']==0]
    table=['Против v10 (W/D/L и смерти — матчи; награда — event-v2):','','| Модель | W / D / L | LH / N | deny / N | farm / spawn | farm / циклы | Смерти / N | <120с / N | reward / N |','|---|---|---|---|---|---|---|---|---|']
    order=['ZOg-d1','pilot1','pilot2','RL-1h','RL-final','v10']
    vs=sorted([r for r in totals if r['matchup'].endswith(' vs v10')],key=lambda r:order.index(r['model']) if r['model'] in order else 99)
    for r in vs:
        n=r['matches'];table.append('| '+' | '.join([r['model'],'; '.join(ratio(r[k],n) for k in ('wins','draws','losses')),ratio(r['lh'],n),ratio(r['deny'],n),ratio(r['lh']+r['deny'],r['lane_creeps_spawned']),ratio(r['lh']+r['deny'],r['attack_cycles_completed']),ratio(r['matches_with_death'],n),ratio(r['early_deaths'],n),ratio(r['reward_sum'],n)])+' |')
    # Historical reference remains explicitly labelled until exact reward is evaluated.
    for model,wdl,lh,deny,spawn,cycles,deaths,early in [('ZOg-d1', (24,0,104),775,626,10058,7253,96,12),('v10',(65,1,62),3651,2132,15820,9907,36,0)]:
        if not any(r['model']==model for r in vs):table.append(f'| {model} (P09b) | {wdl[0]}/128; {wdl[1]}/128; {wdl[2]}/128 | {lh}/128 | {deny}/128 | {lh+deny}/{spawn} | {lh+deny}/{cycles} | {deaths}/128 | {early}/128 | не измерено |')
    table+=['','Против исходной ZOg-d1; парные значения — RL / ZOg-d1:','','| Модель | W / D / L | farm / spawn, обе | Смерти / N, обе | <120с / N, обе |','|---|---|---|---|---|']
    for r in totals:
        if not r['matchup'].endswith(' vs ZOg-d1'):continue
        old=next(q for q in rows if q['matchup']==r['matchup'] and q['side']=='total' and q['perspective']==1);n=r['matches']
        table.append('| '+' | '.join([r['model'],'; '.join(ratio(r[k],n) for k in ('wins','draws','losses')),'; '.join(ratio(q['lh']+q['deny'],q['lane_creeps_spawned']) for q in (r,old)),'; '.join(ratio(q['matches_with_death'],n) for q in (r,old)),'; '.join(ratio(q['early_deaths'],n) for q in (r,old))])+' |')
    table+=['','Полная статистика обоих участников, R/D/сумма: [metrics.csv](P09c/metrics.csv); отдельные матчи и provenance — JSON в P09c. Кривые: CSV `rl-updates`, `rl-100-matches` и SVG по этапам.','',f'GPU: {used():.3f}/300 минут (консервативное время GPU-процессов).']
    if (RUN/'main-plan.json').exists() and read(RUN/'main-plan.json').get('shared_hour_final'):
        table+=['','План длинной фазы — 60 минут: часовая и финальная строки относятся к одной серии на одном checkpoint, а не к двум независимым оценкам. Причина — остаток GPU-бюджета и резерв 35 минут.']
    begin='<!-- P09c results -->';end='<!-- /P09c results -->';block=begin+'\n'+'\n'.join(table)+'\n'+end
    path=ROOT/'reports/P09c.md';text=path.read_text(encoding='utf-8')
    if begin in text:text=text[:text.index(begin)]+block+text[text.index(end)+len(end):]
    else:text=text.replace('## Результаты','## Результаты\n\n'+block,1)
    assert len(text.splitlines())<=150;path.write_text(text,encoding='utf-8',newline='\n')
    for path in REPORT.iterdir():
        if path.is_file():assert path.stat().st_size<1_000_000,path

if __name__=='__main__':main()
