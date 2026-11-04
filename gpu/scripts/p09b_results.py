"""Small, reviewable P09b statistics for both players and training curves."""
import argparse,csv,json,math,re
from pathlib import Path
from p09b_run import ROOT,RUN,REPORT,read,save
from p09_results import ratio

def summaries(name,report,opponent_name='v10'):
    rows=[]
    for perspective,label in ((0,name),(1,opponent_name)):
        for side in ('radiant','dire','total'):
            games=[]
            for g in report['per_match']:
                s=(g['game']%2)^perspective;hero=g['sides'][s]
                if side=='total' or hero['side']==side:games.append((g,hero,s))
            r=dict(model=label,matchup=name+' vs '+opponent_name,perspective=perspective,side=side,matches=len(games))
            for tag,value in (('wins',1),('draws',0),('losses',-1)):
                r[tag]=sum(int(g['result'])*(1 if s==0 else -1)==value for g,h,s in games)
                for reason in ('death','T1','LH600'):r[tag+'_'+reason]=sum(int(g['result'])*(1 if s==0 else -1)==value and g['outcome_reason']==reason for g,h,s in games)
            for key in ('lh','deny','attack_cycles_completed','hero_kills','hero_attacks_released','attack_hero_commands','damage_from_hero','damage_from_creeps','damage_from_tower','deaths'):r[key]=sum(h[key] for g,h,s in games)
            r['lane_creeps_spawned']=sum(g['lane_creeps_spawned'] for g,h,s in games)
            deaths=[h['first_death_game_seconds'] for g,h,s in games if h['first_death_game_seconds'] is not None]
            r.update(matches_with_death=len(deaths),first_death_time_sum=sum(deaths),early_deaths=sum(t<120 for t in deaths))
            for i,spell in enumerate(('near','mid','far')):
                r[spell+'_casts']=sum(h['spell_completions'][i] for g,h,s in games)
                r[spell+'_hero_hits']=sum(h['raze_hero_hits_by_spell'][i] for g,h,s in games)
            for i,cause in enumerate(('hero_attack','raze','creeps','tower','requiem')):
                for key in ('count','own_t1_distance_sum','enemy_t1_distance_sum'):r[cause+'_'+key]=sum(h['death_causes'][i][key] for g,h,s in games)
            have=all('reward' in h for g,h,s in games)
            r['reward_matches']=len(games) if have else 0;r['reward_sum']=sum(h['reward'] for g,h,s in games) if have else None
            for i,key in enumerate(('outcome','lh','deny','hp','mana')):r['reward_'+key]=sum(h['reward_components'][i] for g,h,s in games) if have else None
            r['max_shaping_residual']=max((abs(v) for g,h,s in games for v in h.get('shaping_telescoping_residual',[])),default=None)
            rows.append(r)
    return rows

def csvwrite(path,rows):
    if not rows:return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys,lineterminator='\n');w.writeheader();w.writerows(rows)
def lines(path):
    if not path.exists():return []
    result=[]
    for line in path.read_text(encoding='utf-8').splitlines():
        try:result.append(json.loads(line))
        except ValueError:pass # An active writer can leave an incomplete final line.
    return result
def curves():
    updates=[];episodes=[]
    for folder in sorted(RUN.glob('rl*')):
        if not folder.is_dir():continue
        for r in lines(folder/'history.jsonl'):
            if r['learning'].get('warmup'):continue
            updates.append(dict(run=folder.name,block=r['block'],training_minutes=r['training_seconds']/60,**{k:r['learning'].get(k) for k in ('value_loss','ev_post0','rollout_kl_m2','entropy','clip_fraction','kl_coefficient','valid_decisions')}))
        episodes+=lines(folder/'episodes.jsonl')
    # Restarted optimizer history may replay blocks since its last checkpoint.
    by_time={round(r['training_minutes'],6):r for r in updates};updates=sorted(by_time.values(),key=lambda r:r['training_minutes'])
    csvwrite(REPORT/'rl-updates.csv',updates)
    bins=[]
    for start in range(0,len(episodes),100):
        batch=episodes[start:start+100];n=len(batch);games=[r['game'] for r in batch];heroes=[next(h for h in g['sides'] if h['controller']=='GGD') for g in games]
        row=dict(first_match=start+1,last_match=start+n,matches=n,training_minutes=batch[-1]['training_seconds']/60,reward_sum=sum(r['reward'] for r in batch),farm=sum(h['lh']+h['deny'] for h in heroes),spawn=sum(g['lane_creeps_spawned'] for g in games),death_matches=sum(h['first_death_game_seconds'] is not None for h in heroes),prefix_matches=sum(r.get('initialization_prefix_decisions',0)>0 for r in batch))
        for i,key in enumerate(('outcome','lh','deny','hp','mana')):row['reward_'+key]=sum(r['components'][i] for r in batch)
        bins.append(row)
    csvwrite(REPORT/'rl-100-matches.csv',bins)
    if updates:
        panels=[('KL к старту','rollout_kl_m2'),('Entropy','entropy'),('Value loss','value_loss'),('EV после 0:00','ev_post0'),('Clip fraction','clip_fraction')]
        svg=['<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="1050"><rect width="100%" height="100%" fill="white"/>']
        for j,(title,key) in enumerate(panels):
            values=[(r['training_minutes'],r[key]) for r in updates if r.get(key) is not None and math.isfinite(r[key])]
            if not values:continue
            low=min(y for x,y in values);high=max(y for x,y in values);span=max(high-low,1e-8);top=30+j*205
            pts=' '.join(f'{60+x/480*900:.2f},{top+155-(y-low)/span*130:.2f}' for x,y in values)
            svg.append(f'<text x="60" y="{top}" font-size="17">{title}; min={low:.4g}, max={high:.4g}</text><path d="M60 {top+155} H960" stroke="#ccc"/><polyline points="{pts}" fill="none" stroke="#2266aa" stroke-width="1.5"/>')
        svg.append('</svg>');(REPORT/'rl-curves.svg').write_text('\n'.join(svg),encoding='utf-8')
    return updates,bins

def main():
    REPORT.mkdir(exist_ok=True);groups={};completed={}
    registry=read(RUN/'evaluations.json') if (RUN/'evaluations.json').exists() else []
    for entry in registry:
        path=Path(entry['report'])
        if not path.exists():continue
        report=read(path)
        if not report.get('complete'):continue
        completed[(entry['model'],entry['opponent'],report['seed'])]=report['per_match']
    for folder in sorted(RUN.glob('rl*')):
        for path in folder.glob('eval-*/report.json'):
            report=read(path)
            if not report.get('complete'):continue
            parts=path.parent.name.split('-');block=int(parts[1]);opponent=parts[2]
            progress=read(folder/f'checkpoint-{block:06}/progress.json');hour=round(progress['training_seconds']/3600)
            label=f'RL-{hour}h';other='v10' if opponent=='v10' else ('start' if hour==2 else f'RL-{hour-2}h')
            completed[(label,other,report['seed'])]=report['per_match']
    for (label,other,seed),games in completed.items():groups.setdefault((label,other),[]).extend(games)
    rows=[]
    for (name,opponent),games in groups.items():rows+=summaries(name,dict(per_match=games),opponent)
    csvwrite(REPORT/'metrics.csv',rows);save(REPORT/'metrics.json',rows)
    updates,bins=curves()
    totals=[r for r in rows if r['side']=='total' and r['perspective']==0]
    table=['| Модель / соперник | W / D / L | LH / N | deny / N | farm / spawn | farm / циклы | Смерть / N | <120с / N | reward / N |','|---|---|---|---|---|---|---|---|---|']
    for r in totals:
        n=r['matches'];table.append('| '+' | '.join([r['matchup'],'; '.join(ratio(r[k],n) for k in ('wins','draws','losses')),ratio(r['lh'],n),ratio(r['deny'],n),ratio(r['lh']+r['deny'],r['lane_creeps_spawned']),ratio(r['lh']+r['deny'],r['attack_cycles_completed']),ratio(r['matches_with_death'],n),ratio(r['early_deaths'],n),ratio(r['reward_sum'],n) if r['reward_matches'] else 'не измерено'])+' |')
    path=ROOT/'reports/P09b.md';text=path.read_text(encoding='utf-8')
    begin='<!-- P09b live results -->';end='<!-- /P09b live results -->'
    block=begin+'\n'+'\n'.join(table)+'\n\nПолная статистика обеих моделей, по сторонам и суммарно: [metrics.csv](P09b/metrics.csv). Кривые: [SVG](P09b/rl-curves.svg), [PPO CSV](P09b/rl-updates.csv), [по 100 матчей](P09b/rl-100-matches.csv).\n'+end
    if begin in text:text=text[:text.index(begin)]+block+text[text.index(end)+len(end):]
    else:text=text.replace('## Результаты','## Результаты\n\n'+block,1)
    rounds=sum(any(r['model']==f'ZOg-d{k}' and r['matches']==128 for r in totals) for k in (1,2))
    text=re.sub(r'\| DAgger \| .*? \|',f'| DAgger | {rounds}/2 раунда |',text)
    if updates:
        minutes=updates[-1]['training_minutes'];text=re.sub(r'\| RL \| .*? \|',f'| RL | {minutes:.3f}/480 минут обучения |',text)
        text=re.sub(r'Работа продолжается:.*',f'Работа продолжается: RL, {minutes:.3f}/480 минут обучения.',text)
    assert len(text.splitlines())<=150
    path.write_text(text,encoding='utf-8',newline='\n')
    for p in REPORT.iterdir():
        if p.is_file():assert p.stat().st_size<1_000_000,p

if __name__=='__main__':main()
