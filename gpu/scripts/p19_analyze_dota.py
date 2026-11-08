"""Extract measured balances; keep ambiguous simultaneous creep rewards explicit."""
import argparse
from collections import Counter,defaultdict
import csv
import json
from pathlib import Path


def creep_type(kill):
    name=kill['victim']
    if 'flagbearer' in name:return 'flagbearer'
    if 'siege' in name:return 'siege'
    if 'ranged' in name:return 'ranged'
    if 'melee' in name:return 'flagbearer' if kill.get('flagbearer') else 'melee'
    return None


def analyze(run):
    rows=[json.loads(line) for line in (run/'events.jsonl').read_text().splitlines()]
    economy=[];initial=[];terminals=[];ordered_rows=[]
    for order,line in enumerate((run/'dota.log').open(errors='replace')):
        if 'SF_P19|' in line:
            ordered_rows.append({**json.loads(line.split('SF_P19|',1)[1]),'_order':order})
        for tag,target in [('SF_CHIO_ECONOMY|',economy),('SF_P19_BOT_INITIAL|',initial),('SF_CHIO_TERMINAL|',terminals)]:
            if tag in line:target.append({**json.loads(line.split(tag,1)[1]),'_order':order})
    if ordered_rows:rows=ordered_rows
    samples=[r for r in rows if r['kind']=='sample']
    if samples:
        through=max(r['time'] for r in samples)
        rows=[r for r in rows if r['time']<=through]
        economy=[r for r in economy if r['time']<=through]
    states=[r for r in rows if r['kind']=='state']
    kills=[r for r in rows if r['kind']=='kill']
    attempts=[r for r in economy if r['kind'] in ('attempt','start_purchase')]
    for row in initial:
        if 'xp' in row:row['xp_to_next']=row.pop('xp')
    transactions=[];teams={};awards=[];xp_awards=[];last={}
    for row in rows:
        if row['kind']=='state':last[row['team']]=row
        elif row['kind']=='kill':
            for h in row.get('heroes',[]):
                old=last.get(h['team'])
                if old and creep_type(row) and h['team']!=row['team']:
                    xp_awards.append(dict(team=h['team'],time=row['time'],type=creep_type(row),
                        denied=row.get('attacker_team')==row['team'],xp=h['xp']-old['xp'],
                        bounty=row.get('xp_bounty')))
                last[h['team']]=h
    for team in (2,3):
        values=[r for r in states if r['team']==team]
        purchases=[r for r in attempts if r['team']==team]
        passive=[];ambiguous=[]
        for before,after in zip(values,values[1:]):
            lo,hi=before['time'],after['time']
            buys=[p for p in purchases if (before['_order']<p['_order']<=after['_order']
                  if '_order' in before else lo+1e-5<p['time']<=hi+1e-5)]
            spent=sum(p.get('gold_before',p['gold'])-p['gold'] for p in buys)
            dr=after['reliable']-before['reliable'];du=after['unreliable']-before['unreliable']
            dh=after['lh']-before['lh'];dd=after['deny']-before['deny'];dx=after['xp']-before['xp']
            related=[k for k in kills if lo-.04<k['time']<=hi+.04 and creep_type(k)]
            hit=[k for k in related if k.get('attacker')=='npc_dota_hero_nevermore' and k.get('attacker_team')==team
                 and (k['team']!=team if dh else k['team']==team)]
            special=[k for k in kills if lo-.04<k['time']<=hi+.04 and not creep_type(k)]
            # With no LH, spending is added back to the full balance delta.
            # With an LH and no spending, reliable gold isolates periodic income.
            periodic=dr+du+spent if dh==0 else dr if not buys else None
            if not special and periodic is not None and 0<periodic<=3 and after['deaths']==before['deaths']:
                passive.append(dict(time=hi,gold=periodic))
            if dh or dd:
                unique=dh+dd==len(hit) and len({creep_type(k) for k in hit})==1 and not buys and after['deaths']==before['deaths']
                reward=dict(team=team,time=hi,lh=dh,deny=dd,gold_unreliable=du,xp_delta=dx,
                            candidates=len(hit),attributed=unique)
                if unique:reward.update(type=creep_type(hit[0]),creep=hit[0],gold=du)
                else:ambiguous.append(reward)
                awards.append(reward)
            transactions.append(dict(team=team,time=hi,previous_time=lo,reliable_delta=dr,
                unreliable_delta=du,gold_delta=dr+du,purchase_spend=spent,lh_delta=dh,deny_delta=dd,xp_delta=dx))
        first_items={}
        for item in ('item_wraith_band','item_boots','item_bracer'):
            first=next((r for r in values if any(i['name']==item for i in r['inventory'])),None)
            buy=next((p for p in purchases if p['item']==item and p.get('success')),None)
            first_items[item]={'first_observed_time':first['time'] if first else None,
                               'purchase_time':buy['time'] if buy else None,
                               'purchase_cost':buy.get('gold_before',buy['gold'])-buy['gold'] if buy else None}
        side_samples=[r for r in samples if r['team']==team]
        for sample in side_samples:
            sample['passive_observed_cumulative']=sum(r['gold'] for r in passive if r['time']<=sample['time'])
            for action in ('lh','deny'):
                sample[action+'_gold_attributed_cumulative']=sum(r['gold'] for r in awards
                    if r['team']==team and r['attributed'] and r[action] and r['time']<=sample['time'])
        minute_gold=[]
        for start in range(0,600,60):
            # Complete intervals only. A clipped last interval is never annualized.
            if not side_samples or max(r['time'] for r in side_samples)<start+60-.001:continue
            observed=[r for r in passive if start<=r['time']<start+60]
            minute_gold.append(dict(start=start,seconds=60,gold=sum(r['gold'] for r in observed),
                                    increments=len(observed),method='balance credits after purchase reconciliation; reliable-only on LH'))
        teams[str(team)]=dict(initial=next((r for r in initial if r['team']==team),None),
            samples=len(side_samples),expected_samples=70,missing_times=sorted(set(range(-90,601,10))-{r['target_time'] for r in side_samples}),
            final=side_samples[-1] if side_samples else values[-1],first_items=first_items,passive_intervals=minute_gold,
            passive_observed_gold=sum(r['gold'] for r in passive),ambiguous_credits=len(ambiguous),
            purchases=purchases)
    grouped={}
    for kind in ('melee','ranged','siege','flagbearer'):
        for action in ('lh','deny'):
            subset=[r for r in awards if r['attributed'] and r['type']==kind and r[action]]
            grouped[kind+'_'+action]=dict(events=sum(r[action] for r in subset),gold=sum(r['gold'] for r in subset),
                gold_values=dict(Counter(r['gold'] for r in subset if r[action]==1)),
                multi_kill_totals=[{'gold':r['gold'],'creeps':r[action]} for r in subset if r[action]>1])
    xp_grouped={}
    for kind in ('melee','ranged','siege','flagbearer'):
        for denied in (False,True):
            subset=[r for r in xp_awards if r['type']==kind and r['denied']==denied and r['xp']>0]
            xp_grouped[kind+('_denied' if denied else '_normal')]=dict(events=len(subset),
                xp=sum(r['xp'] for r in subset),xp_values=dict(Counter(r['xp'] for r in subset)))
    samples.sort(key=lambda r:(r['target_time'],r['team']))
    with (run/'samples-10s.csv').open('w',newline='',encoding='utf-8') as f:
        fields=['team','target_time','time','gold','reliable','unreliable','xp','level','lh','deny','deaths',
                'passive_observed_cumulative','lh_gold_attributed_cumulative','deny_gold_attributed_cumulative','inventory']
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader()
        for r in samples:w.writerow({**r,'inventory':json.dumps(r['inventory'])})
    result=dict(teams=teams,reward_by_type=grouped,xp_by_type=xp_grouped,attributed=sum(r['attributed'] for r in awards),
                credits=len(awards),ready=next((r for r in rows if r['kind']=='ready'),None),terminals=terminals,
                method='Ordered server snapshots and purchase logs; bounty is unreliable balance delta on matched deaths of one creep type without a purchase/death. Same-type multi-kills retain total/count. Filters unavailable in native mode21.',
                limitations=['Passive observed totals omit simultaneous LH+purchase or non-creep death transitions; raw balances and transactions retained.',
                             'XP is cumulative hero XP. XP delta near a kill can include other nearby deaths; raw kill snapshots retained.'])
    for name,data in [('analysis',result),('transactions',transactions),('creep-awards',awards),('creep-xp',xp_awards)]:
        (run/(name+'.json')).write_text(json.dumps(data,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='teams'},indent=2))
    for team,t in teams.items():print(team,json.dumps({k:v for k,v in t.items() if k not in ('purchases','final')}))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args();analyze(a.run)
