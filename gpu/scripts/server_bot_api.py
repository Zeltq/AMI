"""Compare the observer's BotAPI attack target with native server NPC targets."""
import argparse,bisect,json
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('captures',nargs='+');p.add_argument('--output',required=True);a=p.parse_args();reports=[]
    for filename in a.captures:
        d=json.loads(Path(filename).read_text());assert d['complete'] and d['bot_frames']
        by_side={s:sorted((f for f in d['bot_frames']if f['team']==s),key=lambda f:f['time'])for s in [2,3]};times={s:[f['time']for f in fs]for s,fs in by_side.items()};phases=[]
        for phase in [-1,0,1]:
            attack_errors=0;aggro_errors=0;queries=0
            for f in d['frames']:
                units=f['units'];ids={u['entity']:(u['name'],u['team'])for u in units}
                for u in units:
                    s=u['team'];t=f['time']+phase/30;i=bisect.bisect_left(times[s],t);candidates=by_side[s][max(0,i-1):i+1]
                    if not candidates:continue
                    b=min(candidates,key=lambda f:abs(f['time']-t))
                    if abs(b['time']-t)>.02:continue
                    observed=next((v for v in [b['hero'],*b['creeps']]if v['name']==u['name']),None)
                    if observed is None:continue
                    target=observed.get('attack_target');identity=(target['name'],target['team'])if target else None
                    attack_errors+=identity!=ids.get(u.get('attack_target'));aggro_errors+=identity!=ids.get(u.get('aggro_target'));queries+=1
            phases.append({'phase_ticks':phase,'queries':queries,'attack_target_errors':attack_errors,'aggro_target_errors':aggro_errors})
        reports.append({'source':filename,'phases':phases,'best':min(phases,key=lambda f:f['attack_target_errors']/max(1,f['queries']))})
    report={'training_eligible':False,'captures':reports,'scope':'Fixed +/-1Think tick phase per capture, all heroes and paired stock creeps. Correlated timing queries; not a general environment certificate.'}
    Path(a.output).write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
