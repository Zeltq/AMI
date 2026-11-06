"""Summarize execution ACKs and independent server telemetry; never policy inputs."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path


def summarize(run):
    tagged={}
    for line in (run/'dota.log').read_text(errors='replace').splitlines():
        for tag in ('SF_CHIO_ACK','SF_PLAY_EVENT','SF_CHIO_METRIC','SF_CHIO_HANDOFF','SF_CHIO_NAV_ACK','SF_P17_AUDIT_FRAME','SF_P17_AUDIT_CASE'):
            if tag+'|' in line:tagged.setdefault(tag,[]).append(json.loads(line.split(tag+'|',1)[1]))
    acks=tagged.get('SF_CHIO_ACK',[]);events=tagged.get('SF_PLAY_EVENT',[]);frames=tagged.get('SF_P17_AUDIT_FRAME',[])
    accepted=[a for a in acks if a['accepted']]
    result=dict(acks=dict(accepted=len(accepted),total=len(acks)),actions={},delay_ticks=dict(Counter(a['tick']-a['observed_tick'] for a in accepted)),
      rejections=dict(Counter(a['reason'] for a in acks if not a['accepted'])),sides={})
    status_path=run/'policy/status.json'
    if status_path.exists():
        policy_status=json.loads(status_path.read_text(encoding='utf-8-sig'))
        result['action_selection']=policy_status.get('action_selection','unrecorded');result['seed']=policy_status.get('seed')
    for kind in range(13):result['actions'][str(kind)]=dict(accepted=sum(a['action']==kind for a in accepted),issued=sum(a['action']==kind for a in acks))
    result['coverage']=dict(executed=sum(r['accepted']>0 for r in result['actions'].values()),total=13)
    for team in (2,3):
        metrics=[r for r in tagged.get('SF_CHIO_METRIC',[]) if r['team']==team]
        if not metrics:continue
        arrived=[r['time'] for r in metrics if r['alive'] and r['x']**2+r['y']**2<2000**2 and abs(r['x']-r['y'])<1200]
        spells={}
        for name in ('nevermore_shadowraze1','nevermore_shadowraze2','nevermore_shadowraze3','nevermore_requiem','item_flask','item_enchanted_mango','item_tango'):
            casts=sum(e['kind']=='cast' and e.get('team')==team and e.get('ability')==name for e in events)
            hits=sum(e['kind']=='hurt' and e.get('team')==team and e.get('ability')==name and e.get('target')=='npc_dota_hero_nevermore' for e in events)
            spells[name]=dict(casts=casts,hero_hits=hits)
        last=metrics[-1]
        handoffs=[r for r in tagged.get('SF_CHIO_HANDOFF',[]) if r['team']==team]
        nav_acks=[r for r in tagged.get('SF_CHIO_NAV_ACK',[]) if r['team']==team]
        deaths=max(last['deaths'],sum(e['kind']=='death' and e.get('team')==team for e in events))
        result['sides'][str(team)]=dict(lh=last['lh'],deny=last['deny'],deaths=deaths,end_time=last['time'],
            mid_time=handoffs[0]['time'] if handoffs else min(arrived) if arrived else None,spells=spells,
            navigation=dict(handoffs=handoffs,accepted=sum(r['accepted'] for r in nav_acks),orders=len(nav_acks),
                            delay7=sum(r['accepted'] and r['tick']-r['observed_tick']==7 for r in nav_acks)))
    if frames:
        moving=[r for r in frames if r['case']==0 and 9<=r['case_tick']<170];angles=[]
        for x,y in zip(moving,moving[1:]):
            dx=y['x']-x['x'];dy=y['y']-x['y']
            if math.hypot(dx,dy)>.01:angles.append(abs(math.degrees(math.atan2(dy,dx))))
        result['move_angle']=dict(moving_intervals=len(angles),intervals=max(0,len(moving)-1),
            mean_degrees=sum(angles)/len(angles) if angles else None,max_degrees=max(angles) if angles else None,direction=0)
        cancel=[r for r in frames if r['case']==13]
        if cancel:result['stop_cancel']=dict(cast_phase_frames=sum(r['requiem_phase'] for r in cancel),total_frames=len(cancel),
            cooldown_max=max(r['requiem_cooldown'] for r in cancel),mana_spent=cancel[0]['mana']-cancel[-1]['mana'],
            passed=any(r['requiem_phase'] for r in cancel) and not cancel[-1]['requiem_phase'] and max(r['requiem_cooldown'] for r in cancel)==0)
    path=run/'policy/decisions.jsonl'
    if path.exists():
        decisions=[json.loads(x) for x in path.read_text().splitlines()]
        result['move_decisions']=dict(numerator=sum(d['action']==0 for d in decisions),denominator=len(decisions))
        result['inference_ms']=dict(sum=sum(d['inference_ms'] for d in decisions),count=len(decisions),max=max(d['inference_ms'] for d in decisions))
        # Five-second windows with continuous MOVE intent and <25 units total path.
        team=decisions[0]['team'];metrics=[r for r in tagged.get('SF_CHIO_METRIC',[]) if r['team']==team];checked=stuck=0
        for i in range(5,len(metrics)):
            window=metrics[i-5:i+1];commands=[d for d in decisions if window[0]['tick']<=d['observed_tick']<=window[-1]['tick']]
            if len(commands)<25 or any(d['action']!=0 for d in commands):continue
            checked+=1;distance=sum(math.hypot(b['x']-a['x'],b['y']-a['y']) for a,b in zip(window,window[1:]));stuck+=distance<25
        result['stuck_windows']=dict(stuck=stuck,checked=checked,definition='continuous MOVE for5s, sum path<25; overlapping1s windows')
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    result=summarize(a.run);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=='__main__':main()
