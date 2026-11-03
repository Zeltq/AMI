"""Separate late transport, rejected targets and purchase failures in P05 ACKs."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from sf_duel.native_data import records
from evaluate_dota_teacher import pending_at_terminal


def audit(cohort):
    totals=Counter();by_action=Counter();issued=Counter();games=[]
    for match in json.loads((cohort/'matches.json').read_text()):
        team=match['model_team'];end=match['result']['time'];decisions={};acks={}
        for row in records(Path(match['run'])/'native-decisions.pb.gz'):
            if row.HasField('execution') and row.execution.team==team:
                e=row.execution;acks[e.sequence]=(e.accepted,e.reason,e.tick)
            if row.HasField('decision') and row.decision.team==team:
                d=row.decision
                if pending_at_terminal(d,end):continue
                target=next((u for u in row.observation.units if u.handle==d.target),None)
                decisions[d.sequence]=dict(action=d.action,time=d.observed_dota_time,target=d.target,
                    target_name=target.name if target else None,target_health=target.health if target else None,
                    target_alive=target.is_alive if target else None,execute_tick=d.execute_tick)
        rejected=[]
        for sequence,d in decisions.items():
            if sequence not in acks:raise ValueError('Missing nonterminal ACK')
            accepted,reason,tick=acks[sequence]
            if tick!=d['execute_tick']:raise ValueError('Nonterminal ACK clock differs')
            issued[str(d['action'])]+=1
            if not accepted:
                totals[reason]+=1;by_action[str(d['action'])]+=1
                rejected.append(dict(sequence=sequence,reason=reason,**d))
        games.append(dict(index=match['index'],team=team,checked=len(decisions),rejected=rejected))
    return dict(reasons=dict(totals),rejected_by_action=dict(by_action),checked_by_action=dict(issued),
                checked=sum(issued.values()),rejected=sum(totals.values()),games=games,
                note='Purchase failure ACK may follow a successfully issued movement/attack; a rejected ACK is not always an entirely unexecuted action. Target availability is checked again at execution, seven ticks after observation.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cohort',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=audit(a.cohort);a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='games'}))
