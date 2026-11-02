"""Four fixed-weight live games after a service repair; no PPO updates."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from selfplay_cycle import ROOT,Config,collect,atomic
from sf_duel.native_data import records


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();base=Path(a.run).resolve()
    checkpoint=ROOT/'runs/mia-20261031/models/MIA0.pt';cfg=Config.read(ROOT/'runs/mia-20261031/config.json')
    matches=collect(checkpoint,base,cfg,[str(checkpoint)],matches=4)
    summary=[];all_late=[]
    for m in matches:
        decisions={};acks={}
        for r in records(Path(m['run'])/'native-decisions.pb.gz'):
            if r.HasField('decision'):decisions[(r.decision.team,r.decision.sequence)]=r.decision
            if r.HasField('execution'):acks[(r.execution.team,r.execution.sequence)]=r.execution
        for team in (2,3):
            pairs=sorted((key[1],max(0,acks[key].tick-d.execute_tick)) for key,d in decisions.items() if key[0]==team and key in acks)
            late=np.array([v for _,v in pairs]);all_late.extend(late.tolist())
            summary.append({'match':m['index'],'team':team,'commands':len(late),'median_late_ticks':float(np.median(late)),'p99_late_ticks':float(np.quantile(late,.99)),'max_late_ticks':int(late.max()),'early_median':float(np.median(late[:max(1,len(late)//10)])),'late_median':float(np.median(late[-max(1,len(late)//10):]))})
    report={'matches':len(matches),'weights_updated':False,'allow_late':True,'timescale':2,'period_ticks':4,'offset_ticks':6,'actors':summary,'combined_p99_late_ticks':float(np.quantile(all_late,.99)),'max_late_ticks':max(all_late)}
    atomic(base/'transport-report.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':main()
