"""Byte regression for legacy profiles and fixed-seed legacy model decisions."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[2]
MODELS=[
 ('GGD0','C:/AMI/runs/gpu/ggd0-il-20261102-202637/GGD0-resumed/epoch-003/weights','003','v9'),
 ('M2','C:/AMI/runs/p02/experiment/train-round-2-attempt-1/export/weights','003','v9'),
 ('ZOg','C:/AMI/runs/p09/ZOg/epoch-003/weights','005','v10'),
 ('000470','C:/AMI/runs/p09/P09c/continue90/rl-continue-001/checkpoint-000470/weights','005','v10'),
 ('000644','C:/AMI/runs/p11/train-001/checkpoint-000644/weights','005','v10'),
]
ENGINES={'003':'C:/AMI/runs/gpu/dota-calibration-003/engine.json',
         '004':'C:/AMI/runs/p06/dota-calibration-004/engine.json','005':'C:/AMI/runs/p07/dota-calibration-005/engine.json'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--baseline',required=True);p.add_argument('--candidate',required=True)
    p.add_argument('--seconds',type=float,default=15);a=p.parse_args()
    a.run.mkdir(parents=True,exist_ok=False);commands=[];results=[]
    def run(argv,name):
        start=time.monotonic()
        record=dict(command=subprocess.list2cmdline(argv),argv=argv,cwd=str(ROOT),stage=name)
        commands.append(record);(a.run/'commands.json').write_text(json.dumps(commands,indent=2))
        with (a.run/(name+'.stdout.log')).open('w') as out,(a.run/(name+'.stderr.log')).open('w') as err:
            proc=subprocess.Popen(argv,cwd=ROOT,stdout=out,stderr=err);record['pid']=proc.pid
            (a.run/'commands.json').write_text(json.dumps(commands,indent=2));code=proc.wait()
        record.update(exit_code=code,elapsed_minutes=(time.monotonic()-start)/60)
        (a.run/'commands.json').write_text(json.dumps(commands,indent=2))
        if code:raise RuntimeError(f'{name}: exit {code}; inspect {a.run/name}')
    for profile,environment,teacher in [('ggd-v2','003','v9'),('ggd-v3','004','v9'),('ggd-v4','005','v10')]:
        config=a.run/(profile+'.json');config.write_text(json.dumps(dict(engine=ENGINES[environment],games=2,parallel_matches=2,
            seconds=a.seconds,seed=20261103,observation_profile=profile,teacher_version=teacher)))
        for label,exe in [('baseline',a.baseline),('candidate',a.candidate)]:
            run([exe,'collect-teacher',str(config),str(a.run/(profile+'-'+label))],profile+'-'+label)
        before=json.loads((a.run/(profile+'-baseline')/'dataset.json').read_text())
        after=json.loads((a.run/(profile+'-candidate')/'dataset.json').read_text())
        files=before['files'];assert len(files)==len(after['files'])
        equal=sum(x['sha256']==y['sha256'] for x,y in zip(files,after['files']))
        results.append(dict(kind='demonstrations',profile=profile,environment=environment,equal=equal,total=len(files),
                            baseline_hashes=[f['sha256'] for f in files],candidate_hashes=[f['sha256'] for f in after['files']]))
    for name,weights,environment,teacher in MODELS:
        for label,exe in [('baseline',a.baseline),('candidate',a.candidate)]:
            run([exe,'evaluate-ggd',ENGINES[environment],weights,str(a.run/(name+'-'+label)),
                 '--matches','2','--seconds',str(a.seconds),'--seed','20261103','--sampling','--teacher-version',teacher],name+'-'+label)
        before=json.loads((a.run/(name+'-baseline')/'report.json').read_text())
        after=json.loads((a.run/(name+'-candidate')/'report.json').read_text())
        results.append(dict(kind='model_evaluation',model=name,environment=environment,
                            equal=sum(x==y for x,y in zip(before['per_match'],after['per_match'])),total=len(before['per_match']),
                            per_match_equal=before['per_match']==after['per_match']))
    passed=all(r['equal']==r['total'] for r in results)
    report=dict(passed=passed,seed=20261103,seconds=a.seconds,results=results,commands=commands,
                baseline_sha256=hashlib.sha256(Path(a.baseline).read_bytes()).hexdigest(),candidate_sha256=hashlib.sha256(Path(a.candidate).read_bytes()).hexdigest())
    (a.run/'result.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(dict(passed=passed,results=results)))
    if not passed:raise SystemExit(1)


if __name__=='__main__':main()
