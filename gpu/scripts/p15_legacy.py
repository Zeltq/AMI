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
    p.add_argument('--seconds',type=float,default=15);p.add_argument('--extended',action='store_true');p.add_argument('--reference-run',type=Path);a=p.parse_args()
    a.run.mkdir(parents=True,exist_ok=False);commands=[];results=[]
    reference=a.reference_run or a.run
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
            if label=='baseline' and a.reference_run:continue
            run([exe,'collect-teacher',str(config),str(a.run/(profile+'-'+label))],profile+'-'+label)
        before=json.loads((reference/(profile+'-baseline')/'dataset.json').read_text())
        after=json.loads((a.run/(profile+'-candidate')/'dataset.json').read_text())
        files=before['files'];assert len(files)==len(after['files'])
        equal=sum(x['sha256']==y['sha256'] for x,y in zip(files,after['files']))
        results.append(dict(kind='demonstrations',profile=profile,environment=environment,equal=equal,total=len(files),
                            baseline_hashes=[f['sha256'] for f in files],candidate_hashes=[f['sha256'] for f in after['files']]))
    for name,weights,environment,teacher in MODELS:
        for label,exe in [('baseline',a.baseline),('candidate',a.candidate)]:
            if label=='baseline' and a.reference_run:continue
            run([exe,'evaluate-ggd',ENGINES[environment],weights,str(a.run/(name+'-'+label)),
                 '--matches','2','--seconds',str(a.seconds),'--seed','20261103','--sampling','--teacher-version',teacher],name+'-'+label)
        before=json.loads((reference/(name+'-baseline')/'report.json').read_text())
        after=json.loads((a.run/(name+'-candidate')/'report.json').read_text())
        results.append(dict(kind='model_evaluation',model=name,environment=environment,
                            equal=sum(x==y for x,y in zip(before['per_match'],after['per_match'])),total=len(before['per_match']),
                            per_match_equal=before['per_match']==after['per_match']))
    if a.extended:
        config=a.run/'dagger.json';config.write_text(json.dumps({'collection':{'engine':ENGINES['005'],'games':2,'parallel_matches':2,'seconds':a.seconds,'seed':20261103,'observation_profile':'ggd-v4','teacher_version':'v10'},'weights':MODELS[3][1],'beta':1.,'round':1}))
        for label,exe in [('baseline',a.baseline),('candidate',a.candidate)]:
            if label=='baseline' and a.reference_run:continue
            run([exe,'collect-dagger',str(config),str(a.run/('dagger-'+label))],'dagger-'+label)
        before=json.loads((reference/'dagger-baseline/dataset.json').read_text())['files'];after=json.loads((a.run/'dagger-candidate/dataset.json').read_text())['files'];assert len(before)==len(after)
        results.append(dict(kind='legacy_dagger_beta1',equal=sum(x['sha256']==y['sha256'] for x,y in zip(before,after)),total=len(before)))
        config=a.run/'ppo.json';config.write_text(json.dumps({'engine':ENGINES['005'],'initial_weights':MODELS[3][1],'matches':2,'seed':20261103,'minutes':60,'max_blocks':1,'batch_sequences':2,'epochs':1,'skip_warmup':True,'warmup_blocks':0,'learning_rate':1e-5,'teacher_version':'v10','reward_mode':'p08','desynchronize':False,'evaluate':False}))
        hashes=[]
        for label,exe in [('baseline',a.baseline),('candidate',a.candidate)]:
            output=(reference if label=='baseline' else a.run)/('ppo-'+label)
            if label!='baseline' or not a.reference_run:run([exe,'train-ggd',str(config),str(output)],'ppo-'+label)
            latest=json.loads((output/'latest.json').read_text());weights=Path(latest['path'])/'weights';manifest=json.loads((weights/'manifest.json').read_text())
            hashes.append({name:hashlib.sha256((weights/value['file']).read_bytes()).hexdigest() for name,value in manifest['tensors'].items()})
        assert hashes[0].keys()==hashes[1].keys()
        results.append(dict(kind='legacy_ppo_one_block_weights',equal=sum(hashes[0][k]==hashes[1][k] for k in hashes[0]),total=len(hashes[0]),baseline_hashes=hashes[0],candidate_hashes=hashes[1]))
    passed=all(r['equal']==r['total'] for r in results)
    report=dict(passed=passed,reference_run=str(reference),seed=20261103,seconds=a.seconds,results=results,commands=commands,
                baseline_sha256=hashlib.sha256(Path(a.baseline).read_bytes()).hexdigest(),candidate_sha256=hashlib.sha256(Path(a.candidate).read_bytes()).hexdigest())
    (a.run/'result.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(dict(passed=passed,results=results)))
    if not passed:raise SystemExit(1)


if __name__=='__main__':main()
