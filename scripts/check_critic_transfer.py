"""One predeclared independent critic check; never updates actor parameters."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from selfplay_cycle import ROOT,Config,atomic,collect,linux
from run_mia_autonomous import prepare_parts,available_dir


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--execute',action='store_true');p.add_argument('--resume-collection',action='store_true');mode=p.add_mutually_exclusive_group();mode.add_argument('--extra-training',action='store_true');mode.add_argument('--extra-validation',action='store_true');a=p.parse_args()
    if not a.execute:raise SystemExit('Explicit --execute required')
    base=Path(a.run).resolve();cfg=Config.read(base/'config.json');initial=base/'pilot/behavior.pt';digest=hashlib.sha256(initial.read_bytes()).hexdigest()
    def status(phase,**extra):atomic(base/'diagnosis-status.json',{'phase':phase,'pid':os.getpid(),'updated_utc':datetime.now(timezone.utc).isoformat(),**extra})
    target=base/('critic-extra-validation' if a.extra_validation else 'critic-extra-training' if a.extra_training else 'critic-independent-check');count=32 if a.extra_training or a.extra_validation else 8
    collection_parts=[]
    if a.resume_collection:
        if not a.extra_validation:raise ValueError('Resume is currently supported only for expanded validation')
        for source in sorted(base.glob('critic-extra-validation*/collection.json')):
            prior=json.loads(source.read_text())
            if prior.get('quarantine') and not (source.parent/'quality-repair.json').exists():raise RuntimeError('Diagnose transport failure before refill')
            collection_parts.append({'path':str(source.parent),'matches':prior['matches']})
        target=available_dir(base,'critic-extra-validation')
    completed=sum(len(p['matches']) for p in collection_parts);assert completed<count
    status('collecting_extra_critic_training' if a.extra_training else 'collecting_independent_critic_validation',path=str(target),target=count,completed_prior=completed)
    new_games=collect(initial,target,cfg,matches=count-completed)
    collection_parts.append({'path':str(target),'matches':new_games});games=[m for part in collection_parts for m in part['matches']]
    if len(games)!=count or json.loads((target/'collection.json').read_text()).get('quarantine'):raise RuntimeError('Critic collection has transport issues; do not train')
    accepted=[]
    for source in [base/'pilot/collection/collection.json',*sorted(base.glob('pilot-extra*/collection.json'))]:
        meta=json.loads(source.read_text());accepted.append({'path':str(source.parent),'matches':meta['matches']})
    assert sum(len(p['matches']) for p in accepted)==32
    pilot_path=base/'pilot-dataset-all-onpolicy';train_path=base/('critic-expanded-training-dataset' if a.extra_training or a.extra_validation else 'pilot-dataset-all-onpolicy');val_path=base/('critic-validation40-dataset' if a.extra_validation else 'critic-independent-dataset')
    status('preparing_critic_check')
    train_parts=accepted+[{'path':str(target),'matches':games}] if a.extra_training else accepted
    train=json.loads((train_path/'dataset.json').read_text()) if a.extra_validation else prepare_parts(train_parts,train_path,digest)
    if a.extra_validation:
        prior=base/'critic-independent-check';old=json.loads((prior/'collection.json').read_text());assert len(old['matches'])==8 and not old.get('quarantine')
        val=prepare_parts([{'path':str(prior),'matches':old['matches']},*collection_parts],val_path,digest)
        assert train['matches']==64 and val['matches']==40 and train['policy_sha256']==digest
    else:
        val=json.loads((val_path/'dataset.json').read_text()) if a.extra_training else prepare_parts([{'path':str(target),'matches':games}],val_path,digest)
    if a.extra_training:
        assert val['matches']==8 and val['policy_sha256']==digest
        assert (pilot_path/'dataset.json').is_file()
    view=base/('critic-validation40-view' if a.extra_validation else 'critic-expanded-view' if a.extra_training else 'critic-independent-view');view.mkdir();entries=[]
    for source,meta,split in [(train_path,train,'train'),(val_path,val,'validation')]:
        for e in meta['episodes']:entries.append({**e,'path':os.path.relpath(source/e['path'],view).replace('\\','/'),'split':split})
    assert {e['match'] for e in entries if e['split']=='train'}.isdisjoint({e['match'] for e in entries if e['split']=='validation'})
    atomic(view/'dataset.json',{'policy_sha256':digest,'episodes':entries,'matches':104 if a.extra_validation else 72 if a.extra_training else 40,'gamma':1,'complete_episodes':True,'critic_pretraining_only':True,'validation_matches':40 if a.extra_validation else 8})
    for folder in (train_path,val_path):subprocess.run(['compact.exe','/C','/I','/Q','/S:'+str(folder),'*.npy'],stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT,check=True)
    output=base/('critic-validation40-pretrain' if a.extra_validation else 'critic-expanded-pretrain' if a.extra_training else 'critic-independent-pretrain');command=['wsl','-d','Ubuntu','--','bash','/mnt/c/AMI/scripts/wsl_python.sh',linux(ROOT/'scripts/pretrain_value.py'),'--checkpoint',linux(initial),'--dataset',linux(view),'--config',linux(base/'config.json'),'--output',linux(output)]
    with (base/(output.name+'.log')).open('wb') as log:
        child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW);status('pretraining_with_independent_validation',child_pid=child.pid,path=str(output));code=child.wait()
    if code:status('independent_critic_gate_failed',path=str(output));raise RuntimeError('Independent critic validation failed; actor remains frozen')
    result=json.loads((output/'complete.json').read_text());assert result['validation_passed'] and result['actor_unchanged']
    atomic(base/'critic-pretrain-reference.json',{'path':str(output),'reason':'Independent fresh MIA0 validation after failed archive transfer'})
    atomic(base/'pilot-data-reference.json',{'path':str(pilot_path),'parts':accepted,'note':'Original 32 on-policy pilot games; extra critic training and validation games are not added to PPO'})
    command=[sys.executable,str(ROOT/'scripts/run_mia_restart.py'),'--execute','--run',str(base)]
    with (base/'coordinator-recovery2.log').open('ab') as log,(base/'coordinator-recovery2-error.log').open('ab') as error:
        coordinator=subprocess.Popen(command,stdout=log,stderr=error,creationflags=subprocess.CREATE_NO_WINDOW)
    status('critic_passed_coordinator_resumed',coordinator_pid=coordinator.pid,result=result)
    p=ROOT/'runs/pilot-status.json';state=json.loads(p.read_text());state.update(mia_pid=coordinator.pid,mia_log=str(base/'coordinator-recovery2.log'),mia_error_log=str(base/'coordinator-recovery2-error.log'),stage='restart_after_independent_critic_validation');atomic(p,state)


if __name__=='__main__':main()
