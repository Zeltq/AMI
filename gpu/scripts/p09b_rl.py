"""Sequential P09b stages after DAgger; selection is recorded before critic fitting."""
import argparse,os,sys
from pathlib import Path
from p09b_run import ROOT,RUN,REPORT,ENGINE,WEIGHTS,INITIAL,DATA,read,save,sha,rust_sources,linux,run,exclusive,status,publish
BINARY=RUN/'rl-gpu.exe'
WSL=['wsl','-d','Ubuntu','--','bash',linux(ROOT/'scripts/wsl_python.sh')]

def verified():
    seal=read(REPORT/'verified-gpu.json')
    assert seal['passed'] and seal['sources']==rust_sources(),'Native source needs verification'
    assert seal['executable_sha256']==sha(BINARY),'Verified executable changed'
def weights_equal(before,after,actor_only=False):
    before=Path(before);after=Path(after);a=read(before/'manifest.json')['tensors'];b=read(after/'manifest.json')['tensors'];assert set(a)==set(b)
    changed=[];checked=[]
    for name in a:
        if actor_only and name.startswith('value_'):continue
        checked.append(name)
        if (before/a[name]['file']).read_bytes()!=(after/b[name]['file']).read_bytes():changed.append(name)
    return dict(checked=len(checked),equal=len(checked)-len(changed),changed=changed)
def registry(model,opponent,report):
    path=RUN/'evaluations.json';entries=read(path) if path.exists() else []
    row=dict(model=model,opponent=opponent,report=str(report))
    if row not in entries:entries.append(row);save(path,entries)
def reference_evaluations():
    models=[('ZOg-e3',WEIGHTS),('ZOg-d1',RUN/'ZOg-d1/export/weights'),('ZOg-d2',RUN/'ZOg-d2/export/weights'),('P08-final',Path('C:/AMI/runs/p08/rev-main/checkpoint-000323/weights'))]
    for name,weights in models:
        for seed in (20261103,20261104):
            target=RUN/f'reward-{name}-{seed}'
            run(f'reward-{name}-{seed}','evaluation',[BINARY,'evaluate-ggd',ENGINE,weights,target,'--mode','teacher','--matches',64,'--seconds',600,'--seed',seed,'--sampling','--teacher-version','v10'],target/'report.json',True)
            r=read(target/'report.json');assert r['complete'] and r['matches']==64
            registry(name,'v10',target/'report.json')
        publish('publish reward evaluation '+name)
    for seed in (20261103,20261104):
        target=RUN/f'reward-v10-{seed}'
        run(f'reward-v10-{seed}','evaluation',[BINARY,'evaluate-teachers',ENGINE,target,'--matches',64,'--seconds',600,'--seed',seed,'--left','v10','--right','v10'],target/'report.json',True)
        r=read(target/'report.json');assert r['complete'] and r['matches']==64;registry('v10','v10',target/'report.json')
    publish('publish complete starting-model comparison')

def selection():
    selected=read(REPORT/'selection.json')
    assert selected['model'] in ('ZOg-e3','ZOg-d1','ZOg-d2') and selected['reason_ru']
    assert Path(selected['checkpoint']).is_file() and (Path(selected['weights'])/'manifest.json').exists()
    manifest=read(Path(selected['weights'])/'manifest.json')
    assert manifest['model_name']==selected['model'] and manifest['source_sha256']==sha(selected['checkpoint'])
    assert manifest['observation_profile']=='ggd-v4' and not manifest.get('timing_oracle',False)
    return selected
def critic():
    selected=selection();datasets=[DATA,RUN/'d1-data',RUN/'d2-data']
    # Eight recorded shards across game phases exercise the GPU cache/head path.
    smoke=RUN/'critic-smoke'
    run('critic-gpu-smoke','training',WSL+[linux(ROOT/'gpu/scripts/p09b_critic.py'),linux(selected['checkpoint']),linux(smoke),'--datasets',linux(DATA),'--epochs',1,'--smoke-files',8],smoke/'result.json',True)
    assert read(smoke/'result.json')['actor_unchanged']
    out=RUN/'critic'
    run('offline-critic','training',WSL+[linux(ROOT/'gpu/scripts/p09b_critic.py'),linux(selected['checkpoint']),linux(out),'--datasets']+[linux(p) for p in datasets]+['--epochs',8],out/'result.json',True)
    result=read(out/'result.json');assert result['actor_unchanged'];save(REPORT/'critic.json',result);save(REPORT/'critic-config.json',read(out/'config.json'))
    run('critic-export','export',WSL+[linux(ROOT/'gpu/scripts/weights.py'),'--checkpoint',linux(out/'best.pt'),'--output',linux(out/'export')],out/'export/weights/manifest.json',True)
    equal=weights_equal(selected['weights'],out/'export/weights',True);assert not equal['changed'];save(REPORT/'critic-policy-equality.json',equal)
    publish('publish critic validation and unchanged policy check')

def base_config():
    return dict(engine=ENGINE,initial_weights=str(RUN/'critic/export/weights'),matches=192,seed=20261109,minutes=480,max_blocks=0,batch_sequences=48,epochs=2,learning_rate=3e-5,critic_learning_rate=3e-4,warmup_blocks=100,warmup_min_blocks=20,entropy=.005,kl_start=.05,kl_end=.01,stop_kl=.02,checkpoint_minutes=30,evaluate=False,evaluation_minutes=120,initialize_weights=None,skip_warmup=True,critic_only=False,desynchronize=True,verify_phase_histogram=True,teacher_version='v10',reward_mode='potential-v1',mana_weight=.5,separate_critic_lr=True,resume_checkpoint=None,schedule_minutes=480,stop_at_minutes=0)
def native(name,config):
    target=RUN/name
    if (target/'config.json').exists():
        old=read(target/'config.json');assert all(old.get(k)==v for k,v in config.items()),'Existing native configuration differs'
    save(REPORT/(name+'.json'),config)
    run(name,'training',[BINARY,'train-ggd',REPORT/(name+'.json'),target],target/'result.json',True)
    return read(target/'result.json'),target
def latest(result):return Path(result['latest'])
def smoke():
    assert read(RUN/'critic/result.json')['actor_unchanged']
    cfg=base_config();cfg.update(minutes=10,learning_rate=0.,checkpoint_minutes=10)
    result,target=native('smoke-zero-lr',cfg)
    equal=weights_equal(cfg['initial_weights'],latest(result)/'weights');assert not equal['changed']
    import json
    episodes=[json.loads(line) for line in (target/'episodes.jsonl').read_text().splitlines() if line.strip()]
    assert episodes,'Smoke must contain completed matches'
    residual=[];components=[0]*5
    for episode in episodes:
        game=episode['game'];hero=next(h for h in game['sides'] if h['controller']=='GGD');side=game['sides'].index(hero)
        assert hero['reward_components'][0]==10*game['result']*(1 if side==0 else -1)
        residual+=list(map(abs,hero['shaping_telescoping_residual']))
        components=[n+int(abs(x)>1e-6) for n,x in zip(components,episode['components'])]
    assert max(residual)<.02
    smoke_check=dict(passed=True,weights=equal,completed_matches=len(episodes),components_nonzero_matches=components,max_shaping_telescoping_residual=max(residual),training_minutes=result['training_seconds']/60)
    save(REPORT/'smoke-check.json',smoke_check)
    progress=read(latest(result)/'progress.json');cfg.update(minutes=progress['training_seconds']/60+5,max_blocks=progress['block']+2,resume_checkpoint=str(latest(result)))
    resumed,resume_target=native('smoke-resume',cfg)
    first=json.loads((resume_target/'history.jsonl').read_text().splitlines()[0])
    expected=.05+(.01-.05)*min(1,progress['training_seconds']/(480*60))
    assert abs(first['learning']['kl_coefficient']-expected)<1e-7
    assert resumed['training_seconds']>progress['training_seconds'] and resumed['resumed_training_seconds']==progress['training_seconds']
    equal=weights_equal(cfg['initial_weights'],latest(resumed)/'weights');assert not equal['changed']
    save(REPORT/'resume-check.json',dict(passed=True,initial_training_seconds=progress['training_seconds'],resumed_training_seconds=resumed['resumed_training_seconds'],expected_first_kl_coefficient=expected,observed_first_kl_coefficient=first['learning']['kl_coefficient'],weights=equal))
    publish('publish zero-lr smoke and checkpoint resume checks')
def warmup():
    assert read(REPORT/'smoke-check.json')['passed'] and read(REPORT/'resume-check.json')['passed']
    cfg=base_config();cfg.update(minutes=120,max_blocks=100,skip_warmup=False,critic_only=True,checkpoint_minutes=30)
    result,target=native('critic-warmup',cfg)
    assert 20<=result['blocks']<=100
    equal=weights_equal(cfg['initial_weights'],latest(result)/'weights',True);assert not equal['changed']
    save(REPORT/'warmup.json',dict(result=result,actor_weights=equal,criterion_met=result['warmup_ev_last5'] is not None and result['warmup_ev_last5']>.3,maximum_blocks=100))
    publish('publish online critic warmup')
def main_training():
    assert read(REPORT/'smoke-check.json')['passed'] and read(REPORT/'resume-check.json')['passed']
    warm=read(REPORT/'warmup.json');assert not warm['actor_weights']['changed']
    cfg=base_config();cfg.update(initialize_weights=str(latest(warm['result'])/'weights'),evaluate=True)
    candidates=[];folders=[p for p in RUN.glob('rl-main-*') if p.is_dir()]
    for folder in folders:
        for progress_path in folder.glob('checkpoint-*/progress.json'):
            p=read(progress_path)
            if all((progress_path.parent/f).exists() for f in ('model.bpk','optimizer.bpk','weights/manifest.json')):candidates.append((p,progress_path.parent))
    if candidates:
        p,path=max(candidates,key=lambda pair:(pair[0]['training_seconds'],pair[0].get('last_evaluation',0)))
        if p['training_seconds']>=480*60:raise RuntimeError('480 minutes already trained: finish/audit final evaluations without more PPO')
        cfg['resume_checkpoint']=str(path)
    history=read(RUN/'commands.json') if (RUN/'commands.json').exists() else []
    names=[p.name for p in folders]+[r['stage'] for r in history if r['stage'].startswith('rl-main-')]
    attempt=max([int(n.rsplit('-',1)[-1]) for n in names if n.rsplit('-',1)[-1].isdigit()]+[0])+1
    name=f'rl-main-{attempt:03d}';result,target=native(name,cfg)
    assert result['training_seconds']>=480*60,'Training stopped before the requested time; resume checkpoint required'
    save(REPORT/'rl-final-run.json',result)
    publish('publish eight-hour RL result')

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['references','critic','smoke','warmup','main']);a=p.parse_args()
    lock=exclusive();os.environ['SF_DUEL_DATA_ROOT']='C:/AMI';verified()
    {'references':reference_evaluations,'critic':critic,'smoke':smoke,'warmup':warmup,'main':main_training}[a.stage]()
    status(dict(stage=a.stage+' complete; continue P09b protocol',pid=None,background_work_remaining=True))

if __name__=='__main__':main()
