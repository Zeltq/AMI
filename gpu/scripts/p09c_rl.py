"""Critic, zero-lr smoke, health-gated pilots, and resumable event-v2 PPO."""
import argparse,json,os,sys,time
from pathlib import Path
from p09c_run import ROOT,RUN,REPORT,ENGINE,INITIAL,WEIGHTS,EXE,WSL,read,save,sha,linux,run,exclusive,status,publish,history,used
from p09b_rl import weights_equal
from p09b_results import lines,summaries
from p09c_reward import NAMES

def note(text):
    path=ROOT/'reports/P09c.md';s=path.read_text(encoding='utf-8');s=s.replace('## Наблюдения','## Наблюдения\n\n- '+text,1);assert len(s.splitlines())<=150;path.write_text(s,encoding='utf-8',newline='\n')

def registry(name,other,path):
    file=RUN/'evaluations.json';rows=read(file) if file.exists() else [];row=dict(model=name,opponent=other,report=str(path))
    if row not in rows:rows.append(row);save(file,rows)

def evaluate(name,weights,other='v10'):
    reports=[]
    for seed in (20261103,20261104):
        target=RUN/f'eval-{name}-vs-{other}-{seed}'
        cmd=[EXE,'evaluate-ggd',ENGINE,weights,target,'--mode','teacher','--matches',64,'--seconds',600,'--seed',seed,'--sampling','--teacher-version','v10']
        if other!='v10':cmd+=['--opponent-weights',WEIGHTS]
        run(target.name,'evaluation',cmd,target/'report.json',True)
        r=read(target/'report.json');assert r['complete'] and r['matches']==64;reports+=r['per_match'];registry(name,other,target/'report.json')
    rows=summaries(name,{'per_match':reports},other,'event-v2');r=next(x for x in rows if x['side']=='total' and x['perspective']==0)
    assert r['matches']==128
    for side in ('radiant','dire'):assert next(x['matches'] for x in rows if x['side']==side and x['perspective']==0)==64
    save(REPORT/(name+'-vs-'+other+'-summary.json'),rows);publish('publish evaluation '+name+' vs '+other);return r

def references():
    evaluate('ZOg-d1',WEIGHTS)
    for seed in (20261103,20261104):
        target=RUN/f'eval-v10-vs-v10-{seed}'
        run(target.name,'evaluation',[EXE,'evaluate-teachers',ENGINE,target,'--matches',64,'--seconds',600,'--seed',seed,'--left','v10','--right','v10'],target/'report.json',True)
        registry('v10','v10',target/'report.json')
    publish('publish v10 reference with event rewards')

def critic():
    out=RUN/'critic'
    run('event-targets-and-feature-cache','audit',[sys.executable,ROOT/'gpu/scripts/p09c_cache.py',out],out/'cache-reuse.json')
    save(REPORT/'feature-cache-reuse.json',read(out/'cache-reuse.json'))
    for index,name in enumerate(('demo','d1','d2')):save(REPORT/(name+'-return-targets.json'),{k:v for k,v in read(out/f'targets-{index}/targets.json').items() if k!='files'})
    # A failed fit keeps last.pt and optimizer; the same output resumes epochs.
    attempts=sum(r['stage'].startswith('critic-fit-') for r in history())
    while not (out/'result.json').exists():
        attempts+=1
        try:run(f'critic-fit-{attempts:03}','training',WSL+[linux(ROOT/'gpu/scripts/p09b_critic.py'),linux(INITIAL),linux(out),'--datasets']+[linux(RUN/(n+'-data')) for n in ('demo','d1','d2')]+['--epochs',8,'--reward-mode','event-v2'],out/'result.json',True)
        except Exception:
            if used()>=300:raise
            if not (out/'last.pt').exists():raise
            note('**Подтверждено:** сбой critic-fit; продолжается последний сохранённый epoch/Adam, расход не сброшен.');continue
    r=read(out/'result.json');assert r['actor_unchanged'];save(REPORT/'critic.json',r)
    run('critic-export','export',WSL+[linux(ROOT/'gpu/scripts/weights.py'),'--checkpoint',linux(out/'best.pt'),'--output',linux(out/'export')],out/'export/weights/manifest.json',True)
    equal=weights_equal(WEIGHTS,out/'export/weights',True);assert not equal['changed'];save(REPORT/'critic-policy-equality.json',equal)
    val=r['selected_validation'];note(f'**Подтверждено:** ZOg-d1-c2, эпоха {r["selected_epoch"]}/8; политика {equal["equal"]}/{equal["checked"]} тензоров неизменна. Validation pregame EV={val["pregame"]["ev"]:.6f}, N={val["pregame"]["n"]}; post0 EV={val["post0"]["ev"]:.6f}, N={val["post0"]["n"]}.')
    publish('publish event critic validation')

def base():
    return dict(engine=ENGINE,initial_weights=str(WEIGHTS),initialize_weights=str(RUN/'critic/export/weights'),matches=192,seed=20261109,minutes=40,max_blocks=0,batch_sequences=48,epochs=2,learning_rate=3e-5,critic_learning_rate=3e-4,warmup_blocks=100,warmup_min_blocks=20,entropy=.005,kl_start=.2,kl_end=.2,stop_kl=.02,checkpoint_minutes=30,evaluate=False,evaluation_minutes=60,skip_warmup=True,critic_only=False,desynchronize=True,verify_phase_histogram=True,teacher_version='v10',reward_mode='event-v2',mana_weight=.5,separate_critic_lr=True,resume_checkpoint=None,schedule_minutes=40,stop_at_minutes=0,kl_decay_start_minutes=0,begin_decay_on_resume=False)

def checkpoints(prefix):
    found=[]
    for folder in RUN.glob(prefix+'-*'):
        for path in folder.glob('checkpoint-*/progress.json'):
            if all((path.parent/f).is_file() for f in ('model.bpk','optimizer.bpk','weights/manifest.json')):found.append((read(path),path.parent))
    return sorted(found,key=lambda r:(r[0]['training_seconds'],r[0].get('last_evaluation',0)))

def native(prefix,cfg):
    completed=[r for r in history() if r['stage'].startswith(prefix+'-') and r.get('exit_code')==0]
    if completed:
        last=completed[-1];result=read(last['artifact'])
        if result['training_seconds']>=cfg['minutes']*60 or cfg.get('critic_only'):return result,Path(last['artifact']).parent
    while used()<300:
        candidates=checkpoints(prefix)
        if candidates:
            p,path=candidates[-1];cfg={**cfg,'resume_checkpoint':str(path)}
            if p['training_seconds']>=cfg['minutes']*60:
                # A crash during final evaluation leaves a valid final checkpoint.
                return dict(training_seconds=p['training_seconds'],latest=str(path),recovered_final_checkpoint=True),path.parent
        attempt=1+sum(r['stage'].startswith(prefix+'-') for r in history());name=f'{prefix}-{attempt:03}';target=RUN/name
        save(REPORT/(name+'.json'),cfg)
        try:
            run(name,'training',[EXE,'train-ggd',REPORT/(name+'.json'),target],target/'result.json',True)
            return read(target/'result.json'),target
        except Exception as exc:
            save(RUN/'last-training-failure.json',dict(run=name,error=str(exc),checkpoints=[str(x[1]) for x in checkpoints(prefix)]))
            if used()>=300:raise
            # Let the heartbeat diagnose a repeatable pre-checkpoint failure.
            if not checkpoints(prefix):raise
            note(f'**Подтверждено:** {name} завершился с ошибкой; повтор с последнего полного checkpoint, GPU-время сохранено.')
    raise RuntimeError('GPU budget exhausted')

def smoke():
    if (REPORT/'smoke-check.json').exists():assert read(REPORT/'smoke-check.json')['passed'];return
    cfg=base();cfg.update(minutes=5,learning_rate=0.,checkpoint_minutes=5,desynchronize=False)
    result,folder=native('smoke-lr0',cfg);equal=weights_equal(cfg['initialize_weights'],Path(result['latest'])/'weights');assert not equal['changed']
    episodes=lines(folder/'episodes.jsonl');counts=[sum(abs(r['components'][i])>1e-6 for r in episodes) for i in range(6)]
    assert episodes and all(counts),(len(episodes),counts)
    for r in episodes:
        h=next(x for x in r['game']['sides'] if x['controller']=='GGD');c=r['components']
        assert abs(c[1]-.5*h['lh'])<1e-3 and abs(c[2]-.2*h['deny'])<1e-3
        assert c[3]<=0 and c[4]>=0 and c[5]<=0
    updates=lines(folder/'history.jsonl');assert all(0<=r['learning']['clip_fraction']<=1 for r in updates)
    proof=dict(passed=True,weights=equal,matches=len(episodes),component_names=NAMES,nonzero_matches=counts,training_minutes=result['training_seconds']/60,max_clip_fraction=max(r['learning']['clip_fraction'] for r in updates))
    save(REPORT/'smoke-check.json',proof);note(f'**Подтверждено:** smoke lr0 {proof["training_minutes"]:.3f} мин; веса {equal["equal"]}/{equal["checked"]} неизменны; ненулевые компоненты {counts}, N={len(episodes)}.');publish('publish five-minute zero-lr smoke')

def warmup():
    if (REPORT/'warmup.json').exists():return read(REPORT/'warmup.json')
    cfg=base();cfg.update(minutes=30,max_blocks=100,skip_warmup=False,critic_only=True,desynchronize=True)
    result,folder=native('warmup',cfg);equal=weights_equal(WEIGHTS,Path(result['latest'])/'weights',True);assert not equal['changed']
    proof=dict(result=result,actor_weights=equal,criterion_met=result.get('warmup_ev_last5') is not None and result['warmup_ev_last5']>.3)
    save(REPORT/'warmup.json',proof);note(f'**Подтверждено:** прогрев {result.get("blocks")} блоков, EVpost0 last5={result.get("warmup_ev_last5")}; политика {equal["equal"]}/{equal["checked"]} неизменна. При лимите 100 блоков продолжение разрешено протоколом; лимит принят как в P09b.');publish('publish critic warmup');return proof

def healthy(r):
    farm=r['lh']+r['deny'];spawn=r['lane_creeps_spawned']
    return spawn>0 and farm*10058*5>=4*1401*spawn and r['early_deaths']<=24 and r['matches']==128

def pilots():
    warm=warmup()
    for index,lr,anchor in ((1,3e-5,.2),(2,1e-5,.5)):
        proofpath=REPORT/f'pilot{index}-health.json'
        if proofpath.exists():proof=read(proofpath)
        else:
            cfg=base();cfg.update(initialize_weights=str(Path(warm['result']['latest'])/'weights'),learning_rate=lr,kl_start=anchor,kl_end=anchor)
            result,folder=native(f'rl-pilot{index}',cfg);r=evaluate(f'pilot{index}',Path(result['latest'])/'weights')
            proof=dict(healthy=healthy(r),reference_farm=1401,reference_spawn=10058,minimum_reference_fraction=.8,maximum_early_deaths=24,summary=r,result=result,actor_lr=lr,anchor=anchor)
            save(proofpath,proof);note(f'**Подтверждено:** pilot{index}: farm {r["lh"]+r["deny"]:.0f}/{r["lane_creeps_spawned"]:.0f}, смерти до120с {r["early_deaths"]}/128; проверка пилота={proof["healthy"]}.');publish(f'publish pilot{index} health gate')
        if proof['healthy']:return proof
    return None

def main_training(pilot):
    planpath=RUN/'main-plan.json'
    if planpath.exists():plan=read(planpath)
    else:
        start=pilot['result']['training_seconds']/60
        eval_minutes=sum(r['minutes'] for r in history() if r['kind']=='evaluation' and r['stage'].startswith('eval-pilot'))
        # Reserve final 35 min, desynchronization 8 min, and an hour evaluation
        # only if it can be reached. Estimate two opponents from pilot throughput.
        available=300-used()-35-8-2
        hour_eval=2*eval_minutes if available>=60+2*eval_minutes else 0
        duration=max(.1,available-hour_eval)
        plan=dict(start_training_minutes=start,end_training_minutes=start+duration,additional_training_minutes=duration,gpu_used_at_plan=used(),final_evaluation_reserve=35,desynchronization_reserve=8,overhead_reserve=2,hour_evaluation_estimate=hour_eval,anchor_start=pilot['anchor'])
        save(planpath,plan);save(REPORT/'main-plan.json',plan)
    cfg=base();cfg.update(minutes=plan['end_training_minutes'],schedule_minutes=plan['end_training_minutes'],kl_decay_start_minutes=plan['start_training_minutes'],begin_decay_on_resume=True,kl_start=pilot['anchor'],kl_end=.05,learning_rate=pilot['actor_lr'],resume_checkpoint=pilot['result']['latest'],initialize_weights=None,evaluate=bool(plan['hour_evaluation_estimate']),evaluation_minutes=60)
    result,folder=native('rl-main',cfg);save(REPORT/'main-result.json',result)
    # Reuse completed in-process final evaluations, or complete them after a crash.
    checkpoint=Path(result['latest']);block=read(checkpoint/'progress.json')['block']
    for other,tag in [('v10','v10'),('ZOg-d1','previous')]:
        paths=[folder/f'eval-{block:06}-{tag}-{seed}/report.json' for seed in (20261103,20261104)]
        if all(p.exists() and read(p).get('complete') for p in paths):
            for p in paths:registry('RL-final',other,p)
            publish('publish final '+other)
        else:evaluate('RL-final',checkpoint/'weights',other)
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['critic','smoke','pilots','main','all','references']);a=p.parse_args();lock=exclusive();os.environ['SF_DUEL_DATA_ROOT']='C:/AMI'
    assert sha(EXE)==read(REPORT/'verified.json')['exe_sha256'],'Executable changed'
    try:
        if a.stage in ('critic','all'):critic()
        if a.stage in ('smoke','all'):smoke()
        if a.stage=='references':references()
        if a.stage in ('pilots','main','all'):
            pilot=pilots()
            if pilot is None:status(dict(stage='both pilots failed health gate; finalize report',background_work_remaining=True,pid=None));return
            if a.stage in ('main','all'):main_training(pilot)
        status(dict(stage=a.stage+' complete; audit and finalize remaining work',background_work_remaining=True,pid=None))
    except Exception as exc:status(dict(stage='failed',error=str(exc),background_work_remaining=True));raise

if __name__=='__main__':main()
