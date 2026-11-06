"""Ordered user revision: sampling, diagnostic, weighted IL, selection, two DAgger rounds."""
from pathlib import Path

def choose_start(baseline,weighted,weights):
    def farm(summary):
        c=summary['controllers']['CHiO']
        return dict(numerator=c['lh']['numerator']+c['deny']['numerator'],denominator=c['actors'])
    a,b=farm(baseline),farm(weighted)
    use_weighted=b['numerator']*a['denominator']>=a['numerator']*b['denominator']
    chosen=b if use_weighted else a;dagger=list(weights) if use_weighted else [1.]*13
    if use_weighted:dagger[4]=1.
    return dict(start='nav-imitation-w' if use_weighted else 'nav-imitation',baseline_farm=a,weighted_farm=b,dagger_class_weights=dagger,below_three=chosen['numerator']<3*chosen['denominator'],rule='weighted >= baseline LH+deny per match, v10 sampling only; equality selects weighted; continue regardless of <3')

def run(r):
    def step(number,**fields):
        result=dict(step=number,finished_utc=r.utc(),**fields)
        r.save(r.REPORT/f'sampling-plan-step{number}.json',result)
        r.save(r.RUN/'sampling-plan.json',dict(active=True,**result))
        r.publish(f'step {number} sampling and weighted imitation plan')
    teacher=r.cached('teacher-1024');preserved=r.cached('nav-dagger-1-256')
    assert preserved['result']['native_completed']==256
    r.wait('nav-dagger-1-256-audit',r.REPORT/'nav-dagger-1-256-audit.json')
    baseline=r.evaluation('nav-imitation',r.RUN/'nav-imitation/weights','v10')
    if not (r.REPORT/'sampling-plan-step2.json').exists():step(2,evaluation=baseline)
    name='nav-imitation-postmid-audit';folder=r.RUN/name
    if not r.cached(name):
        result=r.run_job(name,r.torch_command('p16_action_audit.py',teacher['folder'],r.RUN/'nav-imitation/checkpoint.pt',folder),folder/'result.json')
        assert result['frames']==555778
        r.save(r.REPORT/(name+'.json'),result);r.done(name,result,folder)
    if not (r.REPORT/'sampling-plan-step3.json').exists():step(3,diagnostic=str(r.REPORT/(name+'.json')),selection_uses_diagnostic=False)
    zog=r.read(r.ROOT/'reports/P09a/training-config.json')
    weights=[1,1,4,4,4,4,*[zog['action_weights'][str(i)] for i in (6,7,8)],1,1,1,1]
    cfg=r.read(r.REPORT/'configs/nav-imitation.json');cfg.update(class_weights=weights)
    for key in ('initialize','resume','updates'):cfg.pop(key,None)
    weighted=r.supervised('nav-imitation-w',cfg)
    r.parity('parity-nav-imitation-w',teacher['folder'],weighted['folder'])
    if not (r.REPORT/'sampling-plan-step4.json').exists():step(4,weights=weights,initialization='fresh random',epochs=3,zero_previous=False,parity='parity-nav-imitation-w')
    r.evaluate_group('nav-imitation-w',Path(weighted['folder'])/'weights')
    if not (r.REPORT/'sampling-plan-step5.json').exists():step(5,evaluations=['nav-imitation-w/v10/sampling','nav-imitation-w/v9/sampling'])
    evaluated=r.evaluation('nav-imitation-w',Path(weighted['folder'])/'weights','v10')
    decision=choose_start(r.read(r.ROOT/baseline['summary']),r.read(r.ROOT/evaluated['summary']),weights)
    if not (r.REPORT/'sampling-plan-step6.json').exists():step(6,**decision)
    previous=weighted if decision['start']=='nav-imitation-w' else r.cached('nav-imitation')
    start=previous;older=[str(Path(previous['folder'])/'weights')]
    for i in (1,2):
        name=f'sampling-dagger-{i}'
        cfg=r.read(r.REPORT/f'configs/dagger-{i}-collect.json');cfg['collection'].update(engine=r.current_engine(),parallel_matches=256);cfg['weights']=str(Path(previous['folder'])/'weights')
        data=r.collect(name+'-256',cfg,dagger=True)
        cfg=r.read(r.REPORT/f'configs/dagger-{i}-train.json')
        cfg.update(engine=r.current_engine(),exclude_premid=True,dataset=data['folder'],replay_datasets=[teacher['folder'],preserved['folder']],initialize=str(Path(previous['folder'])/'checkpoint.pt'),class_weights=decision['dagger_class_weights'])
        previous=r.supervised(name+'-il',cfg)
        r.wait(name+'-256-audit',r.REPORT/(name+'-256-audit.json'))
        r.parity('parity-'+name,data['folder'],previous['folder']);r.evaluate_group(name,Path(previous['folder'])/'weights');older.append(str(Path(previous['folder'])/'weights'))
    best=r.choose_best({'nav-imitation','nav-imitation-w','sampling-dagger-1','sampling-dagger-2'})
    candidate=next(row for row in r.read(r.RUN/'stages.json').values() if Path(row['folder'])==Path(best['weights']).parent)
    r.save(r.REPORT/'pre-rl-selection.json',dict(best=best['label'],weights=best['weights'],reason=best['reason'],continue_rl_even_if_farm_remains_weak=True))
    r.publish('step 7 DAgger complete; proceed to critic and RL')
    r.finish_rl(candidate,start,older)
