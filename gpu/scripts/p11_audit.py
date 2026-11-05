"""Input immutability and P11 result checks, without storing model bytes in git."""
import subprocess
from pathlib import Path
from p11_run import ROOT,RUN,REPORT,START,POOL,read,save,sha
from p09b_results import lines
def prepare():
    files=[]
    for weights in POOL:
        files.extend(Path(weights).glob('*.f32'));files.append(Path(weights)/'manifest.json')
    files.extend(START/k for k in ('model.bpk','optimizer.bpk','progress.json'))
    source=subprocess.check_output(['git','ls-files','gpu/src'],cwd=ROOT,text=True).splitlines()
    files.extend(ROOT/p for p in source if p not in ('gpu/src/ggd_ppo.rs','gpu/src/ggd_pool.rs','gpu/src/ggd_arenas.rs','gpu/src/main.rs'))
    save(REPORT/'protected.json',{str(p):sha(p) for p in files})
def finish():
    initial=read(REPORT/'protected.json');changed=[p for p,h in initial.items() if sha(p)!=h];assert not changed,changed
    checks=[]
    for folder in RUN.glob('train-*'):
        if not (folder/'config.json').is_file():continue
        cfg=read(folder/'config.json');rows=lines(folder/'history.jsonl')
        for r in rows:
            assert 0<=r['learning']['clip_fraction']<=1
            # The coefficient belongs to the beginning of a block.
            before=r['training_seconds']-r['collection_seconds']-r['optimization_seconds']
            expected=.03+(.01-.03)*min(1,max(0,before/(cfg['schedule_minutes']*60)))
            assert abs(r['learning']['kl_coefficient']-expected)<1e-6
        checks.append(dict(run=folder.name,blocks=len(rows),clip_and_anchor_checks=len(rows)))
    save(REPORT/'audit.json',dict(protected_files=len(initial),unchanged=len(initial),changed=changed,training=checks,scope='Original weights, Adam, critic, teachers, physics and observation sources; PPO clipping and continuous anchor schedule. Evaluation reward and outcomes checked by p11_report.'))

def final_check():
    """Audit the completed packet, including optional matched-seed baseline."""
    import array,math
    from datetime import datetime,timezone
    from p11_report import refresh
    done=read(RUN/'completion.json');assert done['stage']=='complete'
    finish();refresh()
    entries=[e for e in read(REPORT/'evaluations.json') if not e.get('historical')]
    groups={};models={}
    for e in entries:
        report=read(e['report']);cfg=e['config'];games=report['per_match']
        assert report['complete'] and len(games)==64 and report['matches']==64
        assert sorted(g['game'] for g in games)==list(range(64))
        assert cfg['matches']==64 and cfg['seconds']==600 and not cfg['argmax'] and not cfg['zero_previous']
        assert cfg['teacher_version']==('v9' if e['opponent']=='v9' else 'v10')
        assert read(Path(cfg['weights'])/'manifest.json')['observation_profile']=='ggd-v4'
        key=(e['model'],e['opponent']);groups.setdefault(key,[]).append(e['seed'])
        models.setdefault(e['model'],set()).add(str(Path(cfg['weights'])))
        sides=[g['sides'][g['game']%2]['side'] for g in games]
        assert sides.count('radiant')==32 and sides.count('dire')==32
        assert all(g['simulation_error']==0 and g['result'] in [-1,0,1] and g['outcome_reason'] in ['death','T1','LH600'] for g in games)
        if e['opponent'] not in ('v10','v9'):
            op=Path(read(Path(e['report']).parent/'opponent.json')['weights'])
            if e['model']=='2h' or e['opponent']=='000470':assert op==START/'weights'
            elif e['model']=='4h':assert op==Path(next(x['config']['weights'] for x in entries if x['model']=='2h'))
            else:assert op==Path(read(REPORT/'final-reference.json')['reference'])/'weights'
    assert all(sorted(seeds)==[20261103,20261104] for seeds in groups.values())
    assert all(len(paths)==1 for paths in models.values())
    assert models['final']=={str(Path(done['checkpoint'])/'weights')}
    assert {('2h','v10'),('2h','v9'),('2h','000470'),('4h','v10'),('4h','v9'),('4h','2.00h'),('final','v10'),('final','v9'),('final','2.35h'),('final','000470')}<=groups.keys()
    optional=('000470','v9') in groups
    assert len(groups)==10+int(optional)
    pool=read(REPORT/'pool.json');assert len(pool)==7
    assert all(60*i<=p['training_minutes']<60*i+1 for i,p in enumerate(pool[3:],1))
    tensor_count=0
    for weights in [Path(p['weights']) for p in pool]+[Path(done['checkpoint'])/'weights']:
        manifest=read(weights/'manifest.json');assert len(manifest['tensors'])==59
        for t in manifest['tensors'].values():
            data=(weights/t['file']).read_bytes();assert len(data)==4*math.prod(t['shape'])
            values=array.array('f');values.frombytes(data);assert all(map(math.isfinite,values))
            tensor_count+=1
    frozen=[]
    for p in REPORT.glob('evaluation-freeze-*.json'):
        value=read(p);assert all(sha(f)==h for f,h in value['files'].items())
        value.update(last_checked_utc=datetime.now(timezone.utc).isoformat(),unchanged_after_evaluation=len(value['files']));save(p,value)
        frozen.append(dict(checkpoint=value['checkpoint'],files=len(value['files']),scope=value['scope']))
    native=[r for r in read(RUN/'commands.json') if r['kind'] in ('training','evaluation')]
    assert all(r['exit_code']==0 for r in native)
    summary=dict(checked_utc=datetime.now(timezone.utc).isoformat(),new_evaluation_matches=64*len(entries),series_128=len(groups),baseline_v9_matches=128 if optional else 0,balanced_sides=True,sampling=True,simulation_errors=0,pool_snapshots=len(pool),finite_tensor_files=tensor_count,immutable_checkpoints=frozen,native_jobs_completed=len(native),native_job_failures=0,checkpoint=done['checkpoint'],final_reference_age_minutes=read(REPORT/'final-reference.json')['age_minutes'],protected=read(REPORT/'audit.json'))
    assert abs(summary['final_reference_age_minutes']-120)<1
    save(REPORT/'final-audit.json',summary)
    refresh()
    return summary
if __name__=='__main__':
    import sys
    prepare() if sys.argv[1]=='prepare' else final_check() if sys.argv[1]=='final' else finish()
