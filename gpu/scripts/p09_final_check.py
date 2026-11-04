"""Independent final audit of published P09 counts, files and selected artifacts."""
import hashlib,json,math
from pathlib import Path
from p09_run import RUN,REPORT,ROOT,save

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))

def main():
    rows=read(REPORT/'metrics.json');checks=0;matches=0
    for epoch in range(1,4):
        totals={side:dict(matches=0,lh=0,deny=0,wins=0,draws=0,losses=0,near=0,mid=0,far=0,opponent_near=0,opponent_mid=0,opponent_far=0) for side in ('radiant','dire')}
        for seed in (20261103,20261104):
            report=read(RUN/f'ZOg/epoch-{epoch:03d}/vs-teacher-{seed}/report.json')
            assert report['complete'] and len(report['per_match'])==64
            for game in report['per_match']:
                assert game['simulation_error']==0 and game['native_terminal']
                hero=next(s for s in game['sides'] if s['controller']=='GGD')
                opponent=next(s for s in game['sides'] if s['controller']=='teacher')
                out=totals[hero['side']];out['matches']+=1
                for key in ('lh','deny'):out[key]+=hero[key]
                result=game['result']*(1 if hero['side']=='radiant' else -1)
                out[{1:'wins',0:'draws',-1:'losses'}[result]]+=1
                for i,spell in enumerate(('near','mid','far')):
                    out[spell]+=hero['spell_completions'][i];out['opponent_'+spell]+=opponent['spell_completions'][i]
                assert sum(c['count'] for c in hero['death_causes'])==hero['deaths'];matches+=1
        for side,counts in totals.items():
            row=next(r for r in rows if r['model']==f'ZOg-e{epoch}' and r['side']==side)
            for key,value in counts.items():
                metric=key+'_casts' if key.endswith(('near','mid','far')) else key
                assert row[metric]==value,(epoch,side,key);checks+=1
        r,d,total=[next(r for r in rows if r['model']==f'ZOg-e{epoch}' and r['side']==s) for s in ('radiant','dire','total')]
        for key,value in total.items():
            if isinstance(value,(int,float)):
                assert math.isclose(value,r[key]+d[key],rel_tol=1e-6,abs_tol=1e-6),(epoch,key)
                checks+=1
    for seed in (20261103,20261104):
        baseline=read(RUN/f'teacher-{seed}/report.json');assert baseline['complete'] and len(baseline['per_match'])==64
    checkpoint=RUN/'ZOg/epoch-003.pt';export=RUN/'ZOg/epoch-003/weights';manifest=read(export/'manifest.json')
    checkpoint_sha=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert manifest['model_name']=='ZOg-e3' and manifest['source_sha256']==checkpoint_sha
    for tensor in manifest['tensors'].values():assert hashlib.sha256((export/tensor['file']).read_bytes()).hexdigest()==tensor['sha256']
    selection={'selected_by':'user','model':'ZOg-e3','checkpoint':checkpoint.as_posix(),'burn_export':export.as_posix(),'checkpoint_sha256':checkpoint_sha,'manifest_sha256':hashlib.sha256((export/'manifest.json').read_bytes()).hexdigest()}
    save(REPORT/'selection.json',selection)
    lines=(ROOT/'reports/P09a.md').read_text(encoding='utf-8').splitlines()
    assert len(lines)<=150 and lines[-1]=='DONE P09a'
    assert lines.count('Выбор пользователя: ZOg-e3')==1
    assert any(checkpoint.as_posix() in line for line in lines) and any(export.as_posix() in line for line in lines)
    files=[p for p in REPORT.iterdir() if p.is_file()]
    assert all(p.stat().st_size<1_000_000 for p in files)
    budget=read(REPORT/'budget.json');assert budget['used_minutes']<240
    save(REPORT/'final-check.json',{'candidate_matches_checked':matches,'candidate_matches_expected':384,'arithmetic_checks':checks,'teacher_baseline_matches':128,'selected_tensor_hashes_checked':len(manifest['tensors']),'report_lines':len(lines),'metric_files_checked':len(files),'maximum_metric_bytes':max(p.stat().st_size for p in files),'gpu_minutes':budget['used_minutes'],'command':'C:/Python314/python.exe gpu/scripts/p09_final_check.py'})
    print(json.dumps(read(REPORT/'final-check.json')))

if __name__=='__main__':main()
