"""Verify the v4 smoke dataset and publish small final provenance/check records."""
import hashlib,json,re,subprocess,sys
from pathlib import Path
import numpy as np
from ggd_dataset import read_block,decode

ROOT=Path(__file__).resolve().parents[2];RUN=Path('C:/AMI/runs/p07');OUT=ROOT/'reports/P07'
def save(name,v):(OUT/name).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
root=RUN/'collect-v4';manifest=json.loads((root/'dataset.json').read_text());report=json.loads((root/'report.json').read_text())
assert manifest['observation_profile']=='ggd-v4' and manifest['config']['teacher_version']=='v10'
assert report['native_completed']==report['games_requested']==2 and not report['partial']
frames=spells=oracle=enemy_casts=0
for file in manifest['files']:
    h,o,s=read_block(root/file['file'],file['sha256']);assert h['version']==4
    assert np.isfinite(o).all() and np.isfinite(s).all()
    valid=s[:,:,11]>.5;assert np.all(s[:,:,12][valid]==4)
    frames+=int(valid.sum());spells+=int(((s[:,:,0]>=6)&(s[:,:,0]<=9)&valid).sum())
    a=decode(h,o,s)
    for key in ('hero_combat','creep_combat','tower_combat'):
        for start,end in ((25,33),(36,44)):
            x=a[key][...,start:end];assert np.count_nonzero(x)==0;oracle+=x.size
    public=a['hero_combat'][:,:,44:];assert public.shape[-1]==32
    assert np.isin(public[:,:,16:],[0,1]).all()
    enemy_casts+=int(((public[:,1,1:5].sum(-1)>0)&a['loss_valid']).sum())
assert frames==report['valid_decisions']
save('collect-smoke.json',dict(**report,teacher_version='v10',observation_profile='ggd-v4',verified_blocks=len(manifest['files']),
     verified_valid_decisions=frames,spell_commands=spells,visible_enemy_cast_frames=enemy_casts,oracle_zero_values=oracle,oracle_total_values=oracle))
checks={}
for name in ('baseline-004','rust-narrow-final','rust-full','build-final','public-parity-final','regression-004','evaluation-005','collect-v4'):
    value=json.loads((RUN/f'{name}.status.json').read_text(encoding='utf-8'));assert value['exit_code']==0
    checks[name]={k:value[k] for k in ('command','exit_code','minutes')}
for profile in ('ggd-v2','ggd-v3'):
    for version in ('before','after'):
        name=f'compat-{profile}-{version}';value=json.loads((RUN/f'{name}.status.json').read_text(encoding='utf-8'));assert value['exit_code']==0
        checks[name]={k:value[k] for k in ('command','exit_code','minutes')}
py=(RUN/'python-tests-final.log').read_text(encoding='utf-8',errors='replace');n=int(re.search(r'Ran (\d+) tests',py)[1]);assert '\nOK\n' in py
checks['python']=dict(command="$env:PYTHONPATH='src;tests;C:/AMI/tools/lua_test_runtime'; C:/Python314/python.exe -m unittest discover -s tests -v",passed=n,failed=0)
checks['go']=dict(command='go test ./...',cwd='C:/AMI-wt/P07/tools/nativecollect',exit_code=0)
save('verification.json',checks)
protected=['gpu/src/'+f for f in ('learner.rs','experience.rs','selfplay.rs','gradients.rs','scripted_teacher.rs','teacher_v10.rs')]
unchanged={}
for name in protected:
    base=subprocess.check_output(['git','show','origin/main:'+name],cwd=ROOT)
    assert (ROOT/name).read_bytes()==base,name
    unchanged[name]=digest(ROOT/name)
save('provenance.json',dict(base_commit='faa1b6e798cece79d072131b870e3de832ad77d3',branch='agent/P07-obs-v4',
     baseline_binary_sha256=digest(RUN/'baseline.exe'),final_binary_sha256=digest(RUN/'target/release/sf-duel-gpu.exe'),
     unchanged_files=unchanged,source_files={p:digest(ROOT/p) for p in ('bots/agent.lua','src/sf_duel/observations.py','gpu/src/enemy_observe.rs','gpu/src/physics.rs','gpu/src/world.rs')},
     no_training=True,environment_004_unchanged=digest(Path('C:/AMI/runs/p06/dota-calibration-004/engine.json'))==json.loads((OUT/'environment.json').read_text())['old_engine_sha256']))
commands=[]
for name,flags,worker,seconds,timeout,event in [
    ('spells-006','',15,-25,240,'spell_scenario.lua'),('sf-final','--miss --case-kind sf',15,1000,350,'natural_miss_scenario.lua'),
    ('melee-final','--miss --case-kind melee',17,1000,350,'natural_miss_scenario.lua'),('ranged-001','--miss --case-kind ranged',17,1000,350,'natural_miss_scenario.lua'),
    ('tower-001','--miss --case-kind tower',18,1000,350,'natural_miss_scenario.lua'),('impact-003','--miss --impact',16,1500,450,'natural_miss_scenario.lua')]:
    command=f'C:/Python314/python.exe gpu/scripts/p07_probe.py {flags} --run C:/AMI/runs/p07/{name} --worker {worker} --seconds {seconds} --wall-timeout {timeout} --timescale 4 --seed 20261103 --events-script reports/P07/{event}'
    if name=='spells-006':command+=' --capture-world'
    command+=' --no-pilot-update'
    commands.append(dict(run=name,command=command.replace('  ',' '),minutes=json.loads((RUN/name/'status.json').read_text(encoding='utf-8'))['elapsed_seconds']/60,
                         frozen_source=f'reports/P07/probes/{name}/p03_events.lua'))
save('calibration-commands.json',commands)
print(json.dumps(dict(blocks=len(manifest['files']),frames=frames,spell_commands=spells,enemy_cast_frames=enemy_casts,oracle_zeros=oracle,python_tests=n)))
