"""Publish only actually completed checks, including failed preparation attempts."""
import json,re
from p09_run import RUN,REPORT,save

def main():
    history=json.loads((RUN/'commands.json').read_text())
    checks={}
    for name in ('rust-targeted-2','rust-tests'):
        text=(RUN/(name+'.stdout.log')).read_text(encoding='utf-8',errors='replace')
        match=re.search(r'test result: ok\. (\d+) passed; (\d+) failed',text)
        assert match and int(match[2])==0,name
        checks[name]={'passed':int(match[1]),'total':int(match[1]),'failed':0}
    text=(RUN/'python-tests.stderr.log').read_text(encoding='utf-8',errors='replace')
    match=re.search(r'Ran (\d+) tests',text);assert match and re.search(r'\nOK\s*$',text),text[-800:]
    checks['python-tests']={'passed':int(match[1]),'total':int(match[1]),'skipped':0}
    parity=json.loads((RUN/'validate-smoke.stdout.log').read_text());assert parity['passed']
    checks['pytorch_burn']=parity
    smoke=json.loads((RUN/'training-smoke/result.json').read_text())
    assert smoke['updates']==2 and smoke['value_head_delta']==0 and smoke['actor_delta']>0
    checks['training_smoke']={k:smoke[k] for k in ('updates','actor_delta','value_head_delta','first_action_head_gradient_l1','history')}
    manifest=json.loads((RUN/'smoke-export/weights/manifest.json').read_text())
    assert manifest['observation_profile']=='ggd-v4' and manifest['tensors']['hero.first.weight']['shape'][0]==172
    checks['export_profile']={'profile':manifest['observation_profile'],'hero_width':172,'timing_oracle':manifest['timing_oracle']}
    checks['failed_preparation_attempts']=[dict(stage=r['stage'],exit_code=r['exit_code'],command=r['command_literal']) for r in history if r.get('exit_code',0)!=0]
    checks['notes']=['First collection audit read a reset live arena; final outcome is now captured at terminal event.', 'Reward stacks use active debuff and exactly the observer clock arithmetic; original ggd-v4 profile is unchanged.', 'First targeted GPU test sampled state before observe updated fog proxy units; corrected test snapshots after observe.']
    save(REPORT/'verification.json',checks)

if __name__=='__main__':main()
