"""Read-only final audit of P09c checkpoints, evaluations, budget and source scope."""
import json,subprocess
from pathlib import Path
from p09c_run import RUN,REPORT,WEIGHTS,read,save,sha,history,used,pid_alive
from p09b_results import lines,summaries

def main():
    assert not pid_alive(13760) and not pid_alive(27244),'P09c controller/native process remains alive'
    jobs=history();assert all(r.get('exit_code') is not None for r in jobs)
    assert all(r['exit_code']==0 for r in jobs if r.get('gpu_budget'))
    assert used()<=300
    result=read(RUN/'rl-main-001/result.json');cfg=read(RUN/'rl-main-001/config.json');checkpoint=Path(result['latest'])
    audit=read(REPORT/'final-checkpoint-audit.json');assert str(checkpoint)==audit['checkpoint']
    assert sha(checkpoint/'model.bpk')==audit['model_sha256'];assert sha(checkpoint/'optimizer.bpk')==audit['optimizer_sha256']
    manifest=read(checkpoint/'weights/manifest.json')
    assert manifest['observation_profile']=='ggd-v4'
    for row in audit['tensors']:assert sha(checkpoint/'weights'/manifest['tensors'][row['name']]['file'])==row['sha256']
    progress=read(checkpoint/'progress.json');assert progress['training_seconds']==result['training_seconds'] and progress['last_evaluation']==result['training_seconds']
    assert abs(progress['kl_coefficient_next']-.05)<1e-6
    assert result['resumed_training_seconds']==read(RUN/'rl-pilot1-001/result.json')['training_seconds']
    assert result['training_seconds']-result['resumed_training_seconds']>=3600
    assert cfg['initial_weights']==str(WEIGHTS) and progress['anchor_sha256']==sha(WEIGHTS/'manifest.json')
    assert read(REPORT/'critic-policy-equality.json')['equal']==55 and read(REPORT/'smoke-check.json')['weights']['equal']==59
    assert read(REPORT/'pilot1-health.json')['healthy']
    all_updates=lines(RUN/'rl-pilot1-001/history.jsonl')+lines(RUN/'rl-main-001/history.jsonl')
    assert len(all_updates)==314 and all(0<=r['learning']['clip_fraction']<=1 for r in all_updates)
    checks=[];eval_seconds=0.;full_rows=[]
    required=('lh','deny','attack_cycles_completed','first_death_game_seconds','hero_kills','death_causes','first_death_location','spell_completions','raze_hero_hits_by_spell','attack_hero_commands','hero_attacks_released','damage_from_hero','damage_from_creeps','damage_from_tower','event_v2_components','event_v2_reward')
    for opponent,tag in [('v10','v10'),('ZOg-d1','previous')]:
        games=[]
        for seed in (20261103,20261104):
            path=checkpoint.parent/f'eval-000314-{tag}-{seed}/report.json';r=read(path);ecfg=read(path.parent/'config.json')
            assert r['complete'] and r['matches']==64 and r['seed']==seed and len(r['per_match'])==64
            assert not ecfg['argmax'] and not ecfg['zero_previous'] and ecfg['teacher_version']=='v10' and ecfg['seconds']==600
            assert Path(ecfg['weights'])==checkpoint/'weights'
            engine=read(ecfg['engine']);assert engine['uphill_miss'] and abs(engine['pre_game_seconds']-90.03333333333333)<1e-6
            if tag=='previous':
                other=read(path.parent/'opponent.json');assert Path(other['weights'])==WEIGHTS and other['manifest_sha256']==sha(WEIGHTS/'manifest.json')
            assert {g['game'] for g in r['per_match']}==set(range(64))
            assert sum(g['sides'][g['game']%2]['side']=='radiant' for g in r['per_match'])==32
            for g in r['per_match']:
                assert g['result'] in (-1,0,1) and g['outcome_reason'] in ('death','T1','LH600')
                assert g['native_terminal'] and g['simulation_error']==0 and g['lane_creeps_spawned']>=0
                for side,h in enumerate(g['sides']):
                    assert all(k in h for k in required)
                    c=h['event_v2_components'];assert len(c)==6 and c[0]==10*g['result']*(1 if side==0 else -1)
                    assert abs(c[1]-.5*h['lh'])<1e-3 and abs(c[2]-.2*h['deny'])<1e-3
                    assert c[3]<=0 and c[4]>=0 and c[5]<=0 and abs(sum(c)-h['event_v2_reward'])<1e-3
                    assert abs(c[3]+g['sides'][1-side]['event_v2_components'][4])<1e-3
            games+=r['per_match'];eval_seconds+=r['wall_seconds']
            checks.append(dict(opponent=opponent,seed=seed,matches=64,both_heroes_checked=128,radiant=32,dire=32,report_sha256=sha(path),evaluation_minutes=r['wall_seconds']/60))
        full_rows+=summaries('RL-final',dict(per_match=games),opponent,'event-v2')
    assert abs(eval_seconds-result['evaluation_seconds'])<1e-5
    source=read(REPORT/'protected-source-audit.json')
    subprocess.run(['git','diff','--exit-code',source['baseline'],'--']+[r['path'] for r in source['files']],check=True,capture_output=True)
    for p in REPORT.iterdir():
        if p.is_file():assert p.stat().st_size<1_000_000,p
    summary=dict(passed=True,evaluation_matches=256,hero_perspectives=512,evaluation_series=checks,summary=full_rows,shared_hour_final=True,unique_final_checkpoints=1,checkpoint=str(checkpoint),checkpoint_tensor_files_unchanged=59,model_and_optimizer_unchanged_during_evaluation=True,training_seconds_during_all_evaluations=result['training_seconds'],evaluation_minutes=eval_seconds/60,all_ppo_blocks=314,clip_blocks_in_unit_interval=314,gpu_minutes=used(),gpu_limit_minutes=300,gpu_jobs_completed=sum(r.get('gpu_budget',False) for r in jobs),gpu_job_failures=0,training_restarts=0,protected_files_unchanged=source['files_unchanged'],historical_reference_reward='event-v2 not measured for historical P09b d1-v10 and v10-v10 reference series; no approximation substituted')
    save(REPORT/'final-audit.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('summary','evaluation_series')},indent=2))

if __name__=='__main__':main()
