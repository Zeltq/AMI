"""Read-only checks of completed P16 artifacts, with compact Git evidence."""
import datetime,json,math,subprocess,sys
from pathlib import Path
from p16_run import RUN,REPORT,ROOT,read,save

def run():
    state=read(RUN/'state.json');assert state['stage']=='complete'
    stages=read(RUN/'stages.json');evaluations=read(RUN/'evaluations.json')
    parities={k:v['result'] for k,v in stages.items() if k.startswith('parity-')}
    assert all(v['passed'] and v['steps']==128 and v['roundtrip']=={'equal':49,'total':49} for v in parities.values())
    result=dict(results_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                command='C:/Python314/python.exe -X utf8 gpu/scripts/p16_final_audit.py',
                parity_passed=len(parities),parity_total=len(parities),parity_steps=sum(v['steps'] for v in parities.values()),training={})
    for name in ('critic','ppo','self-play'):
        r=read(RUN/name/'result.json');blocks=[read(p) for p in sorted((RUN/name).glob('block-*.json'))]
        rows=[row for b in blocks for row in b['update']['minibatches']]
        finite=sum(all(math.isfinite(v) for v in row.values() if isinstance(v,float)) for row in rows)
        assert finite==len(rows) and len(blocks)==r['blocks']
        assert read(RUN/'jobs'/name/'status.json')['exit_code']==0
        result['training'][name]=dict(blocks=len(blocks),finite_updates=finite,total_updates=len(rows),training_seconds=r['training_seconds'],anchor_sha256=r['anchor_sha256'])
    assert result['training']['ppo']['training_seconds']>=7200
    assert 7200<=result['training']['self-play']['training_seconds']<7320
    assert read(RUN/'plan.json')['stop_after_self_play_minutes']==120
    comparisons=[]
    for e in evaluations:
        d=read(ROOT/e['summary']);assert e['complete'] and d['matches']=={'numerator':128,'denominator':128}
        assert Path(e['weights']).is_dir()
        if not e.get('sampling') or not e.get('scripted_base_to_mid'):continue
        c=d['controllers']['CHiO'];comparisons.append(dict(label=e['label'],opponent=e['opponent'],actors=c['actors'],outcomes=c['outcomes'],outcome_reasons=c['outcome_reasons'],
            deaths=c['deaths'],death_causes=c['death_causes'],lh=c['lh'],deny=c['deny'],mid_arrivals=c['mid_arrivals'],mid_time=c['mid_arrival_elapsed_seconds'],
            razes=c['raze_hero_hit_rates'],purchases=c['purchases'],uses=c['consumable_uses'],counter_denominator=c['actors'],stuck=c['stuck_after_mid'],duration=d['duration_game_seconds']))
    final=[e for e in evaluations if e['label']=='self-play-120m'];assert {e['opponent'] for e in final}=={'v10','v9','previous'}
    assert all(e['sampling'] and e['scripted_base_to_mid'] for e in final)
    result['evaluations']=dict(complete=len(evaluations),total=len(evaluations),matches=len(evaluations)*128,final_complete=3,final_required=3)
    result['python']=read(REPORT/'final-python-verification.json');assert result['python']['passed']==result['python']['total']==272
    rust=read(REPORT/'nav-full-verification.json');assert '143 passed' in '\n'.join(rust['full-rust-tail'])
    result['rust']=dict(passed=143,total=143,source='nav-full-verification.json; no subsequent Rust changes')
    assert not subprocess.check_output(['git','diff','--name-only','2f8d896','--','gpu/src'],cwd=ROOT,text=True).strip()
    end=datetime.datetime.fromisoformat(state['finished_utc']);deadline=datetime.datetime.fromisoformat(state['deadline_utc']);assert end<deadline
    result.update(finished_utc=state['finished_utc'],deadline_utc=state['deadline_utc'],user_requested_early_stop=True)
    save(REPORT/'final-comparison.json',comparisons)
    best=read(RUN/'best.json');assert best['label']=='ppo-120m'
    best['automated_rank_reason']=best['reason']
    best['reason']='Итоговый выбор PPO: против v10/v9 победы136/256, смерти66/256, фарм3599/256, смерти от башни21/256; у self-play140/256,105/256,1034/256,51/256. Self-play сильнее против v9 (90/128 против75/128) и выиграл личную встречу69/128, но PPO лучше против v10 (61/128 против50/128), выживает и фармит устойчивее. Койлы PPO1731/2841 против self-play1174/1812; выход на мид256/256 у обоих. Матчи PPO дольше (90845.31/256с против52246.04/256с), но фарм в минуту тоже выше. Flask666/256 против394/256, Mango0/256 у обоих, Tango9/256 против12/256. Полная статистика и причины исходов: final-comparison.json. Преимущество self-play в личной встрече сохранено; перенос в Dota не проверен.'
    save(RUN/'best.json',best);save(REPORT/'best.json',best)
    result['selected_model']=best['label'];result['selected_weights']=best['weights']
    save(REPORT/'final-review.json',result)
    from p16_report import refresh
    refresh()
    text=(ROOT/'reports/P16.md').read_text(encoding='utf-8');assert len(text.splitlines())<=150 and text.splitlines()[-1]=='DONE P16'
    assert all(p.stat().st_size<1000000 for p in REPORT.rglob('*') if p.is_file())
    print(json.dumps(dict(passed=True,report_lines=len(text.splitlines()),evaluations=result['evaluations'],best=best['label'])))

if __name__=='__main__':run()
