"""Additional user-requested 128-match final-versus-pilot evaluation; no training."""
import json,os,subprocess
from pathlib import Path
import p09c_run as runner
from p09b_results import summaries,csvwrite
from p09c_run import read,save,sha

def main():
    report=runner.ROOT/'reports/P09c/pilot-final';report.mkdir(parents=True,exist_ok=True)
    runner.REPORT=report
    lock=runner.exclusive();os.environ['SF_DUEL_DATA_ROOT']='C:/AMI'
    assert sha(runner.EXE)==read(runner.ROOT/'reports/P09c/verified.json')['exe_sha256']
    final=runner.RUN/'rl-main-001/checkpoint-000314/weights';pilot=runner.RUN/'rl-pilot1-001/checkpoint-000126/weights'
    original_minutes=read(runner.RUN/'completion.json')['gpu_minutes'];games=[];checks=[]
    for seed in (20261103,20261104):
        out=runner.RUN/'pilot-final'/f'seed-{seed}'
        command=[runner.EXE,'evaluate-ggd',runner.ENGINE,final,out,'--mode','teacher','--matches',64,'--seconds',600,'--seed',seed,'--sampling','--teacher-version','v10','--opponent-weights',pilot]
        runner.run(f'pilot-final-{seed}','evaluation',command,out/'report.json',True)
        r=read(out/'report.json');cfg=read(out/'config.json');other=read(out/'opponent.json')
        assert r['complete'] and r['matches']==64 and r['seed']==seed and not cfg['argmax']
        assert Path(cfg['weights'])==final and Path(other['weights'])==pilot
        assert other['manifest_sha256']==sha(pilot/'manifest.json')
        assert sum(g['sides'][g['game']%2]['side']=='radiant' for g in r['per_match'])==32
        for g in r['per_match']:
            for side,h in enumerate(g['sides']):
                c=h['event_v2_components'];assert c[0]==10*g['result']*(1 if side==0 else -1)
                assert abs(c[1]-.5*h['lh'])<1e-3 and abs(c[2]-.2*h['deny'])<1e-3
                assert abs(sum(c)-h['event_v2_reward'])<1e-3
        games+=r['per_match']
        compact=[{**g,'sides':[{k:v for k,v in h.items() if k!='minute_snapshots'} for h in g['sides']]} for g in r['per_match']]
        evidence=dict(seed=seed,config=cfg,opponent=other,final_manifest_sha256=sha(final/'manifest.json'),evaluation_minutes=r['wall_seconds']/60,source_report_sha256=sha(out/'report.json'),per_match=compact)
        (report/f'matches-{seed}.json').write_text(json.dumps(evidence,ensure_ascii=False,separators=(',',':')),encoding='utf-8')
        checks.append(dict(seed=seed,matches=64,radiant=32,dire=32,hero_perspectives=128,evaluation_minutes=r['wall_seconds']/60))
    rows=summaries('RL-final',dict(per_match=games),'pilot1','event-v2');save(report/'metrics.json',rows);csvwrite(report/'metrics.csv',rows)
    totals=[r for r in rows if r['side']=='total'];assert len(totals)==2 and all(r['matches']==128 for r in totals)
    commands=[r for r in runner.history() if r['stage'].startswith('pilot-final-')]
    save(report/'commands.json',[{k:r[k] for k in ('stage','command_literal','exit_code','minutes')} for r in commands])
    save(report/'budget.json',dict(original_gpu_minutes=original_minutes,additional_gpu_minutes=sum(r['minutes'] for r in commands),cumulative_gpu_minutes=runner.used(),limit_gpu_minutes=300))
    save(report/'audit.json',dict(passed=True,matches=128,hero_perspectives=256,checks=checks,final=str(final),pilot=str(pilot),sampling=True,environment='005',trained_during_evaluation=False))
    text=['# P09c: финал против пилота, дополнительная оценка','', 'По запросу пользователя после завершения основного P09c:128 матчей, seed20261103/20261104 по64, sampling, окружение005, по64 за каждую сторону. Обучение не запускалось.','', '| Модель | Победы / ничьи / поражения | LH / матчи | Deny / матчи | Добиваний в среднем | Смерть / матчи | До120с / матчи |','|---|---|---|---|---|---|---|']
    for r in totals:text.append(f'| {r["model"]} | {r["wins"]}/128; {r["draws"]}/128; {r["losses"]}/128 | {r["lh"]:.0f}/128 | {r["deny"]:.0f}/128 | {(r["lh"]+r["deny"])/128:.2f} | {r["matches_with_death"]}/128 | {r["early_deaths"]}/128 |')
    text+=['','Полная статистика обеих моделей по сторонам: [metrics.csv](metrics.csv). Покомпонентная reward, причины/места смертей, койлы, атаки и урон сохранены в CSV и JSON матчей.','',f'Оценка: {sum(c["evaluation_minutes"] for c in checks):.3f} минуты; точные команды: [commands.json](commands.json). Итоговый GPU-учёт с исходным P09c: {runner.used():.3f}/300 минут.','',f'Финал: `{final}`; пилот: `{pilot}`.','', 'Ограничение: одна серия на двух фиксированных seed; результат не доказывает обобщение на все начальные состояния и соперников.','', 'DONE P09c pilot-final']
    (report/'report.md').write_text('\n'.join(text)+'\n',encoding='utf-8',newline='\n')
    for p in report.iterdir():assert p.stat().st_size<1_000_000,p
    subprocess.run(['git','add','reports/P09c/pilot-final'],cwd=runner.ROOT,check=True)
    subprocess.run(['git','diff','--cached','--stat'],cwd=runner.ROOT,check=True)
    if subprocess.check_output(['git','diff','--cached','--name-only'],cwd=runner.ROOT,text=True).strip():subprocess.run(['git','commit','-m','[P09c] publish direct final-versus-pilot evaluation'],cwd=runner.ROOT,check=True)
    subprocess.run(['git','-c','core.sshCommand=C:/Windows/System32/OpenSSH/ssh.exe -o BatchMode=yes -o ConnectTimeout=15','push','-u','origin','agent/P09c-zog-rl2'],cwd=runner.ROOT,check=True,timeout=120)
    runner.status(dict(stage='pilot-final complete; result audit and chat summary pending',pid=None,background_work_remaining=True,report=str(report/'report.md'),gpu_minutes=runner.used()))

if __name__=='__main__':main()
