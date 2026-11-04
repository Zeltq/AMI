"""Publish small P09 metrics and the review report from completed artifacts."""
import csv,hashlib,json
from pathlib import Path
from p09_run import RUN,REPORT,ROOT,save

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def ratio(n,d):return f'{n:g}/{d:g}'
def table(lines,columns,rows):
    lines+=['| '+' | '.join(columns)+' |','|'+'|'.join(['---']*len(columns))+'|']
    lines+=['| '+' | '.join(str(v) for v in row)+' |' for row in rows];lines+=['']
def csvwrite(name,rows):
    with (REPORT/name).open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def summarize(name,matches,history):
    rows=[]
    for side in ('radiant','dire','total'):
        selected=[]
        for seed,g in matches:
            s=g['game']%2;hero=g['sides'][s]
            assert name=='v10' or hero['controller']=='GGD'
            if side=='total' or hero['side']==side:selected.append((g,hero,s))
        n=len(selected);r=dict(model=name,side=side,matches=n)
        for label,value in [('wins',1),('draws',0),('losses',-1)]:
            r[label]=sum(int(g['result'])*(1 if s==0 else -1)==value for g,h,s in selected)
            for reason in ('death','T1','LH600'):
                r[label+'_'+reason]=sum(int(g['result'])*(1 if s==0 else -1)==value and g['outcome_reason']==reason for g,h,s in selected)
        for k in ('lh','deny','attack_cycles_completed','hero_kills','hero_attacks_released','attack_hero_commands','damage_from_hero','damage_from_creeps','damage_from_tower','deaths'):
            r[k]=sum(h[k] for g,h,s in selected)
        r['lane_creeps_spawned']=sum(g['lane_creeps_spawned'] for g,h,s in selected)
        times=[h['first_death_game_seconds'] for g,h,s in selected if h['first_death_game_seconds'] is not None]
        r.update(matches_with_death=len(times),first_death_time_sum=sum(times),early_deaths=sum(t<120 for t in times))
        for i,spell in enumerate(('near','mid','far')):
            r[spell+'_casts']=sum(h['spell_completions'][i] for g,h,s in selected);r[spell+'_hero_hits']=sum(h['raze_hero_hits_by_spell'][i] for g,h,s in selected)
            r['opponent_'+spell+'_casts']=sum(g['sides'][1-s]['spell_completions'][i] for g,h,s in selected)
        for i,cause in enumerate(('hero_attack','raze','creeps','tower','requiem')):
            for k in ('count','own_t1_distance_sum','enemy_t1_distance_sum'):
                r[cause+'_'+k]=sum(h['death_causes'][i][k] for g,h,s in selected)
        assert r['deaths']==sum(r[c+'_count'] for c in ('hero_attack','raze','creeps','tower','requiem'))
        if history:
            v=history['validation'];v=v if side=='total' else v['attack_by_side'][side]
            r.update(validation_frames=v['frames'],action_correct=v['action_correct'],joint_action_correct=v['joint_action_correct'],nll_sum=v.get('nll_sum',v.get('nll',0)*v['frames']))
        else:r.update(validation_frames=0,action_correct=0,joint_action_correct=0,nll_sum=0)
        rows.append(r)
    return rows

def main():
    result=read(RUN/'ZOg/result.json');audit=read(REPORT/'dataset-audit.json');smoke=read(REPORT/'smoke-audit.json');rows=[];eval_commands=[]
    assert result['complete'] and result['epochs']==3
    for epoch in range(1,4):
        matches=[]
        for seed in (20261103,20261104):
            target=RUN/f'ZOg/epoch-{epoch:03d}/vs-teacher-{seed}';report=read(target/'report.json');assert report['complete'] and report['matches']==64
            games=report['per_match'];matches.extend((seed,g) for g in games)
            for g in games:
                for s in g['sides']:s.pop('minute_snapshots',None)
            save(REPORT/f'ZOg-e{epoch}-{seed}.json',dict(seed=seed,per_match=games))
            gameplay=result['history'][epoch-1]['gameplay'][f'teacher-{seed}']
            eval_commands.append(dict(stage=f'ZOg-e{epoch}-{seed}',command=gameplay['command'],command_literal=gameplay['command_literal'],minutes=report['wall_seconds']/60,included_in_train=True))
        rows+=summarize(f'ZOg-e{epoch}',matches,result['history'][epoch-1])
    matches=[]
    for seed in (20261103,20261104):
        report=read(RUN/f'teacher-{seed}/report.json');assert report['complete']
        matches.extend((seed,g) for g in report['per_match'])
        for g in report['per_match']:
            for s in g['sides']:s.pop('minute_snapshots',None)
        save(REPORT/f'v10-{seed}.json',dict(seed=seed,per_match=report['per_match']))
    rows+=summarize('v10',matches,None);csvwrite('metrics.csv',rows);save(REPORT/'metrics.json',rows)
    save(REPORT/'training.json',result)
    save(REPORT/'training-config.json',read(RUN/'ZOg/config.json'))
    assert all(r['matches']==(128 if r['side']=='total' else 64) for r in rows)
    provenance={'baseline_git':'9475365','observation_profile':'ggd-v4','teacher_sha256':result['teacher_sha256'],'dataset_sha256':result['dataset_sha256'],'files':{}}
    paths=[RUN/'sf-duel-gpu.exe',Path('C:/AMI/runs/p07/dota-calibration-005/engine.json')]
    for epoch in range(1,4):
        checkpoint=RUN/f'ZOg/epoch-{epoch:03d}.pt';manifest=RUN/f'ZOg/epoch-{epoch:03d}/weights/manifest.json'
        exported=read(manifest);assert exported['source_sha256']==hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        assert exported['model_name']==f'ZOg-e{epoch}' and exported['observation_profile']=='ggd-v4' and not exported['timing_oracle']
        paths+=[checkpoint,manifest]+[manifest.parent/t['file'] for t in exported['tensors'].values()]
        for t in exported['tensors'].values():assert hashlib.sha256((manifest.parent/t['file']).read_bytes()).hexdigest()==t['sha256']
    for path in paths:provenance['files'][str(path)]={'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    save(REPORT/'provenance.json',provenance)
    commands=[r for r in read(RUN/'commands.json') if 'exit_code' in r];public=[]
    for c in commands:
        r={k:c[k] for k in ('stage','kind','command_literal','exit_code','environment') if k in c}
        if c['kind'] in ('collection','training','evaluation'):r['minutes']=c['minutes']
        public.append(r)
    save(REPORT/'commands.json',public+eval_commands)
    budget=sum(c['minutes'] for c in commands if c['gpu_budget']);save(REPORT/'budget.json',dict(used_minutes=budget,limit_minutes=240,method='sum parent GPU job wall times; embedded epoch evaluations counted once; includes collection, verification and failed jobs'))
    lines=['# P09a — ZOg: imitation teacher-v10','', '## Результаты','',f"Сбор: {audit['games']}/1024 матчей, {audit['valid_frames']} решений обоих героев; split по матчам 768/1024 train, 128/1024 validation, 128/1024 test. Oracle выключен, ggd-v4/005: pregame 90.033333 с, промах дальних атак снизу вверх 25%.",
           'Оценка каждой эпохи: sampling, seed 20261103 и 20261104, по 64 матча; выбранный герой чередуется, 64 Radiant и 64 Dire. В ориентире v10–v10 выбраны те же стороны по game%2, всего 128 героев в 128 матчах. Данные обоих героев сохранены в JSON.',
           'R/D/Σ = Radiant/Dire/сумма. Время — игровые секунды, включая отрицательный pregame. NLL невзвешенная; совпадение типа команды и всей команды с активными аргументами указаны отдельно.','']
    ident=lambda r:[r['model'],{'radiant':'R','dire':'D','total':'Σ'}[r['side']]]
    table(lines,['Модель','Сторона','W / N','D / N','L / N','Конец: смерть / T1 / LH600'],[ident(r)+[ratio(r[k],r['matches']) for k in ('wins','draws','losses')]+['; '.join(ratio(sum(r[k+'_'+reason] for k in ('wins','draws','losses')),r['matches']) for reason in ('death','T1','LH600'))] for r in rows])
    table(lines,['Модель','Сторона','LH / N','deny / N','farm / spawn','farm / циклы','Смерть / N','<120с / N','Σ первой смерти, с / n','Убийства / N'],[ident(r)+[ratio(r[k],r['matches']) for k in ('lh','deny')]+[ratio(r['lh']+r['deny'],r[k]) for k in ('lane_creeps_spawned','attack_cycles_completed')]+[ratio(r[k],r['matches']) for k in ('matches_with_death','early_deaths')]+[ratio(r['first_death_time_sum'],r['matches_with_death']),ratio(r['hero_kills'],r['matches'])] for r in rows])
    table(lines,['Модель','Сторона','Касты near/mid/far / N','v10 в тех же матчах: near/mid/far / N','Попадания near/mid/far / касты','Атаки героя / N','Урон от героя/крипов/башни / N'],[ident(r)+['; '.join(ratio(r[s+'_casts'],r['matches']) for s in ('near','mid','far')),'; '.join(ratio(r['opponent_'+s+'_casts'],r['matches']) for s in ('near','mid','far')),'; '.join(ratio(r[s+'_hero_hits'],r[s+'_casts']) for s in ('near','mid','far')),ratio(r['hero_attacks_released'],r['matches']),'; '.join(ratio(r['damage_from_'+s],r['matches']) for s in ('hero','creeps','tower'))] for r in rows])
    lines+=['В соседнем столбце v10 — соперник выбранного героя именно в этих матчах; для строки R он играет Dire, для D — Radiant. Строки v10 отдельно по-прежнему относятся к независимому ориентиру teacher-v10 против teacher-v10.','']
    table(lines,['Модель','Сторона','Смерти: атака героя / койл / крипы / башня / ульт','Тип команды / validation','Вся команда / validation','NLL'],[ident(r)+['; '.join(ratio(r[c+'_count'],r['deaths']) for c in ('hero_attack','raze','creeps','tower','requiem'))]+([ratio(r[k],r['validation_frames']) for k in ('action_correct','joint_action_correct')]+[f"{r['nll_sum']/r['validation_frames']:.6f}"] if r['validation_frames'] else ['—','—','—']) for r in rows])
    lines+=['Расстояния смерти до своей/вражеской T1: суммы и числа событий по каждой причине, исходы по каждой причине и W/D/L, приказы атаки героя — [metrics.csv](P09a/metrics.csv). Индивидуальная первая смерть и её расстояния — в JSON матчей. 0/0 означает отсутствие событий. Spawn учитывает реально появившихся крипов обеих сторон до конца матча; циклы — изменения ATTACK_READY. Атаки героя считаются по выпуску, койлы — по завершению каста. Урон — полученный героем, после защиты и с ограничением остатком HP.','', '## Наблюдения','',
            f"- **Подтверждено:** train содержит near {audit['train_action_counts'][6]}/{audit['train_decisions']}, mid {audit['train_action_counts'][7]}/{audit['train_decisions']}, far {audit['train_action_counts'][8]}/{audit['train_decisions']}, ATTACK героя {audit['train_hero_attack_decisions']}/{audit['train_decisions']}.",
            f"- **Подтверждено:** по поправке пользователя веса койлов {audit['action_weights']} ограничены сверху 4; все ATTACK, включая героя, имеют вес 4. Формула койла min(4, max(1, 0.01×N/n)), округление до 0.001; веса вычислены по train. Это частотное усиление с заданным ограничением, оно не задаёт частоту кастов при sampling. Вес умножает весь NLL решения; нормировка — сумма весов валидных кадров.",
            '- **Подтверждено:** модель и AdamW инициализированы с нуля; M2/P08 не загружались. Test не использован для обучения, веса loss выбраны по train.',
            '- **Гипотеза:** различия сторон могут зависеть от геометрии и распределения траекторий; отдельные абляции причин асимметрии не проводились.','', '## Что сделано','',
            '- `demo_telemetry.rs`, `imitation.rs`, `teacher_combat.rs`, `physics.rs`, `teacher_eval.rs`, `main.rs`: отдельные reward-снимки до/после решения и счётчики оценки. Добавлены только записи телеметрии; правила урона, решений учителя и профиль наблюдений сохранены.',
            '- `p09_data.py`: `.reward.npz` на каждый блок/шард; FP32 состояния, valid, actor, финальный outcome/reason на каждом решении; схема и SHA в `reward-metadata.json`. XP хранится внутри уровня и суммарно. Эти поля не подаются политике.',
            '- `train_ggd.py`: ggd-v4, веса редких действий, раздельная validation по сторонам, две оценки после каждой эпохи, checkpoint и Burn-export. AdamW 1e-4, wd 1e-5, 8×128, clip 1, seed 20261102; три эпохи. Автоматического выбора эпохи ZOg нет.',
            '- `p09_run.py`, `p09_results.py`, `tests/test_p09.py`, `gpu_tests.rs`: последовательный запуск, блокировка дубликатов, общий GPU-бюджет, метрики и проверки.',
            f"- Короткий сбор: {smoke['games']}/4 матчей, {smoke['checked_values']}/{smoke['checked_values']} значений, финальная телеметрия {smoke['final_actors']}/8 героев. [Проверка](P09a/smoke-audit.json), [основной датасет](P09a/dataset-audit.json), [тесты](P09a/verification.json).",'', '## Команды, которыми получены результаты','',
            'Рабочая директория `C:/AMI-wt/P09a`. Дословные команды сбора, обучения, всех оценок и проверок: [commands.json](P09a/commands.json); конфигурации — рядом. Длительности сбора/обучения/оценки указаны там в минутах. Время дочерних оценок входит в время train.',
            f'Общий консервативный расход GPU: {budget:.3f}/240 мин, включая короткий сбор и GPU-проверки; дочерние оценки не складываются повторно.','', '## Что не получилось или не проверено','',
            '- Исправлен устаревающий source_sha256 Burn после дополнения checkpoint результатами оценки; отдельный fix 74773a9, целевой тест 1/1. Первый короткий сбор выявил чтение исхода после reset арены; итог сохранён событием, сбор повторён.',
            '- Подтверждено: ggd-v4 supplement сохраняет последний raw stack count после истечения дебаффа; профиль по условию не изменён. Reward metadata содержит активные стаки (0 после STACK_END), как в формуле урона и базовых наблюдениях. Это различие выявлено сверкой короткого сбора.',
            '- Перенос в настоящую Dota, новые seed обучения и дополнительные seed оценки не проверялись. NLL/validation для детерминированного teacher-ориентира не измеряются.',
            '- У героя, умершего в pregame, может не быть валидного решения в момент конца на 0:00; итоговый исход/причина всё равно записаны на всех его решениях. Терминальность transition хранится отдельно. При одновременной смерти и T1 причина помечается смертью; текущая выборка проверяется по сырой телеметрии.',
            '', '## Пути к локальным артефактам','',
            '- `C:/AMI/runs/p09/dataset/` — демонстрации и reward metadata; `smoke4/` — четыре проверочных матча; `sf-duel-gpu.exe` — бинарник.',
            '- ZOg-e1/e2/e3: `C:/AMI/runs/p09/ZOg/epoch-001.pt`, `epoch-002.pt`, `epoch-003.pt`; Burn: `C:/AMI/runs/p09/ZOg/epoch-001/weights`, `epoch-002/weights`, `epoch-003/weights`.',
            '- Оценки: `C:/AMI/runs/p09/ZOg/epoch-00N/vs-teacher-20261103` и `vs-teacher-20261104`; ориентиры `C:/AMI/runs/p09/teacher-20261103`, `teacher-20261104`. Состояние запуска: `C:/AMI/runs/p09/state.json`, namespace P09a в `C:/AMI/runs/pilot-status.json`.','', 'Ожидается выбор пользователя для DAgger.']
    assert len(lines)<=150,len(lines)
    (ROOT/'reports/P09a.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
    for path in REPORT.iterdir():assert path.stat().st_size<1_000_000,(path,path.stat().st_size)

if __name__=='__main__':main()
