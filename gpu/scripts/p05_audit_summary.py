"""Publish compact P05 evidence and explicit categories, without changing inputs."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[2]
STATS=['health_max','mana_max','armor','attack_damage','attack_interval','attack_point',
       'health_regen','mana_regen','move_speed','attack_range']


def location(path,needle):
    lines=(ROOT/path).read_text(encoding='utf-8').splitlines()
    return f'{path}:{next(i+1 for i,line in enumerate(lines) if needle in line)}'


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--artifacts',type=Path,default=Path('C:/AMI/runs/p05'))
    p.add_argument('--output',type=Path,default=ROOT/'reports/P05');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    phase=json.loads((a.artifacts/'phase-audit.json').read_text());rows=[]
    refs=dict(unit=location('bots/agent.lua','local methods='),items=location('bots/agent.lua','local row={inventory='),
              spawn=location('gpu/src/physics.rs','state[p+UNIT_X]=spawns[spawn]'),
              height=location('gpu/src/physics.rs','state[p+UNIT_Z]=ground_z(geometry[cell(nx,ny)])'),
              init=location('gpu/src/world.rs','state[p+UNIT_HEALTH_MAX]+=44.'),
              observer=location('gpu/src/observe.rs','let mut value=state[p+f]'),
              facing=location('gpu/src/physics.rs','let desired=dy.atan2(dx)'),
              clock=location('gpu/src/observe.rs','let row=actor*WIDTH;let game_time='),
              towers=location('gpu/src/observe.rs','let memory=base+(FOG_START+side*4+tower)'),
              native=location('src/sf_duel/native_data.py','for target,source in UNIT_MAP.items()'),
              native_towers=location('src/sf_duel/native_data.py',"u=next((u for u in state.units if u.name==name)"))
    for field in phase['all_value_fields']:
        group,name=field.split('.')
        if name=='attack_cycle':
            category='b';certainty='подтверждено';reason='GPU копирует неизменяемый нулевой prototype anim_cycle; Dota передаёт фазу текущей анимации, включая idle/walk.'
            code=[refs['observer'],refs['unit'],refs['native']]
            proposed='После P02 определить семантику attack_cycle и новый контракт: моделировать фактическую фазу либо отмечать неизвестность; синхронно переобучить/версионировать. Не обнулять достоверное наблюдение Dota ради старой модели.'
        elif group=='heroes' and name.startswith('facing'):
            category='d';certainty='подтверждено';reason='Разные положения и направления героя на двух траекториях; обе стороны преобразуют угол в sin/cos.';code=[refs['unit'],refs['facing']];proposed=None
        elif group=='heroes' or group=='items':
            category='d';certainty='подтверждено';reason='Первый Lua кадр попал в перенос Wraith Band из stash и применение ауры; GPU уже инициализирован с предметом. К 0:00 запрошенные собственные статы равны. TP cooldown отличается на длительность pregame, у обоих готовность около +10 с.';code=[refs['unit'],refs['items'],refs['init']];proposed=None
        elif group=='creeps':
            category='c' if name not in ('dx','dy') else 'd'
            certainty='подтверждено' if name in ('dx','dy') else 'гипотеза'
            reason='Dota spawn event предшествует расстановке и движению; GPU стартует с калиброванной расстановки. После спавна расходятся путь/поворот/высота. dx/dy также зависят от положения наблюдателя; точная доля каждого эффекта не изолирована.'
            code=[refs['spawn'],refs['height'],refs['facing']];proposed=None
        else:
            category='c';certainty='подтверждено';reason='Pregame Dota около 90 с, конфиг GPU 83 с. Оба кодируют именно DotaTime/600 и одинаковый 0:00; первый кадр относится к разным временам.';code=[refs['clock'],'src/sf_duel/observations.py'];proposed=None
        rows.append(dict(field=field,category=category,certainty=certainty,reason=reason,code=code,proposed=proposed))
    masks=[]
    for field in phase['all_known_fields']:
        group=field.split('.')[0]
        reason={'heroes':'Enemy отсутствует в Lua в первом кадре волны, но уже виден GPU; со следующего кадра есть в обоих.',
                'hero_combat':'Наследует наличие enemy и доступность его известных статов.',
                'enemy_items':'Наследует видимость enemy.',
                'towers':'P03 использует Lua snapshot, скрывающий enemy towers; GPU хранит публичное/последнее состояние как native protobuf. Это разные пути наблюдения, не доказанная ошибка адаптера.',
                'trees':'Разные позиции героя, радиус/видимость и сопоставление деревьев (4 единицы). Вне пересечения сравнивать значения нельзя.',
                'items':'Wraith Band ещё в stash: cooldown/mana_cost неизвестны Lua до инвентаря.',
                'creeps':'Lua возвращает raze_remaining=0 у своего крипа, NativeAdapter не добавляет это поле. GPU соответствует native-пути.'}[group]
        masks.append(dict(field=field,category='d',certainty='подтверждено' if group!='trees' else 'гипотеза',reason=reason))
    dense={2:[],3:[]};spawns=[]
    for line in (a.run/'dota.log').read_text(errors='replace').splitlines():
        if line.startswith('SF_P05_PHASE|'):
            s=json.loads(line.split('|',1)[1])['snapshot'];dense[s['team']].append(s)
        elif line.startswith('SF_P05_SPAWN|'):spawns.append(json.loads(line.split('|',1)[1]))
    transitions={}
    for team,samples in dense.items():
        if not samples:raise ValueError('No dense capture')
        transitions[str(team)]=[dict(time=s['time'],stats={k:s['self'].get(k) for k in STATS},
                                       health=s['self'].get('health'),mana=s['self'].get('mana'),
                                       wraith_band=s['items'][0],tp=s['items'][3]) for s in samples]
    initial=[];zero=[]
    for pair in phase['pairs']:
        if pair['index'] not in (0,1):continue
        target=initial if pair['index']==0 else zero
        for name in STATS:
            n=pair['raw']['heroes']['dota'][0][name];g=pair['raw']['heroes']['gpu'][0][name]
            target.append(dict(team=pair['team'],field=name,dota=n,gpu=g,equal_fp16=n==g))
    engine_path=Path('C:/AMI/runs/gpu/ggd0-il-20261102-202637/dataset/environment/engine.json')
    engine=json.loads(engine_path.read_text())
    result=dict(value_fields=rows,known_fields=masks,category_counts=dict(Counter(r['category'] for r in rows)),
                value_fields_total=220,value_fields_different=len(rows),known_fields_different=len(masks),
                initial_stats=initial,zero_stats=zero,transitions=transitions,
                spawn_events=spawns,simulator_spawns=engine['spawns'],
                engine_sha256=hashlib.sha256(engine_path.read_bytes()).hexdigest(),references=refs,
                capture_status=json.loads((a.run/'status.json').read_text()),
                adapter_changes=0,note='Категории описывают причину превышения порога P03, а не объявляют все отличающиеся состояния ошибками. Ошибка (a) по этим данным не подтверждена; адаптеры не подгонялись под GPU.')
    (a.output/'field-categories.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    for name in ('phase-audit.json','native-phase-audit.json','lua-protobuf.json'):
        shutil.copy2(a.artifacts/name,a.output/name)
    print(json.dumps(dict(categories=result['category_counts'],initial_stats_equal=sum(x['equal_fp16'] for x in initial),zero_stats_equal=sum(x['equal_fp16'] for x in zero),stats_denominator=len(zero),spawn_events=len(spawns))))


if __name__=='__main__':main()
