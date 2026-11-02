"""Generated stock-Dota/GPU lane differential scenarios. Diagnostic data only.

generate creates a frozen manifest; capture records one bounded native batch;
compare never recollects native data. Invalid captures cannot count as passes.
"""
import argparse
import copy
from collections import Counter
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
import subprocess
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
VERSION = 1
TOLERANCES = {'position': 16., 'facing': 15., 'hp': 1.2, 'mana': 1.2,
              'event_ticks': 1, 'target_transition_ticks': 2}
FAMILIES = ('attack', 'cancel_before', 'cancel_after', 'reissue', 'retarget',
            'last_hit', 'deny', 'contest', 'raze', 'turn_raze', 'aggro', 'priority', 'move')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def atomic(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=True, allow_nan=False) + '\n', encoding='utf-8')
    # Windows readers can briefly hold the destination without delete sharing.
    # Retry the same complete snapshot; never remove the previous good file.
    for attempt in range(5):
        try:
            temp.replace(path)
            break
        except PermissionError:
            if attempt == 4: raise
            time.sleep(.02 * (attempt + 1))


def generate(count, seed, damage_mode='stock'):
    if not 2 <= count <= 10000:
        raise ValueError('Scenario count must be2..10000')
    if damage_mode not in ('stock','fixed_base_roll'):raise ValueError('Unsupported damage mode')
    cases = []
    for i in range(count):
        team = 2 + i % 2; sign = 1 if team == 2 else -1
        pair = i // 2; family = FAMILIES[pair % len(FAMILIES)]; variant = pair // len(FAMILIES)
        distance = (64, 128, 224, 256, 384, 448, 500, 550, 575, 600, 700, 800)[variant % 12]
        angle = (0, 45, 90, 135, 180, -90, -45, -135)[variant // 12 % 8]
        timing = (1, 5, 9, 13, 20)[variant // 96 % 5]
        actor = 'radiant' if team == 2 else 'dire'
        enemy = 'dire' if team == 2 else 'radiant'
        own = {'id': actor, 'kind': 'hero', 'team': team, 'x': 80. if team==2 else 80.+distance, 'y': -816.,
               'facing': angle + (180 if team == 3 else 0), 'level':(1,3,6)[variant//32%3]}
        other = {'id': enemy, 'kind': 'hero', 'team': 5-team, 'x': 80., 'y': -1516., 'facing': 90.,'level':own['level']}
        victim = {'id': 'victim', 'kind': ('melee', 'ranged', 'siege')[variant // 8 % 3],
                  'team': team if family == 'deny' else 5-team,
                  'x': own['x'] + sign*distance, 'y': -816.+variant//96*32, 'facing': 180 if team == 2 else 0}
        units = [own, other, victim]
        orders = [{'tick': 12, 'actor': actor, 'action': 3, 'target': 'victim'}]
        if family in ('last_hit', 'deny', 'contest'):
            victim['health'] = (15, 30, 45, 55)[variant // 12 % 4]
        if family in ('retarget', 'contest', 'priority'):
            units.append({'id': 'bait', 'kind': 'melee', 'team': team,
                          'x': victim['x'] + sign*(80 + variant % 4*32), 'y': -816., 'facing': 180})
        if family == 'retarget':
            units[-1]['team'] = 5-team
            orders.append({'tick': 12 + timing, 'actor': actor, 'action': 3, 'target': 'bait'})
        if family.startswith('cancel_'):
            orders.append({'tick': 12 + (5 if family == 'cancel_before' else 20) + timing-1,
                           'actor': actor, 'action': 4})
        if family == 'reissue':
            orders.append(dict(orders[0], tick=12 + timing))
        if family in ('raze', 'turn_raze'):
            orders = [{'tick': 12, 'actor': actor, 'action': 6 + variant % 3}]
            if family == 'turn_raze':
                orders.insert(0, {'tick': 12-timing if timing < 12 else 1, 'actor': actor,
                                  'action': 1, 'x': 80., 'y': -560.})
        if family == 'aggro':
            if team==3:
                own['x']=80.+max(distance,384)
                victim['x']=own['x']-distance
            other.update(x=own['x']+sign*384, y=-816.)
            orders = [{'tick': 12, 'actor': actor, 'action': 3, 'target': enemy},
                      {'tick': 13, 'actor': actor, 'action': 4},
                      {'tick': 12+max(30,timing*8), 'actor': actor, 'action': 3, 'target': enemy}]
        if family == 'priority':
            orders = []
        if family == 'move':
            orders = [{'tick': 12, 'actor': actor, 'action': 1,
                       'x': own['x']+sign*256, 'y': own['y']+((variant % 3)-1)*256}]
        case = {'family': family, 'mode': 'controlled', 'actor_team': team,
                'duration_ticks': 180 if family == 'aggro' else 100,
                'units': units, 'orders': sorted(orders, key=lambda o: o['tick']),
                'damage_mode': damage_mode, 'seed': seed,
                'comparison': dict(TOLERANCES), 'split': 'validation' if variant % 5 == 4 else 'calibration'}
        # Eight genuine first-two-wave cases, each requires a fresh native map.
        if i < 8:
            positions = [(-6700., -6700.00390625), (6900., 6649.99609375)] if i < 4 else [(80., -816.), (464., -816.)]
            case.update(family='natural_wave', mode='natural', duration_ticks=1800, orders=[],
                        units=[dict(id=name, kind='hero', team=j+2, x=xy[0], y=xy[1],
                                    facing=45. if j == 0 else 225.)
                               for j, (name, xy) in enumerate(zip(('radiant', 'dire'), positions))])
            case['replicate'] = i % 4
            case['damage_mode']='stock'
        case['id'] = f'{case["family"]}-{team}-' + digest(case)[:16]
        cases.append(case)
    if len({c['id'] for c in cases}) != len(cases):
        raise ValueError('Generated duplicate scenarios; reduce count or extend explicit coverage')
    return {'schema': VERSION, 'seed': seed, 'training_eligible': False,
            'question': 'Do identical admitted commands produce matching lane transitions in stockDota and theGPU simulator?',
            'minimum': 'First qualify one small controlled batch and one natural first-two-wave case; then the frozen matrix.',
            'stop': 'Each declared case has an admitted native reference and GPU comparison; invalid captures stop the batch and are never passes.',
            'tolerances': dict(TOLERANCES), 'cases': cases,
            'manifest_sha256': digest(cases),
            'dod': ['Every declared case executed; no silent skip/invalid case counted as pass',
                    'Stable entity mapping, identical commands and admitted initial states',
                    'Exact target identities, hit/miss, LH/deny and death outcomes',
                    'Position/facing/timing/HP residuals reported with frozen tolerances and exact-equality counters',
                    'Stock RNG differences cannot certify exactHP: deterministic causality or repeated stochastic evidence required',
                    'Every repaired failing case and earlier passing cases rerun against immutable native references',
                    'Fresh validation seeds/heldoutvariants and frozenMIA2 matched-mode control before fullPPO']}


def validate_manifest(manifest):
    if manifest['schema'] != VERSION or digest(manifest['cases']) != manifest['manifest_sha256']:
        raise ValueError('Manifest schema/checksum mismatch')
    if len({c['id'] for c in manifest['cases']}) != len(manifest['cases']):
        raise ValueError('Duplicate case IDs')


def representative_profile(source):
    """Cover all lane families/sides/types/levels before the full matrix."""
    validate_manifest(source)
    if len(source['cases'])!=3000:raise ValueError('Profile requires frozen3000-case matrix')
    cases=[copy.deepcopy(source['cases'][i])for i in (0,1,4,5)]
    for variant in (0,8,16,32,48,65,80,96,112):
        for family_index,family in enumerate(FAMILIES):
            v=1 if variant==0 and family_index<4 else variant
            for side in (0,1):
                case=copy.deepcopy(source['cases'][2*(v*len(FAMILIES)+family_index)+side])
                if case['family']!=family:raise ValueError('Source matrix ordering changed')
                if case['split']!='calibration':raise ValueError('Discovery profile must not consume heldout matrix cases')
                cases.append(case)
    for team in (2,3):
        base=next(c for c in source['cases']if c['family']=='deny'and c['actor_team']==team)
        for kind in ('melee','ranged','siege'):
            for fraction in (.49,.50,.51):
                c=copy.deepcopy(base);c.pop('id');victim=next(u for u in c['units']if u['id']=='victim')
                victim.pop('health',None);victim.update(kind=kind,health_fraction=fraction)
                c['id']='deny-boundary-'+str(team)+'-'+digest(c)[:16];cases.append(c)
    result=dict(source,cases=cases,parent_manifest_sha256=source['manifest_sha256'],manifest_sha256=digest(cases))
    result['question']='Locate common lane transition defects across every family/side, all three creep types, levels1/3/6 and deny49/50/51% boundaries.'
    result['minimum']='256 declared cases; a representative diagnostic before full3000 regression.'
    result['stop']='All256 processed with pass/mismatch/invalid retained, or fatal/incomplete capture. Then fix common causes before larger regression.'
    validate_manifest(result);return result


def lua(value):
    if value is None: return 'nil'
    if isinstance(value, bool): return str(value).lower()
    if isinstance(value, str): return json.dumps(value, ensure_ascii=True)
    if isinstance(value, (int,float)):
        if not math.isfinite(value): raise ValueError('NonfiniteLua value')
        return repr(value)
    if isinstance(value, list): return '{' + ','.join(lua(v) for v in value) + '}'
    return '{' + ','.join('['+lua(k)+']='+lua(v) for k,v in value.items()) + '}'


def admission(case, record):
    errors = []
    frames = record.get('frames', [])
    ticks = [f['tick'] for f in frames]
    if ticks != list(range(case['duration_ticks']+1)):
        errors.append('missing/duplicate native frame ticks')
    orders = record.get('orders', [])
    if case.get('capture_attack_events'):
        actor=case['orders'][0]['actor']
        for event in ('start','release','landed'):
            if sum(e.get('event')==event and e.get('actor')==actor for e in record.get('attack_events',[]))<3:
                errors.append('missing direct native attack events '+event)
    if len(orders) != len(case['orders']): errors.append('missing/extra executed orders')
    for actual, expected in zip(orders, case['orders']):
        if any(actual.get(k) != v for k,v in expected.items()): errors.append('command/tick differs from manifest')
    initial = {u['id']:u for u in frames[0]['units']} if frames else {}
    if case.get('capture_raze_geometry'):
        for unit in initial.values():
            value=unit.get('hull_radius')
            if not isinstance(value,(int,float))or not math.isfinite(value)or value<0:
                errors.append('missing native hull radius '+unit['id'])
            if unit['kind']=='hero':
                geometry=unit.get('raze_geometry',[])
                if len(geometry)!=3 or any(not all(isinstance(g.get(k),(int,float))and math.isfinite(g[k])and g[k]>0 for k in('distance','radius'))for g in geometry):
                    errors.append('missing native raze geometry '+unit['id'])
    if case.get('capture_attachments'):
        for frame in frames:
            for unit in frame['units']:
                if unit['kind']!='hero':continue
                for key in ('attach_attack1','attach_attack2'):
                    point=unit.get('attachments',{}).get(key,{})
                    if not all(isinstance(point.get(axis),(int,float))and math.isfinite(point[axis])for axis in('x','y','z')):
                        errors.append('missing required native attachment '+unit['id']+'/'+key);break
            if any('required native attachment'in e for e in errors):break
    if case['mode']=='natural':
        births={u['id'] for f in frames for u in f['units'] if u['kind']!='hero'}
        if case.get('warm_start'):
            if len([u for u in initial.values()if u['kind']!='hero'])!=8:errors.append('warm start requires complete first8creeps')
            if not frames or frames[0].get('dota_time',0)<=0 or frames[-1].get('dota_time',30)>=30:
                errors.append('warm-start route interval must be strictly between wave spawn ticks0and30s')
            if not record.get('derived_from_native_sha256')or record.get('source_start_tick')!=case['warm_start']['source_start_tick']:
                errors.append('warm start lacks source provenance')
        else:
            if len(births)<8*case.get('expected_waves',2):errors.append('missing declared genuine mid waves')
            if not frames or not -1.1<=frames[0].get('dota_time',100)<-.8:errors.append('natural capture must start before first wave at about-1s')
            if any(u['kind']!='hero'for u in initial.values()):errors.append('natural initial state already contains a wave')
    for u in case['units']:
        got = initial.get(u['id'])
        if not got: errors.append('missing initial unit '+u['id']); continue
        if got['team'] != u['team'] or got['kind'] != u['kind']:
            errors.append('initial identity differs '+u['id'])
        if math.hypot(got['x']-u['x'],got['y']-u['y']) > 16:
            errors.append('initial placement differs '+u['id'])
        if 'health' in u and abs(got['health']-u['health']) > 1:
            errors.append('initial health differs '+u['id'])
        if 'health_fraction'in u and abs(got['health']-math.floor(got['max_health']*u['health_fraction']))>1:
            errors.append('initial health fraction differs '+u['id'])
        if case['mode']=='controlled'and u['kind']=='hero'and u.get('level',1)>1:
            level=u['level'];presence=4 if level>=9 else 3 if level>=8 else 2 if level>=4 else 1 if level>=2 else 0
            expected={**{f'nevermore_shadowraze{i}':min(4,(level+1)//2)for i in(1,2,3)},
                      'nevermore_requiem':int(level>=6),'nevermore_dark_lord':presence}
            if any(got.get('abilities',{}).get(k)!=v for k,v in expected.items()):errors.append('native ability build differs '+u['id'])
        if case.get('damage_mode')=='fixed_base_roll':
            control=got.get('damage_control',{})
            required=('mean_before','mean_after','reported_min','reported_max')
            if not all(isinstance(control.get(k),(int,float))and math.isfinite(control[k])for k in required):
                errors.append('missing fixed-roll mean preservation evidence '+u['id'])
            elif abs(control['mean_after']-control['mean_before'])>.001 or abs(got.get('damage',math.inf)-control['mean_before'])>.001 or abs(control['reported_min']-control['reported_max'])>.001:
                errors.append('fixed roll changed mean damage or retained variance '+u['id'])
    return {'passed': not errors, 'issues': errors}


def warm_start(source, source_sha256, start_tick, ticks, case_id=None, through_contact=False):
    """Start before combat and optionally retain the first engagement."""
    if not source.get('capture_passed')or not source.get('complete'):raise ValueError('Complete admitted source required')
    original=next((c for c in source['cases']if c['mode']=='natural'and(case_id is None or c['id']==case_id)),None)
    if original is None:raise ValueError('Missing natural source case')
    record=next(r for r in source['records']if r['id']==original['id'])
    if not admission(original,record)['passed']:raise ValueError('Source failed admission')
    if start_tick<0 or ticks<1 or start_tick+ticks>original['duration_ticks']:raise ValueError('Invalid warm-start interval')
    frames=copy.deepcopy(record['frames'][start_tick:start_tick+ticks+1])
    if frames[0]['dota_time']<=0 or frames[-1]['dota_time']>=30:raise ValueError('Warm start must be strictly after time0 and finish before30s')
    checked=frames[:1]if through_contact else frames
    if any(u.get('attack_target')for f in checked for u in f['units']):raise ValueError('Route isolation must finish before an observed attack')
    if any(u['health']<u['max_health']-1.2 for f in checked for u in f['units']):raise ValueError('Route isolation contains damage')
    if record.get('orders'):raise ValueError('Route isolation requires passive heroes')
    for frame in frames:frame['tick']-=start_tick
    initial=frames[0]['units']
    case=copy.deepcopy(original)
    case.update(duration_ticks=ticks,units=[{k:u[k]for k in('id','kind','team','x','y','facing','health')}for u in initial],
                warm_start=dict(source_start_tick=start_tick,source_case_id=original['id'],
                                source_native_sha256=source_sha256,
                                through_contact=through_contact,
                                hidden_state_assumption='Initial state precedes attack/damage; lane goal starts at the production route endpoint. Native private AI phase and stock damage RNG are not observed.'))
    case.pop('id');case['id']='warm-start-'+digest(case)[:16]
    result=copy.deepcopy(record);result.update(id=case['id'],frames=frames,orders=[],derived_from_native_sha256=source_sha256,source_start_tick=start_tick)
    gate=admission(case,result)
    if not gate['passed']:raise ValueError(gate['issues'])
    return dict(schema=VERSION,complete=True,capture_passed=True,error=None,cases=[case],records=[result],
                manifest_sha256=digest([case]),derived_from_native_sha256=source_sha256,training_eligible=False)


def capture(manifest, run, worker, timescale, limit, offset):
    from sf_duel.live.game import Game
    validate_manifest(manifest)
    selected = manifest['cases'][offset:offset+limit]
    if not selected: raise ValueError('Empty selection')
    if any(c.get('warm_start')for c in selected):raise ValueError('Warm-start references are derived from saved native history, not recaptured')
    if any(c['mode']=='natural' for c in selected) and len(selected)!=1:
        raise ValueError('Each natural-wave case requires one fresh map; select limit1')
    run = Path(run).resolve();run.mkdir(parents=True,exist_ok=False)
    bots=run/'bots';bots.mkdir()
    shutil.copy2(ROOT/'bots/hero_selection.lua',bots/'hero_selection.lua')
    shutil.copy2(ROOT/'gpu/bots/lane_conformance.lua',bots/'gpu_server.lua')
    if any(c.get('capture_attack_events')or c.get('capture_damage')for c in selected):shutil.copy2(ROOT/'gpu/bots/lane_attack_events.lua',bots/'lane_attack_events.lua')
    (bots/'lane_cases.lua').write_text('return '+lua(selected)+'\n',encoding='utf-8')
    observer='''local json=require('game/dkjson')
local last=-1
function Think()
 local b=GetBot();if not b or b:GetUnitName()~='npc_dota_hero_nevermore'then return end
 local tick=math.floor(GameTime()*30+.5);if tick==last then return end;last=tick
 local t=DotaTime()
 local targets={b};local tracking={}
 for _,u in pairs(GetUnitList(UNIT_LIST_ALLIED_CREEPS))do
  if u:IsAlive()and(string.find(u:GetUnitName(),'npc_dota_creep_')or string.find(u:GetUnitName(),'siege'))then targets[#targets+1]=u end
 end
 for _,u in ipairs(targets)do
  local p=u:GetLocation();local shots={};local available=type(u.GetIncomingTrackingProjectiles)=='function'
  if available then
   local ok,values=pcall(u.GetIncomingTrackingProjectiles,u);available=ok
   if ok then for _,v in ipairs(values or{})do
    if v.location then
     local caster=v.caster
     shots[#shots+1]={x=v.location.x,y=v.location.y,z=v.location.z,speed=v.velocity,is_attack=v.is_attack,
       caster_team=caster and caster:GetTeam()or nil,caster_name=caster and caster:GetUnitName()or nil}
    end
   end end
  end
  tracking[#tracking+1]={name=u:GetUnitName(),team=u:GetTeam(),x=p.x,y=p.y,available=available,shots=shots}
 end
 print('SF_LANE_TRACKING|'..json.encode({tick=tick,time=t,game_time=GameTime(),team=GetTeam(),targets=tracking}))
 if (t>=-1.2 and t<=1.2)or tick%30==0 then
  local creeps={}
  for _,u in pairs(GetUnitList(UNIT_LIST_ALLIED_CREEPS))do
   if u:IsAlive()and string.find(u:GetUnitName(),'npc_dota_creep_')then local p=u:GetLocation();creeps[#creeps+1]={name=u:GetUnitName(),x=p.x,y=p.y,visible=u:CanBeSeen(),invulnerable=u:IsInvulnerable()}end
  end
  print('SF_LANE_BOT_CLOCK|'..json.encode({tick=tick,time=t,game_time=GameTime(),team=GetTeam(),creeps=creeps}))
 end
end
'''
    for filename,body in [('bot_generic.lua',observer),
                          ('item_purchase_generic.lua','function ItemPurchaseThink() end\n'),
                          ('ability_item_usage_generic.lua','function AbilityUsageThink() end\nfunction AbilityLevelUpThink() end\nfunction CourierUsageThink() end\nfunction ItemUsageThink() end\nfunction BuybackUsageThink() end\n')]:
        (bots/filename).write_text(body)
    if selected[0]['mode']=='natural':shutil.copy2(ROOT/'gpu/bots/lane_observer.lua',bots/'bot_generic.lua')
    records={c['id']:{'id':c['id'],'frames':[],'orders':[],'attack_events':[],'damage_events':[]} for c in selected}
    content=ROOT/f'workers/{worker:02d}/game/dota'
    atomic(run/'request.json',{'manifest_sha256':manifest['manifest_sha256'],'cases':[c['id']for c in selected],
                              'timescale':timescale,'worker':worker,'training_eligible':False})
    error=None;error_traceback=None;done=False;started=time.monotonic();offset_bytes=0;tail='';clocks=[];public=[];tracking=[]
    try:
        with Game(run,'lane-conformance',timescale,30,bot_root=bots,content_root=content,
                  game=content.parents[1],game_port=27900+worker,engine_threads=2) as game:
            if selected[0]['mode']=='controlled':game.command('dota_creeps_no_spawning 1')
            game.command('script_reload_code bots/gpu_server')
            # At1x a natural case also waits through the actual90s pregame.
            pregame_wait=90/timescale if selected[0]['mode']=='natural'else 0
            deadline=time.monotonic()+60+pregame_wait+sum((c['duration_ticks']/30+3)/timescale for c in selected)
            while time.monotonic()<deadline:
                if game.process.poll() is not None:raise RuntimeError('Dedicated Dota exited')
                with game.log.open(errors='replace')as f:
                    f.seek(offset_bytes);text=tail+f.read();offset_bytes=f.tell()
                lines=text.split('\n');tail=lines.pop()
                for line in lines:
                    if 'SF_LANE_ERROR|'in line or 'Script Runtime Error:'in line:raise RuntimeError(line)
                    for tag,key in [('FRAME','frames'),('ORDER','orders'),('ATTACK_EVENT','attack_events'),('DAMAGE_EVENT','damage_events')]:
                        marker='SF_LANE_'+tag+'|'
                        if marker in line:
                            value=json.loads(line.split(marker,1)[1]);records[value.pop('case')][key].append(value)
                    if 'SF_LANE_DONE|'in line:done=True
                    if 'SF_LANE_BOT_CLOCK|'in line:clocks.append(json.loads(line.split('SF_LANE_BOT_CLOCK|',1)[1]))
                    if 'SF_LANE_PUBLIC|'in line:public.append(json.loads(line.split('SF_LANE_PUBLIC|',1)[1]))
                    if 'SF_LANE_TRACKING|'in line:tracking.append(json.loads(line.split('SF_LANE_TRACKING|',1)[1]))
                atomic(run/'progress.json',{'pid':game.pid,'frames':sum(len(r['frames'])for r in records.values()),
                       'cases_done':sum(len(r['frames'])>=c['duration_ticks']+1 for c,r in [(c,records[c['id']])for c in selected]),
                       'target':len(selected),'wall_seconds':time.monotonic()-started})
                if done:break
                time.sleep(.05)
            if not done:raise TimeoutError('Bounded native scenario batch incomplete')
    except BaseException as exc:
        error=repr(exc)
        error_traceback=traceback.format_exc()
    server_records=None
    if selected[0]['mode']=='natural'and error is None and done:
        server_records=copy.deepcopy(list(records.values()))
        try:records[selected[0]['id']]=public_natural_reference(selected[0],records[selected[0]['id']],public)
        except ValueError as exc:error=repr(exc)
    for c in selected:records[c['id']]['admission']=admission(c,records[c['id']])
    result={'schema':VERSION,'manifest_sha256':manifest['manifest_sha256'],'cases':selected,
            'records':list(records.values()),'complete':done and error is None,
            'capture_passed':done and error is None and all(r['admission']['passed']for r in records.values()),
            'error':error,'error_traceback':error_traceback,'wall_seconds':time.monotonic()-started,'training_eligible':False,'bot_clock_samples':clocks,
            'source_sha256':hashlib.sha256((bots/'gpu_server.lua').read_bytes()).hexdigest(),
            'server_records':server_records,'public_samples':public,'tracking_projectile_samples':tracking,
            'bot_source_sha256':hashlib.sha256((bots/'bot_generic.lua').read_bytes()).hexdigest(),
            'dota_executable_sha256':hashlib.sha256((content.parents[1]/'game/bin/win64/dota2.exe').read_bytes()).hexdigest()}
    atomic(run/'native.json',result)
    if not result['capture_passed']:raise RuntimeError('Native capture invalid: '+str(error or [r['admission']for r in records.values()]))
    return result


def public_natural_reference(case, server_record, public):
    """Whole paired BotAPI frames; never mix HP/position from different ticks."""
    by_tick={}
    for frame in public:
        group=by_tick.setdefault(frame['tick'],{})
        if frame['team']in group:raise ValueError('Duplicate public frame')
        group[frame['team']]=frame
    paired=[(tick,group)for tick,group in sorted(by_tick.items())if set(group)=={2,3}]
    origin=next(((tick,group)for tick,group in paired if -1.1<=group[2]['time']<=-1.),None)
    if origin is None:raise ValueError('Paired public pre-wave origin missing')
    first=origin[0];out=[];previous={}
    for elapsed in range(case['duration_ticks']+1):
        tick=first+elapsed;group=by_tick.get(tick,{})
        if set(group)!={2,3}:raise ValueError('Missing paired public frame at'+str(tick))
        units=[dict(u)for team in (2,3)for u in group[team]['units']]
        known={**previous,**{u['id']:u for u in units}}
        for u in units:
            target=u.pop('attack_target_ref',None)
            if target is not None:
                candidates=[v for v in known.values()if v['name']==target['name']and v['team']==target['team']]
                matches=sorted((math.hypot(v['x']-target['x'],v['y']-target['y']),v['id'])for v in candidates)
                if not matches or matches[0][0]>16 or(len(matches)>1 and abs(matches[1][0]-matches[0][0])<1e-6):raise ValueError('Public target identity missing/ambiguous')
                u['attack_target']=matches[0][1]
        # Keep a final dead state when an allied creep leaves the active list.
        present={u['id']for u in units}
        for id,old in previous.items():
            if id not in present and old['kind']!='hero':units.append(dict(old,alive=False,health=0,attack_target=None))
        out.append({'tick':elapsed,'dota_time':group[2]['time'],'game_time':group[2]['game_time'],'units':units})
        previous={u['id']:u for u in units}
    return {'id':case['id'],'frames':out,'orders':server_record['orders'],
            'reference_source':'paired publicBotAPI; rawserverpool preserved separately',
            'unavailable_fields':['aggro_target']}


def admitted_subset(source):
    if not source['complete']or source.get('error'):raise ValueError('Only completed error-free captures can supply a partial reference')
    records={r['id']:r for r in source['records']};accepted=[];rejected=[]
    for case in source['cases']:
        check=admission(case,records[case['id']])
        if check['passed']:accepted.append(case)
        else:rejected.append({'id':case['id'],'admission':check})
    if not accepted:raise ValueError('No admitted cases')
    return dict(source,cases=accepted,records=[records[c['id']]for c in accepted],
                capture_passed=True,excluded_cases=rejected,
                partial_reference=bool(rejected))


def compare_case(case, native, gpu):
    gate=admission(case,native)
    if not gate['passed']:return {'id':case['id'],'status':'invalid_native','admission':gate}
    if gpu.get('orders')!=case['orders']:return {'id':case['id'],'status':'invalid_gpu','reason':'GPU consumed command log differs from manifest'}
    if len(gpu.get('frames',[]))!=len(native['frames']):return {'id':case['id'],'status':'invalid_gpu','reason':'framecount'}
    first=None;counts=Counter();maximum=Counter();exact=Counter();total=Counter()
    for nf,gf in zip(native['frames'],gpu['frames']):
        if nf['tick']!=gf['tick']:return {'id':case['id'],'status':'invalid_gpu','reason':'tick mismatch'}
        nu={u['id']:u for u in nf['units']};gu={u['id']:u for u in gf['units']}
        # Engine corpse disposal is not a new death or a missing living NPC.
        for key in list(gu.keys()-nu.keys()):
            if not gu[key]['alive']:gu.pop(key)
        for key in list(nu.keys()-gu.keys()):
            if not nu[key]['alive']:nu.pop(key)
        if set(nu)!=set(gu):
            counts['entity_set']+=1
            if first is None:first={'tick':nf['tick'],'field':'entity_set','native':sorted(nu),'gpu':sorted(gu)}
        for key in sorted(nu.keys()&gu.keys()):
            n,g=nu[key],gu[key]
            position_error=math.hypot(n['x']-g['x'],n['y']-g['y'])
            maximum['position']=max(maximum['position'],position_error)
            if position_error>case['comparison']['position']:
                counts['position']+=1
                if first is None:first={'tick':nf['tick'],'unit':key,'field':'position','native':[n['x'],n['y']],'gpu':[g['x'],g['y']]}
            for field in ('health','mana','x','y','z','facing','alive','level','attack_target','aggro_target','last_hits','denies','raze_stacks'):
                if field in native.get('unavailable_fields',[]):continue
                if field in ('attack_target','aggro_target'):
                    n=dict(n);g=dict(g);n.setdefault(field,None);g.setdefault(field,None)
                elif field in ('last_hits','denies') and n['kind']!='hero':continue
                elif field not in n or field not in g:return {'id':case['id'],'status':'invalid_data','reason':'missing '+field}
                total[field]+=1;exact[field]+=int(n[field]==g[field])
                if field in ('x','y','z'):delta=abs(n[field]-g[field]);threshold=case['comparison']['position']
                elif field=='facing':delta=abs((n[field]-g[field]+180)%360-180);threshold=case['comparison']['facing']
                elif field in ('health','mana'):delta=abs(n[field]-g[field]);threshold=case['comparison']['hp'if field=='health'else'mana']
                else:delta=int(n[field]!=g[field]);threshold=0
                maximum[field]=max(maximum[field],delta)
                if delta>threshold:
                    counts[field]+=1
                    if first is None:first={'tick':nf['tick'],'unit':key,'field':field,'native':n[field],'gpu':g[field]}
    return {'id':case['id'],'family':case['family'],'team':case['actor_team'],
            'status':'mismatch'if counts else'incomplete_observability'if native.get('unavailable_fields')else'pass','first_divergence':first,
            'mismatch_counts':dict(counts),'maximum_errors':dict(maximum),
            'exact_equal_scalars':dict(exact),'compared_scalars':dict(total),
            'rng_note':'StockRNG streams are not synchronized; fixed_base_roll removes only base attack variance. HP differences remain reported and require causality checks.',
            'damage_mode':case['damage_mode'],
            'native_unavailable_fields':native.get('unavailable_fields',[]),
            'native_observed_events':observed_events(native),
            'gpu_observed_events':observed_events(gpu),
            'alignment':'Exact declared ticks; no per-field best-phase search or target relabeling after divergence.'}


def observed_events(record):
    previous={};events=[]
    for frame in record['frames']:
        for row in frame['units']:
            old=previous.get(row['id'])
            if old is not None:
                drop=old['health']-row['health']
                if drop>1.2:events.append({'tick':frame['tick'],'unit':row['id'],'event':'observed_hp_drop','amount':drop})
                for field in ('last_hits','denies'):
                    delta=row.get(field,0)-old.get(field,0)
                    if delta:events.append({'tick':frame['tick'],'unit':row['id'],'event':field,'amount':delta})
                # SF hand changes only when an attack projectile is launched.
                # Native last_attack records the engine release time; report
                # the tick at which that change became observable in each trace.
                native_release=row.get('last_attack',0)>0 and row.get('last_attack')!=old.get('last_attack')
                gpu_release=row.get('kind')=='hero' and 'attack_hand'in row and row['attack_hand']!=old.get('attack_hand',0)
                if native_release or gpu_release:events.append({'tick':frame['tick'],'unit':row['id'],'event':'observed_attack_release'})
            previous[row['id']]=row
    return events


def map_natural_births(native, gpu):
    """Match once at birth by team/kind/minimum spawn distance; never chase targets.

    An ambiguous birth assignment is an invalid comparison, not a pass. A
    one-sided extra birth remains visible as an entity-set mismatch.
    """
    mapping={'radiant':'radiant','dire':'dire'};used=set(mapping.values());evidence=[]
    for nf,gf in zip(native['frames'],gpu['frames']):
        un=[u for u in nf['units']if u['id']not in used and u['alive']]
        ug=[u for u in gf['units']if u['id']not in mapping and u['alive']]
        for team,kind in sorted({(u['team'],u['kind'])for u in un}):
            ns=[u for u in un if (u['team'],u['kind'])==(team,kind)]
            gs=[u for u in ug if (u['team'],u['kind'])==(team,kind)]
            if not ns or len(ns)!=len(gs):continue
            if len(gs)>5:raise ValueError('Too many unmatched newborns; birth sampling invalid')
            scored=sorted((sum((n['x']-g['x'])**2+(n['y']-g['y'])**2 for n,g in zip(ns,order)),tuple(g['id']for g in order))for order in itertools.permutations(gs))
            if len(scored)>1 and abs(scored[1][0]-scored[0][0])<1e-6:
                raise ValueError('Ambiguous natural creep birth identities')
            for n,gid in zip(ns,scored[0][1]):mapping[gid]=n['id'];used.add(n['id'])
            evidence.append({'tick':nf['tick'],'team':team,'kind':kind,'gpu_ids':scored[0][1],
                             'native_ids':[n['id']for n in ns],'squared_distance_cost':scored[0][0]})
    # Copy canonical rows, preserving the immutable raw reference and GPU trace.
    result={'id':gpu['id'],'frames':[],'orders':gpu.get('orders',[])}
    for frame in gpu['frames']:
        units=[]
        for unit in frame['units']:
            row=dict(unit);row['id']=mapping.get(row['id'],row['id'])
            for field in ('attack_target','aggro_target'):
                if row.get(field)is not None:row[field]=mapping.get(row[field],row[field])
            units.append(row)
        result['frames'].append(dict(frame,units=units))
    return result,evidence


def run_suite(a):
    """Serial native batches then GPU; native references survive GPU revisions."""
    manifest=json.loads(Path(a.manifest).read_text());validate_manifest(manifest)
    root=Path(a.run).resolve();root.mkdir(parents=True,exist_ok=True)
    # OS-owned lock releases after crashes; no stale PID-file based exclusion.
    import msvcrt
    with (root/'run.lock').open('a+b')as lock:
        lock.seek(0);lock.write(b'0');lock.flush();lock.seek(0)
        try:msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        except OSError:raise RuntimeError('This suite already has a running coordinator')
        try:
            state_path=root/'status.json'
            if state_path.exists()and not a.resume and not a.recheck:raise ValueError('Existing suite requires --resume or --recheck')
            previous=json.loads(state_path.read_text())if state_path.exists()else{}
            if previous and previous['manifest_sha256']!=manifest['manifest_sha256']:raise ValueError('Cannot resume a different manifest')
            iteration=(previous.get('iteration',0)+1)if a.recheck else previous.get('iteration',1)
            completed=0 if a.recheck else previous.get('completed_cases',0)
            target=min(a.limit,len(manifest['cases'])-a.offset)
            if target<1:raise ValueError('Empty declared cohort')
            if previous and (previous.get('offset',a.offset)!=a.offset or previous['target_cases']!=target):raise ValueError('Cohort changed; use a new run directory')
            exe=Path(a.gpu_exe).resolve();engine=Path(a.engine).resolve()
            exe_hash=hashlib.sha256(exe.read_bytes()).hexdigest()
            if previous and completed and not a.recheck and previous['gpu_executable_sha256']!=exe_hash:raise ValueError('GPU executable changed; use --recheck against cached native references')
            results=[];started=time.monotonic()
            def status(phase,**extra):
                value={'phase':phase,'pid':os.getpid(),'iteration':iteration,
                    'manifest_sha256':manifest['manifest_sha256'],'completed_cases':completed,
                    'target_cases':target,'gpu_executable_sha256':exe_hash,'engine':str(engine),
                    'offset':a.offset,'discovery':a.discovery,
                    'wall_seconds_this_session':time.monotonic()-started,'training_updates':0,**extra}
                atomic(state_path,value)
                project_status=ROOT/'runs/pilot-status.json'
                if project_status.exists():
                    shared=json.loads(project_status.read_text(encoding='utf-8-sig'))
                    shared['gpu_pipeline']['lane_conformance']=dict(value,root=str(root),status_path=str(state_path))
                    shared['gpu_pipeline']['stage']='Lane conformance: '+phase
                    atomic(project_status,shared)
            def child(cmd,log_path,phase,allow_invalid_capture=False):
                with log_path.open('w',encoding='utf-8')as log:
                    process=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                                             creationflags=subprocess.CREATE_NO_WINDOW)
                    status(phase,child_pid=process.pid,command=cmd,log=str(log_path))
                    code=process.wait()
                if code and not allow_invalid_capture:raise RuntimeError(f'{phase} failed exit{code}: {log_path}')
            while completed<target:
                if (root/'stop.json').exists():status('stopped_at_batch_boundary');return
                if shutil.disk_usage(root).free<20*1024**3:raise RuntimeError('Less than20GiB free; stop before another capture')
                offset=a.offset+completed;selection=manifest['cases'][offset:offset+min(a.batch_size,target-completed)]
                if selection[0]['mode']=='natural':selection=selection[:1]
                else:selection=list(itertools.takewhile(lambda c:c['mode']=='controlled',selection))
                native_root=root/'native'/f'{offset:05d}-{len(selection):02d}'
                native_path=native_root/'native.json'
                if not native_path.exists():
                    native_root.parent.mkdir(exist_ok=True)
                    if native_root.exists():raise RuntimeError('Partial capture retained; use a fresh run or repair explicitly: '+str(native_root))
                    child([sys.executable,str(Path(__file__).resolve()),'capture','--manifest',str(Path(a.manifest).resolve()),
                           '--run',str(native_root),'--offset',str(offset),'--limit',str(len(selection)),
                           '--worker',str(a.worker),'--timescale',str(a.timescale)],
                          root/f'native-{offset:05d}.log','native_capture',allow_invalid_capture=a.discovery)
                native=json.loads(native_path.read_text())
                if native['cases']!=selection or not native['complete']or native.get('error'):raise RuntimeError('Native capture incomplete/errored or different cases')
                rejected=[]
                if not native['capture_passed']:
                    if not a.discovery:raise RuntimeError('Native case admission failed; no partial cases counted as pass')
                    admitted=admitted_subset(native)
                    rejected=[{'id':r['id'],'status':'invalid_native','admission':r['admission']}for r in admitted['excluded_cases']]
                    admitted['raw_native_sha256']=hashlib.sha256(native_path.read_bytes()).hexdigest()
                    native_path=native_root/'admitted.json';atomic(native_path,admitted)
                replay_root=root/f'replay-{iteration:03d}';replay_root.mkdir(exist_ok=True)
                gpu_path=replay_root/f'{offset:05d}.gpu.json';report_path=replay_root/f'{offset:05d}.report.json'
                child([str(exe),'replay-lane',str(engine),str(native_path),str(gpu_path)],replay_root/f'{offset:05d}.gpu.log','gpu_replay')
                child([sys.executable,str(Path(__file__).resolve()),'compare','--native',str(native_path),
                       '--gpu',str(gpu_path),'--output',str(report_path)],replay_root/f'{offset:05d}.compare.log','compare')
                report=json.loads(report_path.read_text());report['results'].extend(rejected)
                report['counts']=dict(Counter(r['status']for r in report['results']));atomic(report_path,report)
                results.extend(report['results']);completed+=len(selection)
                status('batch_complete',last_report=str(report_path),counts_this_session=dict(Counter(r['status']for r in results)))
                if not a.discovery and any(r['status'].startswith('invalid')for r in report['results']):raise RuntimeError('Invalid comparison; qualify harness before collecting more')
            # Preserve completed reports from earlier coordinator sessions.
            all_results=[]
            for path in sorted((root/f'replay-{iteration:03d}').glob('*.report.json')):all_results.extend(json.loads(path.read_text())['results'])
            atomic(root/f'summary-{iteration:03d}.json',{'complete':True,'cases':len(all_results),'target':target,
                'counts':dict(Counter(r['status']for r in all_results)),'results':all_results,
                'all_passed':len(all_results)==target and all(r['status']=='pass'for r in all_results),
                'manifest_sha256':manifest['manifest_sha256'],'gpu_executable_sha256':exe_hash,'training_updates':0})
            status('complete',counts=dict(Counter(r['status']for r in all_results)))
        except BaseException as error:
            if 'status'in locals():status('failed',error=repr(error))
            raise
        finally:
            lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1)


def main():
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='command',required=True)
    q=s.add_parser('generate');q.add_argument('--output',required=True);q.add_argument('--count',type=int,default=3000);q.add_argument('--seed',type=int,default=20261104);q.add_argument('--damage-mode',choices=('stock','fixed_base_roll'),default='stock')
    q=s.add_parser('profile');q.add_argument('--manifest',required=True);q.add_argument('--output',required=True)
    q=s.add_parser('capture');q.add_argument('--manifest',required=True);q.add_argument('--run',required=True);q.add_argument('--offset',type=int,default=8);q.add_argument('--limit',type=int,default=8);q.add_argument('--worker',type=int,default=19);q.add_argument('--timescale',type=float,default=1)
    q=s.add_parser('compare');q.add_argument('--native',required=True);q.add_argument('--gpu',required=True);q.add_argument('--output',required=True)
    q=s.add_parser('admit');q.add_argument('--native',required=True);q.add_argument('--output',required=True)
    q=s.add_parser('warm-start');q.add_argument('--native',required=True);q.add_argument('--output',required=True)
    q.add_argument('--case-id');q.add_argument('--start-tick',type=int,required=True);q.add_argument('--ticks',type=int,required=True)
    q.add_argument('--through-contact',action='store_true')
    q=s.add_parser('run');q.add_argument('--manifest',required=True);q.add_argument('--run',required=True)
    q.add_argument('--gpu-exe',required=True);q.add_argument('--engine',required=True)
    q.add_argument('--offset',type=int,default=0);q.add_argument('--limit',type=int,default=3000)
    q.add_argument('--batch-size',type=int,default=16);q.add_argument('--worker',type=int,default=19)
    q.add_argument('--timescale',type=float,default=1);q.add_argument('--resume',action='store_true');q.add_argument('--recheck',action='store_true')
    q.add_argument('--discovery',action='store_true',help='Collect valid cases and preserve invalid-case results; never certify them as passes')
    a=p.parse_args()
    if a.command=='generate':
        result=generate(a.count,a.seed,a.damage_mode);atomic(a.output,result)
        print(json.dumps({'cases':len(result['cases']),'families':dict(Counter(c['family']for c in result['cases'])),'sha256':result['manifest_sha256']}))
    elif a.command=='profile':
        result=representative_profile(json.loads(Path(a.manifest).read_text()));atomic(a.output,result)
        print(json.dumps({'cases':len(result['cases']),'sha256':result['manifest_sha256']}))
    elif a.command=='capture':
        if not 1<=a.timescale<=4 or not 1<=a.limit<=32:raise ValueError('Bounded batch: timescale1..4,limit1..32')
        result=capture(json.loads(Path(a.manifest).read_text()),a.run,a.worker,a.timescale,a.limit,a.offset)
        print(json.dumps({'capture_passed':result['capture_passed'],'cases':len(result['cases'])}))
    elif a.command=='run':
        if not 1<=a.batch_size<=32 or a.limit<1 or a.offset<0 or not 1<=a.timescale<=4:raise ValueError('Invalid bounded suite settings')
        run_suite(a)
    elif a.command=='admit':
        source=json.loads(Path(a.native).read_text());result=admitted_subset(source)
        result['raw_native_sha256']=hashlib.sha256(Path(a.native).read_bytes()).hexdigest()
        atomic(a.output,result);print(json.dumps({'accepted':len(result['cases']),'excluded':len(result['excluded_cases'])}))
    elif a.command=='warm-start':
        path=Path(a.native);result=warm_start(json.loads(path.read_text()),hashlib.sha256(path.read_bytes()).hexdigest(),a.start_tick,a.ticks,a.case_id,a.through_contact)
        atomic(a.output,result);print(json.dumps({'cases':len(result['cases']),'frames':len(result['records'][0]['frames']),'source_sha256':result['derived_from_native_sha256']}))
    else:
        native=json.loads(Path(a.native).read_text());gpu=json.loads(Path(a.gpu).read_text())
        if not native['capture_passed'] or gpu['native_sha256']!=hashlib.sha256(Path(a.native).read_bytes()).hexdigest():raise ValueError('Reference is invalid or GPU trace is for a different native fixture')
        rows={r['id']:r for r in native['records']};sim={r['id']:r for r in gpu['records']}
        results=[]
        for c in native['cases']:
            trace=sim[c['id']];mapping=[]
            if c['mode']=='natural':
                try:trace,mapping=map_natural_births(rows[c['id']],trace)
                except ValueError as error:
                    results.append({'id':c['id'],'status':'invalid_mapping','reason':str(error)});continue
            result=compare_case(c,rows[c['id']],trace);result['birth_mapping']=mapping;results.append(result)
        report={'schema':VERSION,'manifest_sha256':native['manifest_sha256'],'results':results,'counts':dict(Counter(r['status']for r in results)),'training_eligible':False,'tolerances':TOLERANCES}
        atomic(a.output,report);print(json.dumps(report['counts']))


if __name__=='__main__':main()
