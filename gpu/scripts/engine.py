"""Build the simulator's numeric tables from installed rules and runtime capture."""
import argparse,json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from sf_duel.spec import UNIT_FIELDS,ITEMS,TOWERS

def number(value,index=0,default=None):
    if isinstance(value,dict):value=value.get('value',default)
    if value is None:
        if default is None:raise ValueError('Missing rule')
        return default
    values=str(value).split();return float(values[min(index,len(values)-1)])

def main():
    p=argparse.ArgumentParser();p.add_argument('calibration');a=p.parse_args();root=Path(a.calibration)
    cap=json.loads((root/'calibration.json').read_text());rules=json.loads((root/'rules.json').read_text())
    hero=rules['hero']['DOTAHeroes']['npc_dota_hero_nevermore'];abilities=hero['AbilityDefinitions']
    fields=list(UNIT_FIELDS)
    def prototype(r):
        v={k:0. for k in fields}
        for target,source in {'health':'max_hp','health_max':'max_hp','mana':'max_mana','mana_max':'max_mana',
            'attack_damage':'damage','attack_range':'range','attack_interval':'interval','attack_point':'attack_point',
            'projectile_speed':'projectile_speed','move_speed':'speed','armor':'armor','magic_resist':'magic_resist',
            'health_regen':'hp_regen','mana_regen':'mana_regen','level':'level','x':'x','y':'y','z':'z'}.items():v[target]=r[source]
        return [v[k] for k in fields]
    prototypes=[]
    for team in ('2','3'):
        r=dict(cap['heroes'][team]);r['hp_regen']-=r['max_hp']*.05 # fountain's measured +5% maximum HP/s
        r['mana_regen']=number(hero['StatusManaRegen'])+number(hero['AttributeBaseIntelligence'])*.05
        prototypes.append(prototype(r))
    for name in TOWERS:
        r=cap['towers'][name]
        if any(r[k]<0 for k in ('max_hp','damage','interval','armor')):raise ValueError(f'Unknown own tower stats: {name}')
        prototypes.append(prototype(r))
    for kind in ('melee','ranged'):
        prototypes.append(prototype(cap['creeps']['npc_dota_creep_goodguys_'+kind]))
    def installed_unit(rule,position=None):
        row={k:0. for k in UNIT_FIELDS}
        for target,key in {'health':'StatusHealth','health_max':'StatusHealth','attack_range':'AttackRange','attack_interval':'AttackRate',
          'attack_point':'AttackAnimationPoint','projectile_speed':'ProjectileSpeed','move_speed':'MovementSpeed','armor':'ArmorPhysical','level':'Level'}.items():row[target]=number(rule.get(key),default=0)
        row['attack_damage']=(number(rule['AttackDamageMin'])+number(rule['AttackDamageMax']))/2
        row['magic_resist']=number(rule.get('MagicalResistance'),default=0)/100
        row['attack_interval']/=number(rule.get('BaseAttackSpeed'),default=100)/100
        row['health_regen']=number(rule.get('StatusHealthRegen'),default=0)
        if position:row.update({k:position[k] for k in ('x','y','z')})
        return [row[k] for k in UNIT_FIELDS]
    prototypes.append(installed_unit(rules['units']['npc_dota_goodguys_siege']))
    fountain_capture=json.loads((ROOT/'runs/gpu/record-fountains.json').read_text())
    fountains=fountain_capture['fountains']
    prototypes.extend(installed_unit(rules['units']['dota_fountain'],p) for p in fountains)
    prototypes.extend(installed_unit(rules['units'][p['name']],p) for p in fountain_capture['background_towers'])
    spells=[]
    for name in ('nevermore_shadowraze1','nevermore_shadowraze2','nevermore_shadowraze3','nevermore_requiem'):
        ability=abilities[name];values=ability['AbilityValues'];levels=[]
        for level in range(4):
            levels.append([number(values.get('shadowraze_damage',ability.get('AbilityDamage')),level),
                number(ability['AbilityManaCost'],level),number(values['AbilityCooldown'],level),
                number(values.get('shadowraze_range',values.get('requiem_radius')),level),
                number(values.get('shadowraze_radius',values.get('requiem_line_width_start')),level),
                number(ability['AbilityCastPoint'],level),number(values.get('stack_bonus_damage'),level,0),
                number(values.get('duration',values.get('requiem_slow_duration')),level)])
        spells.append(levels)
    item_rows=[]
    for name in ITEMS:
        r=rules['items'][name];v=r.get('AbilityValues',{})
        item_rows.append([number(r.get('ItemCost'),default=0),number(r.get('ItemInitialCharges'),default=1),
            number(r.get('AbilityCooldown'),default=0),number(r.get('AbilityCastPoint'),default=0),
            number(v.get('health_regen',v.get('hp_regen')),default=0),number(v.get('buff_duration',v.get('duration')),default=0),
            number(v.get('replenish_amount',v.get('mana_regen')),default=0),number(r.get('ItemStockInitial'),default=-1),
            number(r.get('ItemStockTime'),default=0),number(v.get('bonus_strength'),default=0),
            number(v.get('bonus_agility'),default=0),number(v.get('bonus_intellect'),default=0),
            number(v.get('bonus_attack_speed'),default=0),number(v.get('bonus_armor'),default=0),
            number(v.get('bonus_movement_speed'),default=0),number(v.get('bonus_damage'),default=0),
            number(r.get('AbilityManaCost'),default=0),number(r.get('ItemStockMax'),default=-1),
            number(r.get('AbilityChannelTime'),default=0),number(v.get('bonus_health'),default=0),number(v.get('bonus_health_regen'),default=0)])
    h=[number(hero[k]) for k in ('AttributeBaseStrength','AttributeBaseAgility','AttributeBaseIntelligence',
        'AttributeStrengthGain','AttributeAgilityGain','AttributeIntelligenceGain','MovementTurnRate')]
    h += [number(abilities['nevermore_dark_lord']['AbilityValues']['presence_armor_reduction'],i) for i in range(4)]
    h += [number(abilities['nevermore_necromastery']['AbilityValues']['necromastery_damage_per_soul']),
        number(abilities['nevermore_necromastery']['AbilityValues']['necromastery_max_souls'])]
    req=abilities['nevermore_requiem']['AbilityValues']
    h += [number(req[k]) for k in ('requiem_line_speed','requiem_line_width_start','requiem_line_width_end',
        'requiem_radius','requiem_soul_conversion','max_soul_release','requiem_slow_duration','requiem_slow_duration_max')]
    h.append(number(abilities['nevermore_shadowraze3']['AbilityValues']['stack_bonus_damage']['special_bonus_unique_nevermore_7']))
    h += [number(rules['items']['item_tpscroll']['AbilityValues'][k]) for k in ('minimum_distance','maximum_distance')]
    projectile_capture=json.loads((ROOT/'runs/gpu/record-projectile-rules.json').read_text())
    h += [projectile_capture['muzzle_forward'],projectile_capture['muzzle_side']]
    h.append(number(req['requiem_reduction_ms'])/100)
    dice=[hero]+[rules['units'][name] for name in ('npc_dota_creep_goodguys_melee','npc_dota_creep_goodguys_ranged','npc_dota_goodguys_siege','npc_dota_goodguys_tower1_mid','dota_fountain')]
    h += [(number(rule['AttackDamageMax'])-number(rule['AttackDamageMin']))/2 for rule in dice]
    lifecycle=json.loads((ROOT/'runs/gpu/record-lifecycle-rules.json').read_text())
    h += [float(lifecycle['respawns'][team][k]) for team in ('2','3') for k in ('x','y','z')]
    combat=rules['combat'];heavy=combat['creep_siege']['AbilityValues'];piercing=combat['creep_piercing']['AbilityValues']
    matrix=[]
    for attacker in range(5):
      for defender in range(5):
        value=1.
        if defender==0 and attacker==2:value*=1+number(combat['creep_irresolute']['AbilityValues']['hero_damage_penalty'])/100
        if attacker==3:
          key='hero_damage_penalty' if defender==0 else 'heavy_damage_penalty' if defender in (1,4) else 'creep_damage_bonus'
          value*=1+number(piercing[key])/100
        if defender in (1,4) and attacker!=4:
          value*=1+number(heavy['incoming_hero_damage_penalty' if attacker==0 else 'incoming_basic_damage_penalty'])/100
        if attacker==4 and defender==1:value*=1+number(heavy['bonus_building_damage'])/100
        matrix.append(value)
    xp_capture=json.loads((ROOT/'runs/gpu/record-xp-rules.json').read_text())
    creep_capture=json.loads((ROOT/'runs/gpu/record-creep-rules.json').read_text())
    spawns=[{'x':x,'y':y,'z':256.} for team in ('2','3') for x,y in creep_capture['train']['spawns'][team]]
    lanes=[spawns[1],{'x':-3190.,'y':-2926.,'z':256.},{'x':-1544.,'y':-1408.,'z':256.},
        {'x':-500.,'y':-300.,'z':128.},{'x':524.,'y':652.,'z':256.},{'x':2496.,'y':2112.,'z':256.},spawns[5]]
    xp=[240.,400.,520.,600.,680.,760.,800.,900.]+[float(1000+100*i) for i in range(21)]+[0.]
    for level,value in xp_capture['thresholds'].items():xp[int(level)-1]=float(value)
    for prototype_row in prototypes[:2]:prototype_row[fields.index('xp_to_next')]=xp[0]
    result={'format':1,'calibration':str(root.resolve()),'build':cap['build'],'geometry':'geometry.f32','trees':cap['trees'],
        'navigation':'navigation.u32',
        'prototypes':prototypes,'spells':spells,'items':item_rows,'hero':h,'combat':matrix,'experience':xp,'spawns':spawns,'lane':lanes,'fountains':fountains,'bases':fountains,
        'extra_vision':json.loads((ROOT/'runs/gpu/record-base-vision.json').read_text())['sources']+json.loads((ROOT/'runs/gpu/dota-authored-fow-001/fow.json').read_text())['revealers'],
        'fow_blockers':'fow_blockers.f32',
        'movement_geometry':'movement-geometry.f32',
        'pre_game_seconds':83.,'decision_ticks':4,'offset_ticks':6,'max_seconds':600.,
        'behavior_validated':False,'approximations':['Runtime fountain regen removed using measured 5% max-HP aura; validate away from fountain.',
            'Engine-derived attribute formulas and projectile ordering require scenario validation.',
            'Courier uses a frozen payload, four-second dispatch cadence and distance/800+4 travel approximation; no courier combat.',
            'XP thresholds through level 8 inferred on training records with separate held-out checks; levels 9+ use explicit +100 extrapolation pending installed-runtime validation.',
            'Mid-lane waypoints use measured tower positions and train-record first-wave positions, not original engine navigation graph.']}
    result['approximations'] += ['Pregame level1 respawn12s at separately calibrated native respawn coordinates; soul retention ceil(0.7*souls), death loss from unreliable gold only. Native executor remembers pregame deaths at time0.',
        'Hero gold uses measured level1/pre-game first-hero326, first-tower202 and later-tower133 values; later hero kill133 is an unvalidated approximation, not a general Dota bounty formula. Hero XP100+floor(0.13*cumulativeXP) is checked on level1 native packets; higher levels need validation.']
    import shutil
    shutil.copyfile(ROOT/'runs/gpu/dota-authored-fow-001/fow_blockers.f32',root/'fow_blockers.f32')
    shutil.copyfile(ROOT/'runs/gpu/dota-native-grid-001/movement-geometry.f32',root/'movement-geometry.f32')
    (root/'engine.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({'output':str(root/'engine.json'),'prototypes':len(prototypes),'behavior_validated':False}))
if __name__=='__main__':main()
