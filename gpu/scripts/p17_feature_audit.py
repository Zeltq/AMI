"""Per-feature coverage/distributions against a saved simulator observation fixture."""
import argparse
import csv
import gzip
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from sf_duel.chio_observations import Observer
from sf_duel.spec import UNIT_FIELDS,COMBAT_UNIT_FIELDS,TREE_FIELDS,PROJECTILE_FIELDS,COMBAT_PROJECTILE_FIELDS,SKILL_FIELDS,ITEM_FIELDS,GLOBAL_FIELDS
from sf_duel.timing_observations import TIMING_FIELDS
from sf_duel.enemy_observations import FIELDS as ENEMY_FIELDS

DIRECT=dict(x='GetLocation',y='GetLocation',z='GetLocation',facing_cos='GetFacing',facing_sin='GetFacing',
 health='GetHealth',health_max='GetMaxHealth',mana='GetMana',mana_max='GetMaxMana',health_regen='GetHealthRegen',
 mana_regen='GetManaRegen',move_speed='GetCurrentMovementSpeed',attack_damage='GetAttackDamage',attack_range='GetAttackRange',
 attack_interval='GetSecondsPerAttack',attack_point='GetAttackPoint',projectile_speed='GetAttackProjectileSpeed',
 armor='GetArmor',magic_resist='GetMagicResist',level='GetLevel',gold='GetGold (self only)',xp_to_next='GetXPNeededToLevel (self only)',
 last_hits='GetLastHits (self only)',denies='GetDenies (self only)',souls='GetModifierStackCount (self only)',
 attack_cycle='GetAnimCycle',active_spell='GetCurrentActiveAbility (self only)',attack_animation='GetAnimActivity',
 since_attack='GetLastAttackTime',ready_in='GetLastAttackTime/GetSecondsPerAttack',
 inventory='GetItemInSlot/GetCurrentCharges',backpack='GetItemInSlot/GetCurrentCharges',stash='GetItemInSlot/GetCurrentCharges',
 courier='GetCourier/GetItemInSlot',cooldown='GetCooldownTimeRemaining (self only)',castable='IsFullyCastable (self only)',
 mana_cost='GetManaCost (self only)',cast_point='GetCastPoint (self only)',channel_time='GetChannelTime (self only)',
 slot='FindItemSlot (self only)',time='DotaTime',next_permanent='local economy',side='GetTeam')


def source(name):
    if name in DIRECT:return DIRECT[name]
    if name.startswith('oracle_'):return 'forbidden; value=known=0'
    if name.startswith('hp_lag'):return 'GetHealth/GetMaxHealth; identity+generation; missing sample=unknown'
    if name.startswith(('own_raze','enemy_raze','relative_angle','attack_reach')):return 'visible GetLocation/GetFacing; own ability specials; GetBoundingRadius/GetAttackRange'
    if name.startswith('own_cast'):return 'accepted queue + IsCastingAbility/IsUsingAbility/GetCurrentActiveAbility'
    if name in ('fear','flask','tango','clarity','raze_stacks','raze_remaining','fear_remaining'):return 'HasModifier/GetModifierStackCount/GetModifierRemainingDuration; private fields masked'
    if name.startswith(('cast_','since_cast_','spell_level_')):return 'visible animation + GetAbilityByName/GetLevel; causal cast onset clock'
    if name.startswith('is_'):return 'IsAttacking/IsUsingAbility/IsCastingAbility/IsChanneling; visible animation for enemy'
    if name.startswith(('creep_','team_')):return 'GetUnitName/GetTeam'
    if name.startswith(('attack_target','incoming_','next_projectile','caster_','target_')) or name=='damage_spread':
        return 'visible GetAttackTarget/GetIncomingTrackingProjectiles; calibrated public damage table; unknown caster=0'
    if name in ('health_fraction','mana_fraction'):return 'current/max resource from visible Bot API'
    if name in ('dx','dy','distance'):return 'visible GetLocation - own GetLocation (or GetTreeLocation)'
    if name=='speed':return 'GetIncomingTrackingProjectiles.velocity or visible caster GetAttackProjectileSpeed'
    return 'Bot API snapshot; see bots/agent.lua and spec.py'


def specs():
    unit={f'hp_lag{i}':1 for i in range(12)}|{f'own_raze{i}':1 for i in range(3)}|{'attack_reach':1,'relative_angle':1}
    hero=unit|{f'enemy_raze{i}':1 for i in range(3)}|{'own_cast_active':1}|{f'own_cast_skill{i}':1 for i in range(4)}
    groups={k:[UNIT_FIELDS] for k in ('heroes','creeps','towers')}
    groups.update(hero_combat=[COMBAT_UNIT_FIELDS,TIMING_FIELDS,ENEMY_FIELDS,hero],creep_combat=[COMBAT_UNIT_FIELDS,TIMING_FIELDS,unit],
                  tower_combat=[COMBAT_UNIT_FIELDS,TIMING_FIELDS,unit],trees=[TREE_FIELDS],projectiles=[PROJECTILE_FIELDS],
                  projectile_combat=[COMBAT_PROJECTILE_FIELDS],skills=[SKILL_FIELDS],items=[ITEM_FIELDS],enemy_items=[ITEM_FIELDS],global_features=[GLOBAL_FIELDS])
    result=[]
    for group,segments in groups.items():
        offset=0
        for fields in segments:
            for i,name in enumerate(fields):result.append((group,name,offset+i,offset+len(fields)+i,source(name)))
            offset+=2*len(fields)
    for i in range(55):result.append(('previous',f'previous_{i}',i,None,'executor last issued action/direction/target category'))
    for i,src in enumerate(('GetHeightLevel','IsLocationVisible/IsLocationPassable','IsLocationVisible','visible lane creep occupancy')):
        result.append(('terrain',f'channel_{i}',i,None,src))
    for name in ('hero_valid','creep_valid','creep_ally','tree_valid','projectile_valid','action_mask','enemy_target_mask','ally_target_mask'):
        result.append((name,'bit',0,None,'CanBeSeen/IsAlive; public categories; cast queue; own IsFullyCastable'))
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--trace',type=Path,required=True);p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((a.reference/'manifest.json').read_text());sim={}
    for k,e in manifest['tensors'].items():sim[k]=np.fromfile(a.reference/e['file'],dtype='<f4').reshape(e['shape'])
    rows=specs();stats={i:dict(values=[],known=0,total=0) for i in range(len(rows))};observer=Observer();count=0;previous=None
    decisions={r['observed_tick']:r for r in map(json.loads,(a.trace.parent/'decisions.jsonl').read_text().splitlines())}
    def flatten(group,data):
        if group=='terrain':return np.moveaxis(data,1,-1).reshape(-1,4)
        if group.endswith(('_valid','_mask')) or group=='creep_ally':return data.reshape(-1,1)
        return data.reshape(-1,data.shape[-1])
    with gzip.open(a.trace,'rt',encoding='utf-8') as f:
        for line in f:
            s=json.loads(line);b=observer.encode(s,previous,pending_cast=s.get('pending_cast'));count+=1
            previous=decisions.get(s['tick'],previous)
            for i,(g,n,v,k,_) in enumerate(rows):
                raw=flatten(g,b[g].numpy());known=np.ones(len(raw),bool) if k is None else raw[:,k]>.5
                stats[i]['values'].extend(raw[known,v].tolist());stats[i]['known']+=int(known.sum());stats[i]['total']+=len(raw)
    output=[]
    for i,(g,n,v,k,src) in enumerate(rows):
        st=stats[i];d=np.asarray(st['values']);raw=flatten(g,sim[g]);known=np.ones(len(raw),bool) if k is None else raw[:,k]>.5;r=raw[known,v]
        dm=float(d.mean()) if len(d) else None;sm=float(r.mean()) if len(r) else None
        output.append(dict(group=g,feature=n,source=src,check='distribution; different trajectories, not state parity',
          dota_known=st['known'],dota_total=st['total'],sim_known=int(known.sum()),sim_total=len(raw),
          dota_mean=dm,sim_mean=sm,mean_delta=dm-sm if dm is not None and sm is not None else None,
          dota_min=float(d.min()) if len(d) else None,dota_max=float(d.max()) if len(d) else None))
    with (a.output/'features.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(output[0]));writer.writeheader();writer.writerows(output)
    unknown=[r for r in output if r['dota_known']==0]
    result=dict(decisions=count,features=len(rows),features_observed=len(rows)-len(unknown),
      source=str(a.trace),reference=str(a.reference),oracle_nonzero=sum(r['dota_known'] for r in output if r['feature'].startswith('oracle_')),
      limitation='Saved simulator fixture vs live Dota feature distributions; no simulator evaluation or GPU launched. Exact tests cover geometry, HP missing samples/generation and masks, not full engine replay.')
    (a.output/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps(result))


if __name__=='__main__':main()
