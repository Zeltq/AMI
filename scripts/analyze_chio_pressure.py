"""Pressure episodes from saved v1 Dota traces; P16 reports remain read-only."""
import bisect
import collections
import gzip
import json
import math
from pathlib import Path
from analyze_chio_dota_transfer import low_priority

ROOT=Path(__file__).resolve().parents[1]
RUN=Path('C:/AMI/runs/p17')


def dist(a,b):return math.hypot(a['x']-b['x'],a['y']-b['y'])


def at(rows,times,t):
    i=bisect.bisect_right(times,t)-1
    return rows[max(0,i)] if rows else None


def analyze(game):
    folder=Path(game['run']);team=game['model_team'];other=5-team
    events=[];heroes={2:{},3:{}};v9=[]
    for line in (folder/'dota.log').open(errors='replace'):
        if 'SF_PLAY_EVENT|' in line:
            e=json.loads(line.split('SF_PLAY_EVENT|',1)[1]);events.append(e)
            if e['kind']=='hero':heroes[e['team']][e['time']]=e
        elif 'SF_V9_STATUS|' in line:v9.append(json.loads(line.split('SF_V9_STATUS|',1)[1]))
    heroes={t:sorted(v.values(),key=lambda x:x['time']) for t,v in heroes.items()}
    ht={t:[r['time'] for r in rows] for t,rows in heroes.items()}
    obs=[];tower=None
    with gzip.open(folder/'policy/observations.jsonl.gz','rt') as f:
        for line in f:
            s=json.loads(line);e=s.get('enemy')
            obs.append(dict(time=s['time'],own=s['self'],enemy=e if e and e.get('visible') else None))
            if tower is None:
                candidate=s['towers'][0 if other==2 else 2]
                if candidate.get('x') is not None:tower=candidate
    ot=[r['time'] for r in obs]
    def enemy_at(t):
        r=at(obs,ot,t)
        return r['enemy'] if r and abs(r['time']-t)<=.41 else None
    def frac(h):
        e=enemy_at(h['time'])
        return h['health']/e['health_max'] if e and e.get('health_max',0)>0 else None
    hits=[e for e in events if e['kind']=='hurt' and e.get('team')==team and e.get('target_team')==other
          and e.get('target')=='npc_dota_hero_nevermore' and e.get('damage',0)>0]
    groups=[]
    for h in hits:
        if not groups or h['time']-groups[-1][-1]['time']>2.000001:groups.append([])
        groups[-1].append(h)
    episodes=[];spawn=heroes[other][0];goal=None
    if tower:goal=dict(x=tower['x']+(-220 if other==2 else 220),y=tower['y']+(-220 if other==2 else 220))
    for index,group in enumerate(groups):
        start=group[0]['time'];end=group[-1]['time'];stop=min(end+2,ht[other][-1]);h0=at(heroes[other],ht[other],start-.00001);h1=at(heroes[other],ht[other],stop)
        window=[h0]+[h for h in heroes[other] if h0['time']<h['time']<=stop]
        own_start=at(heroes[team],ht[team],start-.00001)
        own_window=[own_start]+[h for h in heroes[team] if own_start['time']<h['time']<=stop]
        casts=[e for e in events if e['kind']=='cast' and e.get('team')==other and start<=e['time']<=stop]
        after=[e for e in events if e['kind']=='cast' and e.get('team')==other and stop<e['time']<=end+10]
        gain=collections.Counter();known=0;potential=0.;visible_dt=0.;flask_dt=0.;tango_dt=0.
        for a,b in zip(window,window[1:]):
            ea=enemy_at(a['time']);eb=enemy_at(b['time'])
            if not ea or not eb or b['time']-a['time']>.3:continue
            known+=1
            growth=max(0,eb['health_max']-ea['health_max'])
            amount=max(0,b['health']-a['health']-growth)
            tag='flask_active' if ea.get('flask') else 'tango_active' if ea.get('tango') else 'no_consumable_buff'
            gain[tag]+=amount
        for r,next_r in zip(obs,obs[1:]):
            dt=max(0,min(stop,next_r['time'])-max(start,r['time']));e=r['enemy']
            if not dt or not e:continue
            visible_dt+=dt;potential+=e.get('health_regen',0)*dt
            flask_dt+=dt*bool(e.get('flask'));tango_dt+=dt*bool(e.get('tango'))
        retreat=None
        if goal:
            for h in window:
                f=frac(h)
                if f is None or f>=.56:continue
                later=at(heroes[other],ht[other],h['time']+2)
                me0=at(heroes[team],ht[team],h['time']);me1=at(heroes[team],ht[team],later['time'])
                if later['time']-h['time']>=1.8 and dist(h,goal)-dist(later,goal)>=100 and dist(later,me1)-dist(h,me0)>=75:
                    retreat=dict(time=h['time'],hp=h['health'],hp_fraction=f,x=h['x'],y=h['y'],
                                 toward_t1_goal=True,distance_to_goal=dist(h,goal),distance_after2s=dist(later,goal));break
        death=next((e for e in events if e['kind']=='death' and start<=e['time']<=stop),None)
        ending=('v9_death' if death['team']==other else 'chio_death') if death else 'hit_gap_over2s'
        own_loss=sum(max(0,a['health']-b['health']) for a,b in zip(own_window,own_window[1:]))
        hero_in=sum(e.get('damage',0) for e in events if e['kind']=='hurt' and e.get('team')==other and e.get('target_team')==team
                    and e.get('target')=='npc_dota_hero_nevermore' and start<=e['time']<=stop)
        episodes.append(dict(index=index,start=start,last_hit=end,measurement_end=stop,hits=len(group),damage=sum(e['damage'] for e in group),
            damage_physical=sum(e['damage'] for e in group if not e.get('ability')),damage_spells=sum(e['damage'] for e in group if e.get('ability')),
            v9_hp_start=h0['health'],v9_hp_end=h1['health'],v9_health_gain_lower_bound=sum(gain.values()),health_gain_by_buff_state=dict(gain),
            healing_coverage=dict(known_intervals=known,all_intervals=max(0,len(window)-1)),
            v9_casts=dict(collections.Counter(e['ability'] for e in casts)),v9_casts_next8s=dict(collections.Counter(e['ability'] for e in after)),
            v9_cast_events=[dict(time=e['time'],ability=e['ability']) for e in casts+after],
            flask_active_seconds=flask_dt,tango_active_seconds=tango_dt,visible_seconds=visible_dt,
            regeneration_potential_not_actual_healing=potential,
            flask_healing_exact=None,tango_healing_exact=None,passive_healing_exact=None,fountain_healing_exact=None,
            minimum_distance_to_spawn=min(dist(h,spawn) for h in window),retreat_geometry_proxy=retreat,
            own_hp_loss_lower_bound=own_loss,own_hero_damage_reported=hero_in,
            own_nonhero_damage_lower_bound=max(0,own_loss-hero_in),own_creep_damage_exact=None,own_tower_damage_exact=None,
            end_positions=dict(v9=dict(x=h1['x'],y=h1['y']),chio=dict(x=own_window[-1]['x'],y=own_window[-1]['y'])),
            ending=ending,death_event=death,source_hit_times=[h['time'] for h in group]))
    # A crossing below30% is directly observable only when Bot API supplies max HP.
    crossings=[]
    for a,b in zip(heroes[other],heroes[other][1:]):
        fa,fb=frac(a),frac(b)
        if fa is not None and fb is not None and fa>=.30 and fb<.30:
            later=at(heroes[other],ht[other],b['time']+5)
            crossings.append(dict(time=b['time'],fraction=fb,distance_to_t1_goal=dist(b,goal) if goal else None,
                progress_to_goal5s=dist(b,goal)-dist(later,goal) if goal else None,
                position=dict(x=b['x'],y=b['y']),position_after5s=dict(x=later['x'],y=later['y'])))
    full_gain=0;full_known=0
    for a,b in zip(heroes[other],heroes[other][1:]):
        ea,eb=enemy_at(a['time']),enemy_at(b['time'])
        if ea and eb and b['time']-a['time']<=.3:
            full_known+=1;full_gain+=max(0,b['health']-a['health']-max(0,eb['health_max']-ea['health_max']))
    flask_casts=[e for e in events if e['kind']=='cast' and e.get('team')==other and e.get('ability')=='item_flask']
    return dict(seed=game['seed'],model_team=team,run=str(folder),episodes=episodes,low_hp_crossings=crossings,t1_retreat_goal=goal,
        full_match_v9_health_gain_lower_bound=full_gain,full_match_healing_coverage=dict(known=full_known,total=max(0,len(heroes[other])-1)),
        v9_flask_casts_in_pressure=sum(any(ep['start']<=e['time']<=ep['measurement_end'] for ep in episodes) for e in flask_casts),
        total_hero_damage=sum(e['damage'] for e in hits),v9_flask_casts=len(flask_casts),
        v9_tango_casts=sum(e['kind']=='cast' and e.get('team')==other and e.get('ability')=='item_tango' for e in events))


def main():
    low_priority();series=json.loads((RUN/'series.json').read_text())['ppo-final'];assert len(series['games'])==8
    rows=[analyze(g) for g in series['games']];eps=[e for g in rows for e in g['episodes']]
    aggregate=dict(matches=8,episodes=len(eps),sustained_3plus_hits=sum(e['hits']>=3 for e in eps),hits=sum(e['hits'] for e in eps),
        game_seconds=sum(max(v['time'] for v in g['sides'].values()) for g in series['games']),
        damage=sum(e['damage'] for e in eps),physical=sum(e['damage_physical'] for e in eps),spells=sum(e['damage_spells'] for e in eps),
        health_gain_lower_bound=sum(e['v9_health_gain_lower_bound'] for e in eps),own_nonhero_damage_lower_bound=sum(e['own_nonhero_damage_lower_bound'] for e in eps),
        retreat_geometry_episodes=sum(e['retreat_geometry_proxy'] is not None for e in eps),
        v9_flask_casts=sum(g['v9_flask_casts'] for g in rows),v9_tango_casts=sum(g['v9_tango_casts'] for g in rows),
        full_match_health_gain_lower_bound=sum(g['full_match_v9_health_gain_lower_bound'] for g in rows),
        flask_casts_in_pressure=sum(g['v9_flask_casts_in_pressure'] for g in rows),
        fountain_min_distance=min(e['minimum_distance_to_spawn'] for e in eps),endings=dict(collections.Counter(e['ending'] for e in eps)))
    raw=json.loads(Path('C:/AMI/runs/p16/eval-ppo-120m-v9-sampling/report.json').read_text())
    sim=[]
    for g in raw['per_match']:
        me=next(s for s in g['sides'] if s['controller']=='CHiO');v9=next(s for s in g['sides'] if s['controller']!='CHiO')
        sim.append(dict(game=g['game'],seconds=g['game_seconds'],damage_chio_to_v9=v9['damage_from_hero'],
            chio_damage_from_creeps=me['damage_from_creeps'],chio_damage_from_tower=me['damage_from_tower'],
            v9_flask=v9['consumable_uses']['flask'],v9_tango=v9['consumable_uses']['tango'],v9_deaths=v9['deaths']))
    sa={k:sum(r[k] for r in sim) for k in ['damage_chio_to_v9','chio_damage_from_creeps','chio_damage_from_tower','v9_flask','v9_tango','v9_deaths']}
    sa['game_seconds']=sum(r['seconds'] for r in sim)
    out=dict(definition='Consecutive CHiO damage events on v9 separated by <=2s; singleton episodes included. Measure HP through last hit+2s.',
        dota=rows,dota_aggregate=aggregate,simulator_matches=sim,simulator_aggregate=sa,
        missing_dota=['Incoming entity_hurt from creeps/towers was not recorded; exact split unavailable.',
          'No entity_healed events or fountain modifier. HP gains below exclude observed max-HP growth; not total healing or exact per-source amounts.',
          'Regen rate*time is potential healing only; HP cap, damage timing and missing enemy visibility prevent an exact integral.',
          'Retreat geometry is a movement proxy, not a recorded retreat flag/order destination.'],
        missing_simulator=['Per-hit timestamps/attacker/target/damage timeline','Healing amounts and timestamps by source',
          'HP-max and position timeline; minute snapshots lack positions/max HP','Flask/Tango buff intervals and retreat state/order timestamps',
          'Incoming creep/tower damage timestamps needed to restrict totals to pressure episodes'],
        retreat_policy_source=dict(dota='bots/scripted_teacher_v9.lua: fraction<0.30 enter; >0.56 and threats==0 exit',
            simulator='C:/AMI-wt/P16/gpu/src/scripted_teacher.rs: same 0.30/0.56 hysteresis'))
    dest=ROOT/'reports/P17/pressure-v1.json';dest.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(dota=aggregate,simulator=sa),ensure_ascii=False))


if __name__=='__main__':main()
