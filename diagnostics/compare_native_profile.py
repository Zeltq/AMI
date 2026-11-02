import collections,gzip,json,struct,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from sf_duel.protos.worldstate_pb2 import CMsgBotWorldState
from sf_duel.live.bridge import LogPackets
run=Path(sys.argv[1]);states={};counts={}
for team in (2,3):
 states[team]={};count=0
 with gzip.open(run/f'world-{team}.pb.gz','rb') as f:
  while header:=f.read(4):
   n=struct.unpack('<I',header)[0];s=CMsgBotWorldState.FromString(f.read(n))
   states[team][round(s.game_time*30)]=s;count+=1
 counts[team]=count
packets=LogPackets(run/'dota.log',start_marker='SF_CAPTURE_READY');frames=packets.read()
terminal=min((e['time'] for e in packets.events),default=float('inf'))
frames=[s for s in frames if s['time']<terminal]
fields={'health':'health','health_max':'health_max','mana':'mana','mana_max':'mana_max','facing':'facing','health_regen':'health_regen','mana_regen':'mana_regen','attack_damage':'attack_damage','attack_range':'attack_range','attack_point':'attack_anim_point','projectile_speed':'attack_projectile_speed','armor':'armor','magic_resist':'magic_resist','level':'level','move_speed':'current_movement_speed','attack_cycle':'anim_cycle','last_hits':'last_hits','denies':'denies','xp_to_next':'xp_needed_to_level'}
visibility=collections.Counter()
deltas={k:collections.Counter() for k in ("health","mana")};large=[];after_checks=collections.defaultdict(collections.Counter);gaps=[];last={}
checks=collections.defaultdict(collections.Counter);examples=[];offset_checks={offset:collections.defaultdict(collections.Counter) for offset in (-2,-1,0,1,2)};matched=0;missing=0;actions=collections.Counter();profile={}
for s in frames:
 old=last.get(s['team'])
 if old and s['demonstration']['decision_tick']!=old['demonstration']['decision_tick']+2:
  gaps.append({'team':s['team'],'before':old['time'],'after':s['time']})
 last[s['team']]=s
 actions[s['demonstration']['command']['action']]+=1
 native=states[s['team']].get(round(s['probe']['native_game_time']*30))
 if native is None:missing+=1;continue
 u=next((u for u in native.units if u.name=='npc_dota_hero_nevermore' and u.team_id==s['team']),None)
 if u is None:missing+=1;continue
 matched+=1
 native_enemy=any(v.name=='npc_dota_hero_nevermore' and v.team_id!=s['team'] and v.is_alive for v in native.units)
 lua_enemy=bool(s.get('enemy'))
 visibility[f'lua_{lua_enemy}_native_{native_enemy}']+=1
 for offset,stats in offset_checks.items():
  state=states[s['team']].get(round(s['probe']['native_game_time']*30)+offset)
  if state is None:continue
  hero=next((v for v in state.units if v.name=='npc_dota_hero_nevermore' and v.team_id==s['team']),None)
  if hero is None:continue
  compare={k:(s['self'].get(k),getattr(hero,v) if hero.HasField(v) else None) for k,v in fields.items()}
  compare.update({k:(s['self'].get(k),getattr(hero.location,k)) for k in ('x','y','z')})
  compare['action_type']=(s['probe']['native_action_type'],hero.action_type)
  for k,(a,b) in compare.items():
   stats[k]['missing' if a is None or b is None else 'equal' if abs(a-b)<.0001 else 'different']+=1
 pairs={k:(s['self'].get(k),getattr(u,v) if u.HasField(v) else None) for k,v in fields.items()}
 pairs.update({k:(s['self'].get(k),getattr(u.location,k)) for k in ('x','y','z')})
 pairs['action_type']=(s['probe']['native_action_type'],u.action_type)
 for k,(a,b) in pairs.items():
  result='missing' if a is None or b is None else 'equal' if abs(a-b)<.0001 else 'different'
  checks[k][result]+=1
  if k in deltas and a is not None and b is not None:
   deltas[k][b-a]+=1
   if abs(b-a)>10 and len(large)<30:large.append({'time':s['time'],'team':s['team'],'field':k,'lua_before':a,'native':b,'native_damage_events':[{'damage':e.damage,'ability_id':e.ability_id} for e in native.damage_events if e.victim_unit_handle==u.handle],'command':s['demonstration']['command'],'lua_after':s['probe'].get('native_after')})
  if k in s['probe'].get('native_after',{}) and b is not None:
   after=s['probe']['native_after'][k]
   after_checks[k]['equal' if abs(after-b)<.0001 else 'different']+=1
  if result=='different' and len([e for e in examples if e['field']==k])<3:examples.append({'time':s['time'],'field':k,'lua':a,'native':b})
for line in (run/'dota.log').read_text(errors='replace').splitlines():
 if 'SF_PROFILE|' in line:
  x=json.loads(line.split('SF_PROFILE|',1)[1]);profile[x['team']]=x
report={'native_packets':counts,'enemy_visibility':visibility,'lua_frames':len(frames),'matched':matched,'missing':missing,'checks':checks,'deltas':deltas,'large_differences':large,'after_checks':after_checks,'offset_checks':offset_checks,'examples':examples,'profile':profile,'actions':actions,'errors':packets.errors,'decision_gaps':gaps}
(run/'comparison.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
