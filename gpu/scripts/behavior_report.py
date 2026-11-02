"""Read-only behavior audit of both policies from admitted native recordings."""
import argparse,json,math,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from sf_duel.native_data import records

class Motion:
 def __init__(self):
  self.previous=None;self.order=None;self.run=0.;self.maximum=0.;self.stuck=0.;self.first_mid=None;self.frames=0;self.gaps=0
  self.passive_run=0.;self.passive_max=0.;self.passive_seconds=0.
  self.idle_run=0.;self.idle_max=0.;self.idle_seconds=0.
 def sample(self,time,x,y,alive,blocked=False,moving=None,idle=False):
  if alive and self.first_mid is None and x*x+y*y<2000**2 and abs(x-y)<1200:self.first_mid=time
  if self.previous is not None:
   t,px,py=self.previous;dt=time-t
   continuous=0<dt<=.5
   if not continuous:self.gaps+=1
   eligible=continuous and alive and not blocked and moving is not False and self.order is not None and math.dist((x,y),self.order)>32
   if eligible and math.hypot(x-px,y-py)<1:
    old=self.run;self.run+=dt;self.maximum=max(self.maximum,self.run)
    if self.run>=10:self.stuck+=self.run if old<10 else dt
   else:self.run=0.
   if continuous and alive and not blocked and self.first_mid is None and time>=0 and math.hypot(x-px,y-py)<1:
    old=self.passive_run;self.passive_run+=min(dt,time);self.passive_max=max(self.passive_max,self.passive_run)
    if self.passive_run>=10:self.passive_seconds+=self.passive_run if old<10 else min(dt,time)
   else:self.passive_run=0.
   if continuous and alive and not blocked and idle and self.first_mid is not None and math.hypot(x-px,y-py)<1:
    old=self.idle_run;self.idle_run+=dt;self.idle_max=max(self.idle_max,self.idle_run)
    if self.idle_run>=10:self.idle_seconds+=self.idle_run if old<10 else dt
   else:self.idle_run=0.
  self.previous=(time,x,y);self.frames+=1
 def result(self):return dict(first_mid_time=self.first_mid,max_stuck_seconds=self.maximum,stuck_seconds=self.stuck,max_stationary_before_mid_seconds=self.passive_max,stationary_before_mid_seconds=self.passive_seconds,max_idle_after_mid_seconds=self.idle_max,idle_after_mid_seconds=self.idle_seconds,observation_frames=self.frames,observation_gaps=self.gaps)

def game_behavior(match):
 motions={s:Motion()for s in (2,3)};legacy={s:Motion()for s in (2,3)};commands={};deaths={2:0,3:0};outcome=match.get('outcome')or{}
 for row in records(Path(match['run'])/'native-decisions.pb.gz'):
  if row.HasField('decision'):commands[row.decision.team,row.decision.sequence]=row.decision
  if row.HasField('execution'):
   e=row.execution;d=commands.pop((e.team,e.sequence),None)
   if e.accepted and d is not None and d.action:
    motions[e.team].order=(d.x,d.y)if d.action in (1,2)else None
    legacy[e.team].order=motions[e.team].order
  if not row.HasField('decision')or not row.HasField('observation'):continue
  s=row.decision.team;o=row.observation
  for player in o.players:
   if player.hero_id==11 and player.team_id in deaths:deaths[player.team_id]=max(deaths[player.team_id],player.deaths)
  own=next((u for u in o.units if u.name=='npc_dota_hero_nevermore'and u.team_id==s),None)
  if own is None:continue
  extra=next((u for u in row.model_supplement.units if u.handle in (0,own.handle)),None)
  blocked=own.is_using_ability or own.is_channeling or any(a.is_in_ability_phase for a in own.abilities)or any(m.name=='modifier_nevermore_requiem_fear'for m in own.modifiers)
  if extra is not None:blocked=blocked or extra.casting or extra.channeling or extra.using_ability
  action=own.action_type if own.HasField('action_type')else None
  motions[s].sample(o.dota_time,own.location.x,own.location.y,own.is_alive and own.health>0,blocked,moving=action in(2,3)if action is not None else None,idle=action==1)
  legacy[s].sample(o.dota_time,own.location.x,own.location.y,own.is_alive and own.health>0,blocked)
 terminal_deaths={s:int(outcome.get('reason')=='death'and outcome.get('winner')==5-s)for s in (2,3)}
 return {'index':match['index'],'model_team':match['model_team'],'time':match['time'],'outcome':outcome,
  'sides':{str(s):{**match['sides'][str(s)],**motions[s].result(),'legacy_stuck_seconds':legacy[s].stuck,'legacy_max_stuck_seconds':legacy[s].maximum,'deaths_observed':deaths[s],
   'deaths_minimum':max(deaths[s],terminal_deaths[s])}for s in (2,3)}}

def summarize(games):
 result={}
 for role in ('learner','baseline'):
  rows=[g['sides'][str(g['model_team']if role=='learner'else 5-g['model_team'])]for g in games]
  times=[r['first_mid_time']for r in rows if r['first_mid_time']is not None]
  result[role]={'matches':len(rows),'reached_mid':len(times),'mean_first_mid_time':sum(times)/len(times)if times else None,
   'terminal_death_losses':sum(g['outcome'].get('reason')=='death'and g['outcome'].get('winner')==(5-g['model_team']if role=='learner'else g['model_team'])for g in games),
   'stuck_matches':sum(r['stuck_seconds']>0 for r in rows),'max_stuck_seconds':max((r['max_stuck_seconds']for r in rows),default=0),
   'stationary_before_mid_matches':sum(r['stationary_before_mid_seconds']>0 for r in rows),'max_stationary_before_mid_seconds':max((r['max_stationary_before_mid_seconds']for r in rows),default=0),
   **{'mean_'+k:sum(r.get(k,0)for r in rows)/max(1,len(rows))for k in ('last_hits','denies','level','deaths_observed','deaths_minimum','stuck_seconds','stationary_before_mid_seconds','idle_after_mid_seconds','legacy_stuck_seconds')}}
 return result

def main():
 p=argparse.ArgumentParser();p.add_argument('--evaluation',required=True);p.add_argument('--output',required=True);a=p.parse_args()
 source=Path(a.evaluation);evaluation=json.loads(source.read_text());games=[]
 for match in evaluation['games']:
  games.append(game_behavior(match));print(f"Audited {len(games)}/{len(evaluation['games'])}",flush=True)
 report={'source':str(source.resolve()),'games':games,'summary':summarize(games),'by_learner_side':{str(s):summarize([g for g in games if g['model_team']==s])for s in (2,3)},'read_only':True,
  'definitions':{'mid':'First alive observation within2000 of origin and abs(x-y)<1200; negative pregame times allowed.',
  'stuck':'Last accepted movement goal and current native action_type MOVE/MOVE_DIRECT; destination>32units, displacement<1unit per observation continuously>=10s. If action_type is unavailable the last-order fallback is retained. Gaps, death, casting/channeling/fear reset the run. Legacy last-order-only counts remain in legacy_stuck_seconds.',
  'idle_after_mid':'Actual native action_type IDLE after reaching mid, alive and stationary>=10s outside casting/channeling/fear. This preserves long standing intervals after a completed movement; it does not establish why the policy waits.',
  'stationary_before_mid':'Alive, not casting/channeling/feared, displacement<1unit per observation continuously>=10s after DotaTime0 before ever reaching mid, regardless of order. Reported separately so repeated CONTINUE at spawn cannot be mistaken for successful navigation merely because there is no movement order.',
  'deaths_observed':'Maximum public SF scoreboard deaths observed in either side recording; a final death after the final observation may be censored. deaths_minimum also includes the terminal loser in a death outcome; this remains a lower bound, not an invented final scoreboard.'}}
 Path(a.output).write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
if __name__=='__main__':main()
