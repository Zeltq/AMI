"""Native first-wave acquisition and attack-order aggro; validation-only records."""
import argparse,json,shutil,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from sf_duel.live.game import Game

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--worker',type=int,default=19);p.add_argument('--stop-time',type=float,default=180);p.add_argument('--close',action='store_true');p.add_argument('--pulses',default='20,20.5,21,23.1,26.2');p.add_argument('--pull',action='store_true');a=p.parse_args()
 pulses=[float(t)for t in a.pulses.split(',')if t.strip()]
 if pulses!=sorted(set(pulses))or any(not 0<=t<a.stop_time for t in pulses):p.error('pulses must be unique increasing times before stop-time')
 run=Path(a.run).resolve();run.mkdir(parents=True,exist_ok=False);bots=run/'bots';bots.mkdir()
 for source,name in[(ROOT/'bots/hero_selection.lua','hero_selection.lua'),(ROOT/'bots/agent.lua','agent.lua'),(ROOT/'gpu/bots/aggro.lua','bot_generic.lua')]:shutil.copy2(source,bots/name)
 (bots/'aggro_config.lua').write_text('return {close='+str(a.close).lower()+',pull='+str(a.pull).lower()+',pulses={'+','.join(map(str,pulses))+'}}\n')
 (bots/'item_purchase_generic.lua').write_text('function ItemPurchaseThink() end\n')
 (bots/'ability_item_usage_generic.lua').write_text('function AbilityUsageThink() end\nfunction AbilityLevelUpThink() end\nfunction CourierUsageThink() end\nfunction ItemUsageThink() end\nfunction BuybackUsageThink() end\n')
 content=ROOT/f'workers/{a.worker:02d}/game/dota';frames=[];orders=[];offset=0;tail='';complete=False
 with Game(run,'gpu-aggro',4,30,bot_root=bots,content_root=content,game=content.parents[1],game_port=27900+a.worker,engine_threads=2)as game:
  deadline=time.monotonic()+180
  while time.monotonic()<deadline:
   if game.process.poll()is not None:raise RuntimeError('Dota exited during aggro capture')
   with game.log.open(errors='replace')as f:f.seek(offset);text=tail+f.read();offset=f.tell()
   lines=text.split('\n');tail=lines.pop()
   for line in lines:
    if 'SF_GPU_ERROR|'in line:raise RuntimeError(line)
    for tag,target in[('AGGRO',frames),('AGGRO_ORDER',orders)]:
     marker=f'SF_GPU_{tag}|'
     if marker in line:target.append(json.loads(line.split(marker,1)[1]))
   latest={side:max((f['time']for f in frames if f['team']==side),default=-1000)for side in(2,3)}
   temp=run/'progress.tmp';temp.write_text(json.dumps({'pid':game.pid,'latest_time_by_team':latest,'frames':len(frames),'orders':orders,'training_eligible':False},indent=2),encoding='utf-8');temp.replace(run/'progress.json')
   if min(latest.values())>a.stop_time:complete=True;break
   time.sleep(.1)
  if not complete:raise TimeoutError('Incomplete aggro capture')
 pulse_passed=len(orders)==len(pulses) and all(o['accepted']for o in orders)
 report={'complete':True,'training_eligible':False,'timescale':4,'pulse_capture_passed':pulse_passed,'idle_capture':not any(o['accepted']for o in orders),'close_heroes':a.close,'configured_pulses':pulses,'pull':a.pull,'frames':frames,'orders':orders,'scope':'Observed first-wave targets and short Radiant hero-attack orders; individual captures do not prove cooldown/radius alone.'}
 (run/'aggro.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps({'frames':len(frames),'orders':orders}),flush=True)
 if not pulse_passed:raise RuntimeError('Missing native hero-attack pulses; idle data retained, pulse inference invalid')
if __name__=='__main__':main()
