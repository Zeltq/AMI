"""Controlled stock-Dota aggro using server VScript; diagnostics only."""
import argparse,json,shutil,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from sf_duel.live.game import Game

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--worker',type=int,default=19);p.add_argument('--timescale',type=float,default=4);p.add_argument('--suite',choices=['radius','pull','priority','economy','attack-range','requiem','attack-once','fountain'],default='radius');p.add_argument('--kind',choices=['melee','ranged','siege'],default='melee');p.add_argument('--first-source',choices=['hero','tower'],default='hero');p.add_argument('--victim-gold',type=int,default=95);p.add_argument('--actor-mode',choices=['server','bot'],default='server');p.add_argument('--probe-magic',action='store_true');p.add_argument('--actor-team',type=int,choices=[2,3],default=2);p.add_argument('--validation-cases',action='store_true');p.add_argument('--natural-wave',action='store_true');a=p.parse_args()
    if not 1<=a.timescale<=4:p.error('timescale must be1..4')
    if a.actor_mode=='bot'and a.suite not in('radius','pull','priority','attack-once'):p.error('bot actor mode is for radius/pull/priority suites')
    if a.natural_wave and a.suite!='priority':p.error('natural-wave requires priority')
    run=Path(a.run).resolve();run.mkdir(parents=True,exist_ok=False);bots=run/'bots';bots.mkdir()
    if not 0<=a.victim_gold<=100000:p.error('victim-gold must be0..100000')
    source={'economy':'economy_server','attack-range':'attack_range_server','requiem':'requiem_server','attack-once':'attack_once_server','fountain':'fountain_server'}.get(a.suite,'aggro_server')
    shutil.copy2(ROOT/'bots/hero_selection.lua',bots/'hero_selection.lua');shutil.copy2(ROOT/f'gpu/bots/{source}.lua',bots/'gpu_server.lua')
    (bots/'gpu_server_config.lua').write_text('return {suite="'+a.suite+'",kind="'+a.kind+'",first_source="'+a.first_source+'",victim_gold='+str(a.victim_gold)+',actor_mode="'+a.actor_mode+'",actor_team='+str(a.actor_team)+',probe_magic='+str(a.probe_magic).lower()+',validation_cases='+str(a.validation_cases).lower()+',natural_wave='+str(a.natural_wave).lower()+'}\n')
    (bots/'gpu_actor_actions.lua').write_text('return {enabled=false}\n')
    shutil.copy2(ROOT/'gpu/bots/aggro_observer.lua',bots/'bot_generic.lua');(bots/'item_purchase_generic.lua').write_text('function ItemPurchaseThink() end\n')
    shutil.copy2(ROOT/'gpu/bots/aggro_callbacks.lua',bots/'ability_item_usage_generic.lua')
    content=ROOT/f'workers/{a.worker:02d}/game/dota';frames=[];orders=[];cases=[];bot_frames=[];bot_orders=[];callbacks=[];done=False
    with Game(run,'gpu-server-scenario',a.timescale,30,bot_root=bots,content_root=content,game=content.parents[1],game_port=27900+a.worker,engine_threads=2)as game:
        game.command('script_reload_code bots/gpu_server')
        deadline=time.monotonic()+(180 if a.natural_wave and a.kind=='siege'else 65);offset=0;tail=''
        while time.monotonic()<deadline:
            if game.process.poll()is not None:raise RuntimeError('Dota exited; inspect dota.log')
            with game.log.open(errors='replace')as f:f.seek(offset);text=tail+f.read();offset=f.tell()
            lines=text.split('\n');tail=lines.pop()
            for line in lines:
                if 'SF_GPU_SERVER_ERROR|'in line or 'SF_GPU_BOT_ERROR|'in line or 'Script Runtime Error:'in line:raise RuntimeError(line)
                if 'SF_GPU_SERVER_AGGRO|'in line:frames.append(json.loads(line.split('SF_GPU_SERVER_AGGRO|',1)[1]))
                if 'SF_GPU_SERVER_ORDER|'in line:orders.append(json.loads(line.split('SF_GPU_SERVER_ORDER|',1)[1]))
                if 'SF_GPU_SERVER_CASE|'in line:
                    c=json.loads(line.split('SF_GPU_SERVER_CASE|',1)[1]);cases.append(c)
                    if a.actor_mode=='bot':
                        text='return {enabled=true,case='+str(c['case'])+',start='+str(c['time'])+',pull='+str(c.get('pull',False)).lower()+',hold_attack='+str(c.get('hold_attack',False)).lower()+',pulses={'+','.join(map(str,c['pulses']))+'}}\n'
                        temporary=bots/'gpu_actor_actions.tmp';temporary.write_text(text);temporary.replace(bots/'gpu_actor_actions.lua')
                if 'SF_GPU_SERVER_OBSERVER|'in line:bot_frames.append(json.loads(line.split('SF_GPU_SERVER_OBSERVER|',1)[1]))
                if 'SF_GPU_BOT_ORDER|'in line:bot_orders.append(json.loads(line.split('SF_GPU_BOT_ORDER|',1)[1]))
                if 'SF_GPU_BOT_CALLBACK|'in line:callbacks.append(json.loads(line.split('SF_GPU_BOT_CALLBACK|',1)[1]))
                if 'SF_GPU_SERVER_DONE|'in line:done=True
            if done:break
            time.sleep(.05)
    expected=sum(len(c['pulses'])for c in cases)if a.actor_mode=='bot'else 0
    delays=[o['elapsed']-next(c for c in cases if c['case']==o['case'])['pulses'][o['pulse']-1]for o in bot_orders]
    quality={'server_complete':done,'paired_bot_frames':len(bot_frames),'expected_bot_orders':expected,'actual_bot_orders':len(bot_orders),'rejected_bot_orders':sum(not o['accepted']for o in bot_orders),'maximum_order_lateness_seconds':max(delays,default=0)}
    quality['priority_cases_without_acquisition']=[c['case']for c in cases if a.suite=='priority'and not any(f['case']==c['case']and f['units'][2].get('aggro_target')for f in frames)]
    quality['passed']=done and bool(frames)and bool(bot_frames)and not quality['priority_cases_without_acquisition']and(a.actor_mode!='bot'or(len(bot_orders)==expected and quality['rejected_bot_orders']==0 and all(-.001<=v<=1/30+.001 for v in delays)))
    report={'complete':quality['passed'],'quality':quality,'training_eligible':False,'timescale':a.timescale,'suite':a.suite,'kind':a.kind,'parameters':{'first_source':a.first_source,'victim_gold':a.victim_gold,'actor_mode':a.actor_mode,'probe_magic':a.probe_magic,'actor_team':a.actor_team,'validation_cases':a.validation_cases,'natural_wave':a.natural_wave},'cases':cases,'frames':frames,'orders':orders,'bot_frames':bot_frames,'bot_orders':bot_orders,'callbacks':callbacks,'scope':'Controlled stock-engine units with paired BotAPI observations. Economy suite alone forces one respawn and bypasses invulnerability to inspect gold transactions; it is not a natural respawn scenario. Pull cases add a128-unit vision source around the stationary enemy hero to isolate order/cooldown from hidden-target rejection.'}
    (run/'server-scenarios.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps({'complete':done,'frames':len(frames),'orders':orders}))
    if not done:raise RuntimeError('Server scenario incomplete; inspect VScript errors in dota.log')
    if not quality['passed']:raise RuntimeError('Native capture quality failed: '+json.dumps(quality))

if __name__=='__main__':main()
