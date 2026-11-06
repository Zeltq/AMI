"""Two-team headless navigation/handoff diagnostic; CPU fixture, not a model match."""
import argparse
import gzip
import json
import msvcrt
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from play_vs_chio import below_normal,save
from p17_matches import bundle
from p17_p16_guard import check
from sf_duel.live.game import Game


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();below_normal();check(force=True)
    import torch
    from sf_duel.chio_model import Policy
    from p15_model_fixture import export_weights
    torch.set_num_threads(1);torch.manual_seed(1706)
    a.run.mkdir(parents=True,exist_ok=False);bots=a.run/'bots';bundle(bots,2,navigation_probe=True)
    weights=a.run/'diagnostic-weights';export_weights(Policy().cpu().eval(),weights)
    (bots/'p17_navigation_exit.lua').write_text("for _,h in pairs(Entities:FindAllByClassname('npc_dota_hero_nevermore')) do h:Stop();h:SetAbsOrigin(GetGroundPosition(Vector(h:GetTeamNumber()==2 and -6000 or 6000,h:GetTeamNumber()==2 and -6000 or 6000,0),h)) end\nprint('SF_CHIO_NAV_EXIT|true')\n")
    children=[];logs=[];handoffs={};acks=[];record=dict(stage='starting',pid=__import__('os').getpid(),command=' '.join(sys.argv),background_work_remaining=True)
    def status(**kw):record.update(kw);save(a.run/'status.json',record);save(ROOT/'runs/pilot-status.json',{'P17':record})
    status()
    with Path('C:/AMI/runs/p17/runtime.lock').open('a+b') as lock:
        lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        try:
            for team in (2,3):
                log=(a.run/f'policy-{team}.log').open('wb');logs.append(log)
                args=[sys.executable,str(ROOT/'scripts/play_vs_chio.py'),'--model',str(weights),'--log',str(a.run/'dota.log'),
                      '--bots',str(bots),'--output',str(a.run/f'policy-{team}'),'--team',str(team),'--navigation-probe']
                children.append(subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW|subprocess.BELOW_NORMAL_PRIORITY_CLASS))
            deadline=time.monotonic()+60
            while not all((a.run/f'policy-{t}/ready').exists() for t in (2,3)):
                if any(c.poll() is not None for c in children):raise RuntimeError('CPU probe startup failed')
                if time.monotonic()>deadline:raise TimeoutError('CPU probe readiness')
                time.sleep(.1)
            runtime=Path('C:/AMI/runs/p17/bridge-001/runtime')
            with Game(a.run,'P17-navigation',1,30,bot_root=bots,content_root=runtime/'game/dota',game=runtime,game_port=27927,engine_threads=1,below_normal=True) as game:
                status(stage='navigation',dota_pid=game.pid,policy_pids=[c.pid for c in children])
                game.command('script_reload_code bots/p17_setup')
                offset=0;tail='';both_time=None;exit_time=None;deadline=time.monotonic()+150
                while time.monotonic()<deadline:
                    if any(c.poll() is not None for c in children):raise RuntimeError('CPU probe failed')
                    with game.log.open(errors='replace') as f:f.seek(offset);data=tail+f.read();offset=f.tell()
                    lines=data.split('\n');tail=lines.pop()
                    for line in lines:
                        if any(tag in line for tag in ('SF_CHIO_ERROR|','SF_P17_ERROR|','Script Runtime Error:')):raise RuntimeError(line)
                        if 'SF_CHIO_HANDOFF|' in line:
                            r=json.loads(line.split('SF_CHIO_HANDOFF|',1)[1]);handoffs.setdefault(r['team'],[]).append(r)
                        if 'SF_CHIO_NAV_ACK|' in line:acks.append(json.loads(line.split('SF_CHIO_NAV_ACK|',1)[1]))
                    if len(handoffs)==2 and both_time is None:both_time=time.monotonic()
                    if both_time and not exit_time and time.monotonic()-both_time>2:
                        game.command('script_reload_code bots/p17_navigation_exit');exit_time=time.monotonic()
                    if both_time and time.monotonic()-both_time>=30:break
                    status(stage='navigation',handoffs=handoffs,acks=len(acks));time.sleep(.05)
                else:raise TimeoutError('Both teams did not hand off')
            for team in (2,3):(a.run/f'policy-{team}/stop').touch()
            for c in children:c.wait(timeout=15)
            result=dict(teams={},acks=dict(total=len(acks),on_time=sum(r['accepted'] and r['tick']-r['observed_tick']==7 for r in acks)),training_eligible=False)
            for team in (2,3):
                decisions=[json.loads(x) for x in (a.run/f'policy-{team}/decisions.jsonl').read_text().splitlines()]
                with gzip.open(a.run/f'policy-{team}/observations.jsonl.gz','rt',encoding='utf-8') as f:observations=[json.loads(x) for x in f]
                boundary=[r for r in decisions if r['handoff']];before=[r for r in decisions if not r['navigation_ready']]
                outside=[s for s in observations if s['navigation_ready'] and s['self']['x']**2+s['self']['y']**2>=2000**2]
                periodic=[r for r in decisions if r['decision_index']%128==0]
                result['teams'][str(team)]=dict(handoffs=handoffs.get(team,[]),boundary=boundary,
                    pre_handoff_observations=len(before),pre_handoff_model_executions=sum(r['policy_executed'] for r in before),
                    outside_after_handoff=len(outside),periodic_resets=sum(r['reset'] for r in periodic),periodic_expected=len(periodic),
                    hp_history_known_at_handoff=[r['hp_history_known'] for r in boundary],
                    passed=len(handoffs.get(team,[]))==len(boundary)==1 and boundary[0]['reset'] and boundary[0]['input_previous_action']==1
                        and not any(r['policy_executed'] for r in before) and len(outside)>0 and all(r['reset'] for r in periodic)
                        and len([s for s in observations if s['navigation_handoff']])==1)
            result['passed']=all(r['passed'] for r in result['teams'].values()) and result['acks']['on_time']==len(acks)
            save(a.run/'result.json',result);status(stage='complete',background_work_remaining=False,passed=result['passed'])
            print(json.dumps(result))
            if not result['passed']:raise RuntimeError('Navigation acceptance failed')
        except BaseException as e:status(stage='failed',background_work_remaining=False,error=repr(e));raise
        finally:
            for team in (2,3):
                path=a.run/f'policy-{team}'
                if path.exists():(path/'stop').touch()
            for c in children:
                try:c.wait(timeout=5)
                except subprocess.TimeoutExpired:c.terminate();c.wait(timeout=5)
            for log in logs:log.close()


if __name__=='__main__':main()
