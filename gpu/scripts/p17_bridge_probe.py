"""Isolated no-render bridge probe; tests cheats gate and Bot API sandbox."""
import argparse
import ctypes
import datetime
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from sf_duel.live.game import Game
from sf_duel.live.install import isolated_content
from sf_duel.live.bridge import literal


def save(path,data):
    tmp=path.with_suffix('.pending')
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    tmp.replace(path)


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--seconds',type=int,default=45)
    p.add_argument('--runtime',type=Path,default=Path('C:/AMI/runs/p17/bridge-001/runtime'))
    a=p.parse_args()
    k=ctypes.WinDLL('kernel32',use_last_error=True)
    k.GetCurrentProcess.restype=ctypes.c_void_p
    k.SetPriorityClass.argtypes=[ctypes.c_void_p,ctypes.c_ulong]
    if not k.SetPriorityClass(k.GetCurrentProcess(),0x4000):raise ctypes.WinError(ctypes.get_last_error())
    a.run.mkdir(parents=True,exist_ok=False);bots=a.run/'bots';bots.mkdir()
    for src,dest in [('bots/hero_selection.lua','hero_selection.lua'),('gpu/bots/p17_probe_actor.lua','bot_generic.lua'),('gpu/bots/p17_bridge.lua','p17_bridge.lua')]:shutil.copy2(ROOT/src,bots/dest)
    (bots/'item_purchase_generic.lua').write_text('function ItemPurchaseThink() end\n')
    (bots/'ability_item_usage_generic.lua').write_text('require("bots/bot_generic")\nfunction AbilityUsageThink() Think() end\nfunction AbilityLevelUpThink() end\nfunction CourierUsageThink() end\nfunction ItemUsageThink() end\nfunction BuybackUsageThink() end\n')
    for team in (2,3):(bots/f'chio_commands_{team}.lua').write_text('return {}\n')
    baseline=json.loads(Path('C:/AMI/runs/p16/teacher-1024/status.json').read_text())
    save(a.run/'p16-baseline.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),status=baseline,
        stderr_tail=Path('C:/AMI/runs/p16/jobs/teacher-1024/stderr.log').read_text().splitlines()[-8:]))
    record=dict(pid=os.getpid(),stage='starting',command=' '.join(sys.argv),artifacts=[str(a.run)],background_work_remaining=True)
    def status(**kw):
        record.update(kw);save(a.run/'status.json',record)
        # P17 owns this worktree-local registry; never race P16's shared registry.
        pilot=ROOT/'runs/pilot-status.json';pilot.parent.mkdir(exist_ok=True);save(pilot,{'P17':record})
    status();rows=[];commands=[];offset=0;tail='';phase='cheats0'
    try:
        content=a.runtime/'game/dota'
        if not (a.runtime/'isolation.json').exists():content,_=isolated_content(a.runtime,bots)
        with Game(a.run,'P17-bridge',1,30,bot_root=bots,content_root=content,game=content.parents[1],
                  game_port=27927,engine_threads=1,below_normal=True) as game:
            status(stage='bridge-probe',dota_pid=game.pid)
            game.command('sv_cheats 0');game.command('script_reload_code bots/p17_bridge')
            started=time.monotonic();enabled=False
            while time.monotonic()-started<a.seconds:
                if game.process.poll() is not None:raise RuntimeError('Dota exited')
                if not enabled and time.monotonic()-started>8:
                    game.command('sv_cheats 1');game.command('script_reload_code bots/p17_bridge');enabled=True;phase='cheats1'
                with game.log.open(errors='replace') as f:
                    f.seek(offset);data=tail+f.read();offset=f.tell()
                lines=data.split('\n');tail=lines.pop()
                for line in lines:
                    for tag in ('SF_CHIO_REQUEST','SF_CHIO_ACK','SF_P17_POSITION','SF_P17_BOT_API','SF_P17_BRIDGE_READY'):
                        if tag+'|' not in line:continue
                        row=json.loads(line.split(tag+'|',1)[1]);rows.append(dict(tag=tag,phase=phase,**row))
                        if tag=='SF_CHIO_REQUEST':
                            commands.append(row);commands=commands[-16:]
                            tmp=bots/'chio_commands_2.tmp';tmp.write_text('return '+literal(commands)+'\n');tmp.replace(bots/'chio_commands_2.lua')
                    if 'Script Runtime Error:' in line:raise RuntimeError(line)
                time.sleep(.01)
        acks=[r for r in rows if r['tag']=='SF_CHIO_ACK'];accepted=[r for r in acks if r['accepted']]
        histogram={}
        for r in accepted:
            key=str(r['tick']-r['observed_tick']);histogram[key]=histogram.get(key,0)+1
        result=dict(rows=rows,acks=len(acks),accepted=len(accepted),delay_ticks=histogram,
                    cheats0_ready=sum(r['tag']=='SF_P17_BRIDGE_READY' and r['phase']=='cheats0' for r in rows),
                    cheats1_ready=sum(r['tag']=='SF_P17_BRIDGE_READY' and r['phase']=='cheats1' for r in rows))
        save(a.run/'result.json',result);status(stage='complete',background_work_remaining=False,accepted=len(accepted),acks=len(acks))
    except BaseException as e:
        status(stage='failed',error=str(e),background_work_remaining=False);raise


if __name__=='__main__':main()
