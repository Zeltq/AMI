"""Bounded dual-stream experiment; never changes the production teacher."""
import argparse,json,subprocess,sys,time,uuid,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from prepare_teacher import prepare
from sf_duel.live.install import isolated_content
from sf_duel.live.game import Game
from sf_duel.live.bridge import LogPackets
parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);parser.add_argument('--until',type=float,default=180);parser.add_argument('--frames',type=int,default=2)
a=parser.parse_args();run=Path(a.run);run.mkdir()
source=run/'bots';shutil.copytree(prepare(),source)
p=source/'teacher_controller.lua';s=p.read_text()
s=s.replace('local M={}','local clock=(os and os.clock) or RealTime or Time\nlocal timing={}\nlocal function mark(name,start)\n    timing[name]=(timing[name] or 0)+(clock()-start)\nend\nlocal M={}')
s=s.replace('    monitor.tick(true)','    local t0=clock()\n    monitor.tick(true)')
s=s.replace('    s.seq=sequence','    mark("observation",t0)\n    local t1=clock()\n    s.seq=sequence\n    s.probe.native_game_time=GameTime()\n    s.probe.native_action_type=bot:GetCurrentActionType()')
s=s.replace('    collecting=false\n    local command','    mark("teacher",t1)\n    local t2=clock()\n    collecting=false\n    local command')
s=s.replace('    if settings.collect_teacher then emit(s) end','    s.probe.native_after={health=bot:GetHealth(),mana=bot:GetMana(),action_type=bot:GetCurrentActionType(),x=bot:GetLocation().x,y=bot:GetLocation().y}\n    mark("execution_shop",t2)\n    local t3=clock()\n    if settings.collect_teacher then emit(s) end\n    mark("encode_print",t3)\n    if sequence%300==0 then\n        print("SF_PROFILE|"..json.encode({team=s.team,time=s.time,seq=sequence,\n            clock=(os and os.clock) and "os.clock" or "RealTime/Time",stages=timing}))\n    end')
p.write_text(s)
content,bots=isolated_content(ROOT/'workers/00',source)
with (run/'native-report.json').open('w') as report:
 receiver=subprocess.Popen([str(ROOT/'tools/nativecollect/nativecollect.exe'),'--run',str(run),'--radiant','12250','--dire','12251'],stdout=report,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
 try:
  with Game(run,uuid.uuid4().hex,50,15,bot_root=bots,collect_teacher=True,content_root=content,game=content.parents[1],worldstate_ports=(12250,12251),engine_threads=2,worldstate_frames=a.frames) as game:
   game.command('host_timescale');game.command('echo SF_CAPTURE_READY')
   packets=LogPackets(game.log,start_marker='SF_CAPTURE_READY');deadline=time.monotonic()+240
   while time.monotonic()<deadline:
    frames=packets.read()
    if packets.errors:raise RuntimeError(packets.errors)
    if packets.events or (frames and frames[-1]['time']>=a.until):break
    time.sleep(.05)
   else:raise TimeoutError('probe wall limit')
  receiver.wait(timeout=40)
 finally:
  if receiver.poll() is None:receiver.kill();receiver.wait()
print(json.dumps({'run':str(run),'native_exit':receiver.returncode}))
