"""Run the native Go teacher experiment; random policy/training are never loaded."""
import argparse,hashlib,json,re,shutil,subprocess,sys,time,uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from prepare_native_teacher import prepare
from compact_artifacts import compact_file
from sf_duel.live.install import isolated_content
from sf_duel.live.game import Game

def requested_recording(run,override,explicit=None):
 if explicit:return explicit
 name=override.get('recordings',{}).get(run.name.split('-attempt-',1)[0])
 if name is None:return None
 if not isinstance(name,str)or not re.fullmatch(r'[A-Za-z0-9_-]{1,100}',name):raise ValueError('Recording name must be a simple ASCII filename')
 return name+'_'+run.name

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--frozen-bots');p.add_argument('--frozen-collector');p.add_argument('--record');p.add_argument('--routing-file');p.add_argument('--allow-late',action='store_true');p.add_argument('--policy-address',default='');p.add_argument('--policy-team',type=int,default=0);p.add_argument('--threaded-worldstate',action='store_true');p.add_argument('--timescale',type=float,default=2);p.add_argument('--until',type=float,default=60);p.add_argument('--worker',type=int,default=0);p.add_argument('--period',type=int,default=4);p.add_argument('--offset',type=int,default=4);p.add_argument('--wall-timeout',type=float,default=240);a=p.parse_args()
 if not 0<=a.worker<20:p.error('Use an existing stable worker slot 0..19')
 run=Path(a.run).resolve();run.mkdir(parents=True,exist_ok=False);source=run/'bots'
 if a.routing_file:shutil.copy2(a.routing_file,run/'policy-routing.json')
 requested_timescale=a.timescale
 runtime=run.parent/'runtime-overrides.json';override={}
 if runtime.exists():
  override=json.loads(runtime.read_text());speed=float(override.get('timescale',a.timescale))
  if not 0<speed<=50:raise ValueError('Runtime timescale outside supported bounds')
  a.timescale=speed
 a.record=requested_recording(run,override,a.record)
 (run/'effective-run-config.json').write_text(json.dumps({'requested_timescale':requested_timescale,'effective_timescale':a.timescale,'period_ticks':a.period,'offset_ticks':a.offset,'policy_team':a.policy_team,'policy_address':a.policy_address,'allow_late':a.allow_late,'record_name':a.record},indent=2))
 if a.frozen_bots:shutil.copytree(a.frozen_bots,source)
 else:prepare(source)
 if a.allow_late:
  shutil.copy2(ROOT/'bots/native_executor.lua',source/'native_executor.lua')
  (run/'executor-override.json').write_text(json.dumps({'reason':'user-authorized late command execution','sha256':hashlib.sha256((source/'native_executor.lua').read_bytes()).hexdigest()}))
 if a.frozen_collector:
  collector=Path(a.frozen_collector)
 else:
  collector=run/'collector';collector.mkdir()
  for original in (ROOT/'tools/nativecollect').glob('*'):
   if original.is_file() and (original.suffix in ('.go','.proto') or original.name in ('go.mod','go.sum','nativecollect.exe')):shutil.copy2(original,collector/original.name)
  for folder in ('duelwire','worldstate'):shutil.copytree(ROOT/'tools/nativecollect'/folder,collector/folder)
  (run/'collector-manifest.json').write_text(json.dumps({str(f.relative_to(collector)):hashlib.sha256(f.read_bytes()).hexdigest() for f in collector.rglob('*') if f.is_file()},indent=2))
 (run/'collector-origin.json').write_text(json.dumps({'path':str(collector),'sha256':hashlib.sha256((collector/'nativecollect.exe').read_bytes()).hexdigest()},indent=2))
 content,_=isolated_content(ROOT/'workers'/f'{a.worker:02d}',source)
 ports=(12250+a.worker*2,12251+a.worker*2)
 with (run/'go.log').open('wb') as log:
  command=[str(collector/'nativecollect.exe'),'--duel','--run',str(run),'--bots',str(source),'--log',str(run/'dota.log'),'--radiant',str(ports[0]),'--dire',str(ports[1]),'--period',str(a.period),'--offset',str(a.offset),'--until',str(a.until),'--timeout',str(a.wall_timeout)]
  if a.allow_late:command+=['--allow-late']
  if a.policy_address:command+=['--policy-address',a.policy_address,'--policy-team',str(a.policy_team)]
  process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
  try:
   with Game(run,uuid.uuid4().hex,a.timescale,30/a.period,bot_root=source,collect_teacher=False,recording=a.record,content_root=content,game=content.parents[1],game_port=27155+a.worker*10,worldstate_ports=ports,worldstate_frames=1,worldstate_threaded=a.threaded_worldstate,engine_threads=2) as game:
    game.command('host_timescale');(run/'ready').write_text('ready')
    process.wait(timeout=a.wall_timeout)
  finally:
   if process.poll() is None:process.kill();process.wait()
 report=json.loads((run/'duel-result.json').read_text())
 if a.record:
  replay=content/(a.record+'.dem');status={'requested':a.record,'source':str(replay),'complete':False}
  if replay.is_file()and replay.stat().st_size>8:
   saved=run/'replay.dem';shutil.copy2(replay,saved)
   with saved.open('rb')as f:header=f.read(8)
   status.update(path=str(saved),bytes=saved.stat().st_size,header=header.decode('ascii',errors='replace'),complete=header==b'PBDEMS2\x00')
   if status['complete']:report['replay']=str(saved)
  else:status['error']='Dota recording file missing or empty; match result remains independently auditable'
  (run/'recording-status.json').write_text(json.dumps(status,indent=2))
  (run/'duel-result.json').write_text(json.dumps(report,indent=2))
 print(json.dumps(report),flush=True)
 # This file is also the live Lua->Go transport, so compress only after both
 # game and collector have exited. NTFS keeps the exact bytes and original path.
 if (run/'dota.log').is_file():
  try:(run/'log-storage.json').write_text(json.dumps(compact_file(run/'dota.log'),indent=2))
  except OSError as error:(run/'log-storage-warning.txt').write_text(str(error))
 if process.returncode:raise SystemExit(process.returncode)
if __name__=='__main__':main()
