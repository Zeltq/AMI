"""Compare same-tick native protobuf and Lua observations before optimizing."""
import gzip,json,socket,struct,sys,threading,time,uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from prepare_teacher import prepare
from sf_duel.live.install import isolated_content
from sf_duel.live.game import Game
from sf_duel.live.bridge import LogPackets
from sf_duel.protos.worldstate_pb2 import CMsgBotWorldState
from google.protobuf.json_format import MessageToDict

run=ROOT/'runs/native-alignment-001';run.mkdir()
content,bots=isolated_content(ROOT/'workers/00',prepare())
path=bots/'teacher_controller.lua';source=path.read_text()
source=source.replace('s.seq=sequence','''s.seq=sequence
    s.probe.native_action_type=bot:GetCurrentActionType()
    s.probe.entity_index_available=type(bot.GetEntityIndex)=="function"
    if type(bot.GetEntityIndex)=="function" then s.probe.entity_index=bot:GetEntityIndex() end
''')
path.write_text(source)
states={2:{},3:{}};errors=[];stop=threading.Event()
def receive(team,port):
    deadline=time.monotonic()+100
    while not stop.is_set():
        try:conn=socket.create_connection(('127.0.0.1',port),timeout=1);break
        except OSError:
            if time.monotonic()>deadline:errors.append('native connection timeout');return
            time.sleep(.1)
    else:return
    conn.settimeout(1)
    def read(n):
        value=bytearray()
        while len(value)<n and not stop.is_set():
            try:chunk=conn.recv(n-len(value))
            except socket.timeout:continue
            if not chunk:raise EOFError()
            value.extend(chunk)
        if len(value)!=n:raise EOFError()
        return bytes(value)
    with conn,gzip.open(run/f'world-{team}.pb.gz','wb',compresslevel=1) as output:
      try:
        while not stop.is_set():
            header=read(4);size=struct.unpack('<I',header)[0]
            if not 0<size<32_000_000:raise ValueError('Invalid protobuf size')
            payload=read(size);output.write(header+payload)
            state=CMsgBotWorldState.FromString(payload)
            tick=round(state.dota_time*30)
            unit=next((u for u in state.units if u.name=='npc_dota_hero_nevermore' and u.team_id==team),None)
            if unit:
                states[team][tick]=MessageToDict(unit,preserving_proto_field_name=True)
      except EOFError:pass
      except Exception as exc:errors.append(str(exc))

threads=[threading.Thread(target=receive,args=(team,12250+team-2),daemon=True) for team in (2,3)]
for thread in threads:thread.start()
observations=[]
try:
 with Game(run,uuid.uuid4().hex,20,15,bot_root=bots,collect_teacher=True,
           content_root=content,game=content.parents[1],worldstate_ports=(12250,12251),engine_threads=2) as game:
    game.command('echo SF_CAPTURE_READY');packets=LogPackets(game.log,start_marker='SF_CAPTURE_READY')
    deadline=time.monotonic()+100
    while time.monotonic()<deadline:
        observations.extend(packets.read())
        if packets.errors:raise RuntimeError(packets.errors)
        if observations and observations[-1]['time']>=90:break
        time.sleep(.02)
finally:
 stop.set()
 for thread in threads:thread.join(3)
checks={};examples=[]
for s in observations:
    tick=s['demonstration']['decision_tick'];u=states[s['team']].get(tick)
    if not u:continue
    h=s['self']
    for field,native in [('health',u.get('health')),('mana',u.get('mana')),
                         ('x',u.get('location',{}).get('x')),('y',u.get('location',{}).get('y')),
                         ('action_type',u.get('action_type'))]:
        actual=s['probe']['native_action_type'] if field=='action_type' else h.get(field)
        c=checks.setdefault(field,{'equal':0,'different':0,'missing':0})
        if native is None:c['missing']+=1
        elif abs(native-actual)<.0001:c['equal']+=1
        else:
            c['different']+=1
            if len(examples)<12:examples.append({'team':s['team'],'time':s['time'],'field':field,
                'lua':actual,'native':native,'command':s['demonstration']['command']})
report={'checks':checks,'mismatches':examples,'native_frames':{t:len(v) for t,v in states.items()},
        'lua_frames':len(observations),'errors':errors,
        'entity_index_available':observations[0]['probe'].get('entity_index_available') if observations else None,
        'example_unit':next(iter(states[2].values()),{})}
(run/'alignment.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report),flush=True)
