"""Persistent protobuf inference service; one GPU worker batches independent actors."""
import argparse,json,os,socket,socketserver,struct,threading,time,traceback,uuid,hashlib,math
from pathlib import Path
from queue import Queue,Empty,Full
import torch
from .model import Policy
from .native_data import NativeAdapter,decode_action,action_label,examples
from .observations import encode,batch
from .protos.duel_pb2 import Record

def policy_contract(saved):
 pipeline=saved.get('gpu_pipeline',{})
 reset=pipeline.get('recurrent_reset_interval',0);storage=pipeline.get('observation_storage','f32')
 if reset not in (0,128):raise ValueError('Unsupported recurrent reset contract')
 if storage not in ('f32','f16'):raise ValueError('Unsupported observation storage contract')
 return reset,storage

def memory_clock_group(dota_time,origin):
 if not math.isfinite(dota_time)or not math.isfinite(origin):raise ValueError('Nonfinite memory clock')
 return (round(dota_time*30)-round(origin*30))//512

def exact(conn,n):
 data=bytearray()
 while len(data)<n:
  chunk=conn.recv(n-len(data))
  if not chunk:raise EOFError()
  # WSL localhost forwarding can combine proxy-side Nagle buffering with Linux
  # delayed ACKs (~40 ms). Re-arm QUICKACK after each received fragment so the
  # proxy immediately releases the remainder of a length-prefixed request.
  quickack=getattr(socket,'TCP_QUICKACK',None)
  if quickack is not None:conn.setsockopt(socket.IPPROTO_TCP,quickack,1)
  data.extend(chunk)
 return bytes(data)

class Service:
 def __init__(self,checkpoint,out,teacher_hash=None,max_batch_warmup=4,stochastic=False,record_root=None,opponents=(),seed=0,device='cuda',reset_origin=None):
  self.out=Path(out);self.out.mkdir(parents=True,exist_ok=True);self.queue=Queue();self.actors={};self.stats={'requests':0,'batches':{},'episodes':{},'errors':[],'checkpoint':str(checkpoint),'training_started':False}
  self.device=device;torch.set_num_threads(2);torch.manual_seed(seed);self.stats.update(seed=seed,device=device);self.model=Policy().to(device).eval();saved=torch.load(checkpoint,map_location='cpu',weights_only=False);self.model.load_state_dict(saved['state_dict'],strict=True)
  self.recurrent_reset_interval,self.observation_storage=policy_contract(saved)
  if reset_origin is not None and(not math.isfinite(reset_origin)or self.recurrent_reset_interval!=128):raise ValueError('Game clock requires the128-step memory contract')
  self.reset_origin=reset_origin;self.stats['memory_reset_origin_seconds']=reset_origin
  self.contracts={'current':(self.recurrent_reset_interval,self.observation_storage)}
  if teacher_hash and saved['config']['teacher_policy_sha256']!=teacher_hash:raise ValueError('Checkpoint teacher provenance mismatch')
  if any(not torch.isfinite(v).all() for v in self.model.state_dict().values()):raise ValueError('Nonfinite checkpoint')
  self.stochastic=stochastic;self.models={'current':self.model};self.hashes={'current':hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()};self.canonical_policy={'current':'current'};self.last_save=0.;self.batch_wait_ms=2.;self.max_batch_size=max(1,min(32,max_batch_warmup))
  for index,path in enumerate(opponents):
   saved_opponent=torch.load(path,map_location='cpu',weights_only=False)
   self.contracts[str(index)]=policy_contract(saved_opponent)
   if reset_origin is not None and self.contracts[str(index)][0]!=128:raise ValueError('Opponent memory clock contract differs')
   m=Policy().to(device).eval();m.load_state_dict(saved_opponent['state_dict'],strict=True)
   self.models[str(index)]=m;self.hashes[str(index)]=hashlib.sha256(Path(path).read_bytes()).hexdigest()
   self.canonical_policy[str(index)]=next(k for k,v in self.hashes.items() if v==self.hashes[str(index)])
  self.writer=None
  self.route_cache={};self.registered_files=set();(self.out/'registrations').mkdir(exist_ok=True)
  def preload():
   while True:
    try:self.prefetch_routes()
    except Exception as error:print('Route prefetch: '+str(error),flush=True)
    time.sleep(.1)
  threading.Thread(target=preload,daemon=True).start()
  if record_root:
   from .selfplay.rollouts import RolloutWriter
   self.writer=RolloutWriter(record_root)
  from .observations import warmup_batch
  from .spec import UNIT_FIELDS,TREE_FIELDS,PROJECTILE_FIELDS
  with torch.inference_mode():
   for n in range(1,max_batch_warmup+1):
    for pc in (16,32,64):
     for pt in (32,64,128):
      for pp in (16,32):
       b=warmup_batch(n,device)
       for group,count,width,mask in [('creeps',pc,2*len(UNIT_FIELDS),'creep_valid'),('trees',pt,2*len(TREE_FIELDS),'tree_valid'),('projectiles',pp,2*len(PROJECTILE_FIELDS),'projectile_valid')]:
        b[group]=torch.zeros((n,count,width),device=device);b[mask]=torch.zeros((n,count),device=device,dtype=torch.bool)
       b['creep_ally']=torch.zeros((n,pc),device=device,dtype=torch.bool)
       for key in ('attack_mask','tree_mask','tp_mask','self_mask'):b[key]=torch.zeros((n,7+pc+pt),device=device,dtype=torch.bool)
       b['self_mask'][:,0]=True
       for _ in range(2):self.model.decide(b,need_entropy=False)
  self.telemetry_queue=Queue(1);self.telemetry_error=None;self.telemetry_ms=0.;self.rpc_timings=Queue(64)
  self.telemetry_thread=threading.Thread(target=self.telemetry_work,daemon=True);self.telemetry_thread.start()
  if device=='cuda':torch.cuda.synchronize()
  threading.Thread(target=self.work,daemon=True).start()
 def work(self):
  while True:
   requests=[self.queue.get()];batch_started=time.monotonic();deadline=batch_started+self.batch_wait_ms/1000
   while len(requests)<self.max_batch_size:
    try:requests.append(self.queue.get(timeout=max(0,deadline-time.monotonic())))
    except Empty:break
   try:
    for _,q in requests:q.dequeued_monotonic=time.monotonic()
    replies=self.decide([r for r,q in requests])
    for (_,q),reply in zip(requests,replies):q.put(reply)
    elapsed_ms=(time.monotonic()-batch_started)*1000
    if elapsed_ms>50:
     events=self.stats.setdefault('slow_batches',[])
     events.append({'wall_time':time.time(),'milliseconds':elapsed_ms,'requests':[{'episode':r.episode,'team':r.decision.team,'sequence':r.decision.sequence,'tick':r.decision.observed_tick} for r,_ in requests]})
     del events[:-64]
    if time.monotonic()-self.last_save>=2:
     self.stats['batch_wait_ms']=self.batch_wait_ms
     self.queue_telemetry();self.last_save=time.monotonic()
   except BaseException:
    error=traceback.format_exc();self.stats['errors'].append(error);self.queue_telemetry();print(error,flush=True)
    for _,q in requests:q.put(None)
 def queue_telemetry(self):
  # Snapshot on the owning inference thread, but never wait for filesystem I/O.
  timings=getattr(self,'rpc_timings',None)
  if timings is not None:
   for _ in range(64):
    try:event=timings.get_nowait()
    except Empty:break
    events=self.stats.setdefault('slow_rpc',[]);events.append(event);del events[:-64]
  self.stats['telemetry_write_ms']=self.telemetry_ms
  if self.telemetry_error:self.stats['telemetry_error']=self.telemetry_error
  payload=json.dumps(self.stats)
  try:self.telemetry_queue.put_nowait(payload)
  except Full:
   try:self.telemetry_queue.get_nowait()
   except Empty:pass
   self.telemetry_queue.put_nowait(payload)
 def record_rpc_timing(self,request,started,received,dequeued,answered,sent):
  if (sent-started)*1000<=50:return
  event={'wall_time':time.time(),'episode':request.episode,'team':request.decision.team,'sequence':request.decision.sequence,'tick':request.decision.observed_tick,
         'read_parse_ms':(received-started)*1000,'queue_ms':(dequeued-received)*1000,'service_ms':(answered-dequeued)*1000,'send_ms':(sent-answered)*1000,'total_ms':(sent-started)*1000}
  try:self.rpc_timings.put_nowait(event)
  except Full:pass
 def telemetry_work(self):
  while True:
   payload=self.telemetry_queue.get()
   if payload is None:return
   started=time.monotonic()
   try:
    control=self.out/'runtime.json'
    if control.exists():
     wait_ms=float(json.loads(control.read_text()).get('batch_wait_ms',self.batch_wait_ms))
     if not 0<=wait_ms<=50:raise ValueError('Invalid batch wait')
     self.batch_wait_ms=wait_ms
    self.write_stats(payload)
   except Exception:self.telemetry_error=traceback.format_exc()
   finally:self.telemetry_ms=(time.monotonic()-started)*1000
 def finish_telemetry(self):
  self.telemetry_queue.put(None);self.telemetry_thread.join()
 def prefetch_routes(self):
  for path in (self.out/'registrations').glob('*.json'):
   if path.name in self.registered_files:continue
   value=json.loads(path.read_text());route=value['routing']
   if any(policy not in self.models for policy in route.values()):raise ValueError('Unknown preregistered policy')
   self.route_cache[value['episode']]=route;self.registered_files.add(path.name)
 def save(self):
  self.write_stats(json.dumps(self.stats))
 def write_stats(self,payload):
  temporary=self.out/'stats.tmp';temporary.write_text(payload);temporary.replace(self.out/'stats.json')
 def decide(self,requests):
  # A single GPU queue, batching separately for each immutable policy version.
  groups={}
  for i,r in enumerate(requests):
   if r.HasField('reward_frame'):raise ValueError('Privileged reward state must not enter policy RPC')
   key=(r.episode,r.decision.team)
   if key not in self.actors:
    route_path=r.episode.replace('\\','/')
    if os.name!='nt'and len(route_path)>1 and route_path[1]==':':route_path='/mnt/'+route_path[0].lower()+route_path[2:]
    path=Path(route_path)/'policy-routing.json';policy='current';route=getattr(self,'route_cache',{}).get(r.episode)
    if route is not None:
     policy=route[str(r.decision.team)];self.stats['route_cache_hits']=self.stats.get('route_cache_hits',0)+1
    elif path.exists():
     policy=json.loads(path.read_text())[str(r.decision.team)];self.stats['route_cache_misses']=self.stats.get('route_cache_misses',0)+1
    if policy not in self.models:raise ValueError('Unregistered opponent checkpoint')
    self.actors[key]={'adapter':NativeAdapter(),'state':self.models[policy].initial_state(1),'previous':None,'seq':0,'shots':set(),'cooldowns':{},'policy':policy}
   policy=self.actors[key]['policy'];canonical=getattr(self,'canonical_policy',{}).get(policy,policy);groups.setdefault(canonical,[]).append((i,r))
  replies=[None]*len(requests)
  for policy,group in groups.items():
   for (index,_),reply in zip(group,self.decide_group([r for _,r in group],policy)):replies[index]=reply
  return replies
 def decide_group(self,requests,policy):
  reset_interval,storage=getattr(self,'contracts',{}).get(policy,(getattr(self,'recurrent_reset_interval',0),getattr(self,'observation_storage','f32')))
  keys=[];observations=[];states=[]
  for r in requests:
   d=r.decision;key=(r.episode,d.team)
   if not r.episode:raise ValueError('Missing episode identity')
   actor=self.actors[key]
   if d.sequence!=actor['seq']+1:raise ValueError('Out-of-order recurrent request')
   # Native Think also pauses during fear: an RPC gap is not proof of death.
   # Own scoreboard deaths are public and authoritative. Keep legacy
   # checkpoints' recurrent contract unchanged.
   deaths=next((p.deaths for p in r.observation.players if p.team_id==d.team and p.hero_id==11),actor.get('deaths',0))
   life_reset=bool(reset_interval and actor.get('tick') is not None and deaths>actor.get('deaths',0))
   if life_reset:
    actor['previous']=None;self.stats['life_resets']=self.stats.get('life_resets',0)+1
   snapshot=actor['adapter'].snapshot(r,r.model_supplement);e=encode(snapshot,actor['previous'])
   origin=getattr(self,'reset_origin',None)
   clock_group=memory_clock_group(r.observation.dota_time,origin)if origin is not None else None
   periodic=(actor.get('clock_group')!=clock_group)if origin is not None else bool(reset_interval and actor['seq']%reset_interval==0)
   if life_reset or periodic:
    actor['state']=self.models[policy].initial_state(1)
   actor['clock_group']=clock_group
   actor['tick']=d.observed_tick
   actor['deaths']=deaths
   keys.append(key);observations.append(e);states.append(actor['state'])
   h=next(u for u in r.observation.units if u.name=='npc_dota_hero_nevermore' and u.team_id==d.team)
   stats=self.stats['episodes'].setdefault(r.episode+':'+str(d.team),{'team':d.team,'first_mid_time':None,'creep_projectiles':0,'raze_casts':0,'last_hits':0,'denies':0,'decisions':0})
   if stats['first_mid_time'] is None and (h.location.x*h.location.x+h.location.y*h.location.y)<2000**2 and abs(h.location.x-h.location.y)<1200:stats['first_mid_time']=r.observation.dota_time
   for u in r.observation.units:
    if not (u.name.startswith('npc_dota_creep_') or u.name.endswith('_siege')):continue
    for shot in u.incoming_tracking_projectiles:
     if shot.is_attack and shot.caster_handle==h.handle:
      token=(shot.handle,u.handle,h.last_attack_time)
      if token not in actor['shots']:actor['shots'].add(token);stats['creep_projectiles']+=1
   for ability in h.abilities:
    if ability.ability_id in (5059,5060,5061):
     previous=actor['cooldowns'].get(ability.ability_id,ability.cooldown_remaining)
     if ability.cooldown_remaining>previous+1:stats['raze_casts']+=1
     actor['cooldowns'][ability.ability_id]=ability.cooldown_remaining
   stats.update(last_hits=h.last_hits,denies=h.denies,last_time=r.observation.dota_time,decisions=d.sequence)
  if len(set(keys))!=len(keys):raise ValueError('Two pending states for one actor')
  started=time.perf_counter()
  with torch.inference_mode():
   b=batch(observations,getattr(self,'device','cuda'))
   if storage=='f16':
    b={k:v.half().float() if v.is_floating_point() else v for k,v in b.items()}
   memory=tuple(torch.cat([s[i] for s in states],1) for i in range(2))
   input_memory=memory
   chosen,logp,_,values,memory=self.models[policy].decide(b,memory,deterministic=not self.stochastic,need_entropy=False)
   action_tensor=torch.stack([chosen[k] for k in ('action','buy','item','target','x','y')],-1)
   if self.writer:
    # The small integer action indices are exactly representable in float32. All policy
    # outputs and recurrent state use one GPU->CPU synchronization for recording.
    download=torch.cat((action_tensor.float(),logp[:,None],values[:,None],input_memory[0][0],input_memory[1][0]),dim=1).cpu().numpy()
    packed=download[:,:6];saved_logp=download[:,6];saved_value=download[:,7];width=input_memory[0].shape[-1]
    saved_h=download[:,8:8+width];saved_c=download[:,8+width:8+2*width]
   else:packed=action_tensor.cpu().tolist()
  replies=[]
  for index,(r,e,key) in enumerate(zip(requests,observations,keys)):
   a=dict(zip(('action','buy','item','target','x','y'),map(int,packed[index])))
   d=decode_action(a,e,b['creeps'].shape[1],b['trees'].shape[1],r.decision.sequence,r.decision.observed_tick,r.decision.execute_tick-r.decision.observed_tick)
   actor=self.actors[key];actor['state']=tuple(v[:,index:index+1].contiguous() for v in memory);actor['previous']=action_label(d,e);actor['seq']=d.sequence
   if self.writer:
    from .selfplay.rewards import server_state
    actor_policy=actor['policy']
    self.writer.append(r.episode,d.team,self.hashes[actor_policy],actor_policy=='current',dict(observation=e.arrays,action=actor['previous'],
     old_log_prob=saved_logp[index],old_value=saved_value[index],h=saved_h[index],c=saved_c[index],
     sequence=d.sequence,tick=d.observed_tick,time=r.observation.dota_time,reward_state=server_state(r.observation,d.team)))
   replies.append(d.SerializeToString())
  self.stats['requests']+=len(replies);size=str(len(replies));self.stats['batches'][size]=self.stats['batches'].get(size,0)+1;self.stats['latest_inference_ms']=(time.perf_counter()-started)*1000
  return replies

def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);p.add_argument('--teacher-hash');p.add_argument('--max-batch-warmup',type=int,default=4);p.add_argument('--port',type=int,default=12360);p.add_argument('--stochastic',action='store_true');p.add_argument('--record-root');p.add_argument('--opponent',action='append',default=[]);p.add_argument('--seed',type=int,default=0);p.add_argument('--device',choices=['cpu','cuda'],default='cuda');p.add_argument('--reset-origin',type=float);a=p.parse_args();service=Service(a.checkpoint,a.output,a.teacher_hash,a.max_batch_warmup,a.stochastic,a.record_root,a.opponent,a.seed,a.device,a.reset_origin)
 class Handler(socketserver.BaseRequestHandler):
  def handle(self):
   self.request.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1)
   try:
    while True:
     n=struct.unpack('<I',exact(self.request,4))[0]
     if not 0<n<32*1024*1024:raise ValueError('Bad native request size')
     started=time.monotonic();request=Record.FromString(exact(self.request,n));received=time.monotonic();answer=Queue(1);service.queue.put((request,answer));reply=answer.get(timeout=10);answered=time.monotonic()
     if reply is None:return
     self.request.sendall(struct.pack('<I',len(reply))+reply)
     service.record_rpc_timing(request,started,received,answer.dequeued_monotonic,answered,time.monotonic())
   except (EOFError,ConnectionError):return
 class Server(socketserver.ThreadingTCPServer):allow_reuse_address=True;daemon_threads=True
 with Server(('0.0.0.0',a.port),Handler) as server:
  def stop():
   while not (service.out/'stop').exists():time.sleep(.5)
   server.shutdown()
  threading.Thread(target=stop,daemon=True).start();(service.out/'ready.json').write_text(json.dumps({'pid':os.getpid(),'port':a.port,'device':torch.cuda.get_device_name() if a.device=='cuda' else 'cpu'}));server.serve_forever(poll_interval=.2)
 if service.writer:service.writer.close()
 service.finish_telemetry()
 service.save()
if __name__=='__main__':main()
