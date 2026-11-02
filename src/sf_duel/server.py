"""Bot API endpoint. A single worker batches inference and owns recurrent state."""
import argparse
import gzip
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
from pathlib import Path
from queue import Queue,Empty
import threading
import time
import traceback
import socketserver
import socket
import struct
import uuid

class Service:
    def __init__(self,mode,run,device='cuda',seed=1,service_id=None):
        self.mode=mode;self.run=Path(run);self.run.mkdir(parents=True,exist_ok=True)
        self.queue=Queue();self.states={};self.previous={};self.replies={};self.stats={'requests':0,'batches':{},'errors':[], 'api_errors':{},'latest':{},'inference_timing':{'count':0,'total':0.,'maximum':0.}}
        self.snapshots=gzip.open(self.run/'snapshots.jsonl.gz','wt',encoding='utf-8')
        self.records=gzip.open(self.run/'decisions.jsonl.gz','wt',encoding='utf-8')
        self.device=device
        self.stats['service_id']=service_id or uuid.uuid4().hex
        if mode=='policy':
            import torch
            from .model import Policy
            from .observations import warmup_batch
            torch.set_num_threads(2)
            torch.manual_seed(seed)
            self.model=Policy().to(device).eval()
            self.stats['parameters']=sum(p.numel() for p in self.model.parameters())
            torch.save({'state_dict':self.model.state_dict(),'seed':seed,'spec_version':1},self.run/'random-policy.pt')
            started=time.monotonic()
            with torch.inference_mode():
                for size in (1,2):
                    warm=warmup_batch(size,device)
                    for _ in range(2):self.model.decide(warm)
                if device=='cuda':torch.cuda.synchronize()
            self.stats['warmup_seconds']=time.monotonic()-started
        elif mode=='teacher':
            from .teacher import Teacher
            self.teachers={}
        self.worker=threading.Thread(target=self.work,daemon=True);self.worker.start()

    def work(self):
        while True:
            first=self.queue.get()
            requests=[first]
            deadline=time.monotonic()+.002
            while len(requests)<32:
                try: requests.append(self.queue.get(timeout=max(0,deadline-time.monotonic())))
                except Empty: break
            try:
                start=time.monotonic()
                output=self.decide([r[0] for r in requests])
                elapsed=time.monotonic()-start;timing=self.stats['inference_timing']
                timing['count']+=1;timing['total']+=elapsed;timing['maximum']=max(timing['maximum'],elapsed)
                for request,result in zip(requests,output): request[1].put((200,result))
            except BaseException as exc:
                detail=traceback.format_exc();self.stats['errors'].append(detail)
                print(detail,flush=True)
                for request in requests: request[1].put((500,{'error':str(exc)}))

    def decide(self,snapshots):
        from .spec import blank_action
        from .observations import encode,batch,target_ref
        if len(snapshots)==1:
            s=snapshots[0];key=(s['match'],s['team'],s['seq'])
            if key in self.replies: return [self.replies[key]]
        elif any((s['match'],s['team'],s['seq']) in self.replies for s in snapshots):
            return [self.decide([s])[0] for s in snapshots]
        for s in snapshots:
            for key in self.replies:
                if key[:2]==(s['match'],s['team']) and s['seq']<key[2]:
                    raise ValueError('Out-of-order observation would corrupt recurrent state')
            self.snapshots.write(json.dumps(s,separators=(',',':'))+'\n')
            self.stats['api_errors'].update(s.get('errors',{}))
            self.stats['latest'][str(s['team'])]={'time':s['time'],'self':s.get('self'),
                'towers':s.get('towers'),'execution':s.get('execution'),'items':s.get('items')}
        self.snapshots.flush()
        if self.mode=='probe':
            self.stats['requests']+=len(snapshots)
            replies=[dict(action=0,buy=0,seq=s['seq'],match=s['match'],observed_time=s['time']) for s in snapshots]
            for s,r in zip(snapshots,replies): self.remember(s,r)
            return replies
        active=[];encoded=[];answers=[dict(action=0,buy=0,seq=s['seq'],match=s['match'],observed_time=s['time']) for s in snapshots]
        for index,s in enumerate(snapshots):
            if not s.get('self') or s['self'].get('health',0)<=0: continue
            key=(s['match'],s['team'])
            encoded.append(encode(s,self.previous.get(key)))
            active.append(index)
        if not encoded: return answers
        if self.mode=='policy':
            import torch
            with torch.inference_mode():
                b=batch(encoded,self.device)
                keys=[(e.snapshot['match'],e.snapshot['team']) for e in encoded]
                if len(set(keys))!=len(keys): raise ValueError('Two concurrent decisions for one recurrent state')
                hs=[self.states[k] if k in self.states else self.model.initial_state(1,self.device) for k in keys]
                state=tuple(torch.cat([s[j] for s in hs],1) for j in range(2))
                chosen,logp,entropy,value,new_state=self.model.decide(b,state)
                packed=torch.stack([chosen[k] for k in ('action','x','y','target','item','buy')]+[logp,value],-1).cpu().tolist()
                actions=[]
                for j,row in enumerate(packed):
                    self.states[keys[j]]=tuple(v[:,j:j+1].contiguous() for v in new_state)
                    a={k:int(row[i]) for i,k in enumerate(('action','x','y','target','item','buy'))}
                    ref=target_ref(encoded[j],a['target'],b['creeps'].shape[1],b['trees'].shape[1])
                    actions.append((a,ref,{'log_prob':row[-2],'value':row[-1]}))
        else:
            from .teacher import Teacher
            actions=[]
            for e in encoded:
                key=(e.snapshot['match'],e.snapshot['team'])
                if key not in self.teachers:self.teachers[key]=Teacher()
                a,ref=self.teachers[key].decide(e.snapshot)
                actions.append((a,ref,{}))
        from .spec import COORDINATES,Act
        for index,e,(a,ref,extra) in zip(active,encoded,actions):
            s=e.snapshot;key=(s['match'],s['team'])
            command=dict(a,seq=s['seq'],match=s['match'],observed_time=s['time'])
            if a['action'] in (Act.MOVE,Act.MOVE_DIRECT):
                command['x']=s['self']['x']+COORDINATES[a['x']]
                command['y']=s['self']['y']+COORDINATES[a['y']]
            if a['action']==Act.ITEM and a['item']==3:
                command['x']=ref['x']+COORDINATES[a['x']]
                command['y']=ref['y']+COORDINATES[a['y']]
            if ref is not None:
                command['target_id']=ref.get('id')
                command['target_type']=ref.get('type')
                a['target_offset']=((ref.get('x',s['self']['x'])-s['self']['x'])/2048,
                                    (ref.get('y',s['self']['y'])-s['self']['y'])/2048)
            self.previous[key]=dict(a)
            self.records.write(json.dumps({'match':s['match'],'team':s['team'],'seq':s['seq'],
                'time':s['time'],'action':a,'command':command,'execution':s.get('execution'),
                'controller':self.mode,'target_layout':({'creeps':b['creeps'].shape[1],
                    'trees':b['trees'].shape[1]} if self.mode=='policy' else None),**extra},separators=(',',':'))+'\n')
            answers[index]=command
        self.records.flush()
        self.stats['requests']+=len(snapshots)
        n=len(encoded);self.stats['batches'][str(n)]=self.stats['batches'].get(str(n),0)+1
        for s,r in zip(snapshots,answers): self.remember(s,r)
        return answers

    def remember(self,snapshot,reply):
        prefix=(snapshot['match'],snapshot['team'])
        for key in list(self.replies):
            if key[:2]==prefix:del self.replies[key]
        self.replies[(*prefix,snapshot['seq'])]=reply

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=['probe','policy','teacher'],required=True)
    parser.add_argument('--run',required=True);parser.add_argument('--device',default='cuda')
    parser.add_argument('--service-id')
    args=parser.parse_args();service=Service(args.mode,args.run,args.device,service_id=args.service_id)
    class InferenceHandler(socketserver.BaseRequestHandler):
        def handle(self):
            self.request.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1)
            from .live.rpc import exact
            while True:
                try:
                    size=struct.unpack('<I',exact(self.request,4))[0]
                    if not 0<size<=4_000_000:raise ValueError('Invalid request size')
                    snapshot=json.loads(exact(self.request,size));reply=Queue(maxsize=1)
                    service.queue.put((snapshot,reply));status,result=reply.get(timeout=20)
                    body=json.dumps(result,separators=(',',':')).encode()
                    self.request.sendall(struct.pack('<I',len(body))+body)
                except (OSError,ConnectionError):return
    class InferenceServer(socketserver.ThreadingTCPServer):
        allow_reuse_address=True
        daemon_threads=True
    inference=InferenceServer(('0.0.0.0',12261),InferenceHandler)
    threading.Thread(target=inference.serve_forever,daemon=True).start()
    class Handler(BaseHTTPRequestHandler):
        protocol_version='HTTP/1.1'
        disable_nagle_algorithm=True
        wbufsize=65536
        def do_GET(self):
            body=json.dumps(service.stats).encode();self.send_response(200)
            self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
        def do_POST(self):
            if self.path=='/shutdown':
                if self.headers.get('X-SF-Service')!=service.stats['service_id']:
                    self.send_response(403);self.send_header('Content-Length','0');self.end_headers();return
                self.send_response(200);self.send_header('Content-Length','0');self.end_headers()
                threading.Thread(target=server.shutdown,daemon=True).start()
                return
            try:
                size=int(self.headers['Content-Length'])
                if size>4_000_000: raise ValueError('Oversized observation')
                snapshot=json.loads(self.rfile.read(size));reply=Queue(maxsize=1)
                service.queue.put((snapshot,reply));status,result=reply.get(timeout=15)
            except BaseException as exc: status,result=500,{'error':str(exc)}
            body=json.dumps(result,separators=(',',':')).encode()
            self.send_response(status);self.send_header('Content-Type','application/json')
            self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
        def log_message(self,*args): pass
    server=ThreadingHTTPServer(('0.0.0.0',12260),Handler)
    print(json.dumps({'ready':True,'mode':args.mode,**service.stats}),flush=True)
    try: server.serve_forever()
    finally:
        inference.shutdown();inference.server_close()
        service.snapshots.close();service.records.close()
        (service.run/'service-result.json').write_text(json.dumps(service.stats,indent=2))

if __name__=='__main__': main()
