"""Reassemble JSON Bot API packets and return atomic Lua orders."""
import json
import os
from pathlib import Path
import time
import re
from ..telemetry import unpack_snapshot

class LogPackets:
    def __init__(self,path,start_marker=None):
        self.path=Path(path);self.offset=0;self.buffer=b'';self.parts={};self.events=[];self.errors=[]
        self.start_marker=start_marker.encode() if start_marker else None
        self.started=start_marker is None
    def read(self):
        if not self.path.exists(): return []
        with self.path.open('rb') as f:
            f.seek(self.offset);data=f.read();self.offset=f.tell()
        self.buffer+=data
        lines=self.buffer.split(b'\n');self.buffer=lines.pop()
        result=[]
        for line in lines:
            if self.start_marker and self.start_marker in line:
                self.started=True;self.parts.clear();continue
            error=line.find(b'SF_RULE_ERROR|')
            if error>=0:
                self.errors.append(json.loads(line[error+14:].rstrip(b'\r')));continue
            if not self.started:continue
            end=line.find(b'SF_END|')
            if end>=0:
                self.events.append(json.loads(line[end+7:].rstrip(b'\r')))
                continue
            start=line.find(b'SF_OBS|')
            if start<0: continue
            header=line[start:].rstrip(b'\r').split(b'|',5)
            if len(header)!=6: raise ValueError('Truncated telemetry header')
            _,pid,seq,part,total,body=header
            key=(int(pid),int(seq));i,n=int(part),int(total)
            if not 1<=i<=n<=128: raise ValueError('Invalid packet fragment count')
            parts=self.parts.setdefault(key,{})
            parts[i]=body
            if len(parts)==n:
                snapshot=unpack_snapshot(json.loads(b''.join(parts[j] for j in range(1,n+1))))
                if snapshot['seq']!=key[1] or snapshot['player']!=key[0]: raise ValueError('Telemetry identity mismatch')
                result.append(snapshot);del self.parts[key]
        if len(self.parts)>8: raise ValueError('Incomplete telemetry packets accumulating')
        return result

def literal(value):
    if isinstance(value,bool): return 'true' if value else 'false'
    if value is None: return 'nil'
    if isinstance(value,(int,float)): return repr(value)
    if isinstance(value,str): return json.dumps(value)
    if isinstance(value,dict): return '{'+','.join('['+literal(k)+']='+literal(v) for k,v in value.items())+'}'
    if isinstance(value,(tuple,list)): return '{'+','.join(literal(v) for v in value)+'}'
    raise TypeError(type(value))

def publish(player,response,root):
    if not isinstance(player,int) or not 0<=player<24: raise ValueError('Unknown controlled player')
    match=response.get('match','')
    if not re.fullmatch('[0-9a-f]{32}',match):raise ValueError('Invalid match namespace')
    path=Path(root)/f'response_{match}_{player}.lua'
    temporary=path.with_suffix('.tmp')
    temporary.write_text('return '+literal(response)+'\n',encoding='utf-8')
    for _ in range(100):
        try: os.replace(temporary,path);return
        except PermissionError: time.sleep(.001)
    raise TimeoutError('Lua response file remained locked')
