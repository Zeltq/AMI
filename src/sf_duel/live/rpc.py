"""Persistent local HTTP transport; POST retries require server idempotency."""
import http.client
import json
import socket
import struct
import threading

local=threading.local()
def exact(sock,n):
    data=bytearray()
    while len(data)<n:
        block=sock.recv(n-len(data))
        if not block:raise ConnectionError('Inference connection closed')
        data.extend(block)
    return bytes(data)

def infer(payload):
    body=json.dumps(payload,separators=(',',':')).encode()
    for attempt in range(2):
        sock=getattr(local,'inference',None)
        if sock is None:
            sock=socket.create_connection(('127.0.0.1',12261),timeout=20)
            sock.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1);local.inference=sock
        try:
            # One send for header + body avoids a delayed-ACK bubble at the WSL boundary.
            sock.sendall(struct.pack('<I',len(body))+body)
            size=struct.unpack('<I',exact(sock,4))[0]
            if not 0<size<=4_000_000:raise ValueError('Invalid response length')
            response=json.loads(exact(sock,size))
            if 'error' in response:raise RuntimeError(response['error'])
            return response
        except OSError:
            sock.close();local.inference=None
            if attempt:raise

def request(method,path,payload=None):
    if method=='POST' and path=='/act':return infer(payload)
    body=None if payload is None else json.dumps(payload,separators=(',',':')).encode()
    for attempt in range(2):
        connection=getattr(local,'connection',None)
        if connection is None:
            connection=http.client.HTTPConnection('127.0.0.1',12260,timeout=20)
            connection.connect();connection.sock.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1)
            local.connection=connection
        try:
            connection.request(method,path,body=body,headers={'Content-Type':'application/json'})
            response=connection.getresponse();data=response.read()
            if response.status!=200: raise RuntimeError(data.decode(errors='replace'))
            return json.loads(data) if data else None
        except (OSError,http.client.HTTPException):
            connection.close();local.connection=None
            if attempt: raise
