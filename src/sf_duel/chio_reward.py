"""Independent NumPy reference and offline reconstruction of lastorder-fb-v1."""
import gzip
import json
import struct
import numpy as np

SNAP=85
COMPONENTS=('experience','gold_received','mana','health','last_hits','denies','kills','deaths','own_t1','flask','mango','tango','flask_break','outcome')
GAMMA=.997
GAE_LAMBDA=.95


def fraction(value,maximum):return np.clip(value/np.maximum(maximum,1),0,1)
def health(h):return h+1-(1-h)**4
def tower(h):return h+1-(1-h)**3


def components(before,after,zero_sum=True):
    before=np.asarray(before,dtype=np.float64);after=np.asarray(after,dtype=np.float64)
    if before.shape!=after.shape or before.ndim!=2 or before.shape[1]!=SNAP or len(before)%2:
        raise ValueError('paired actor snapshots [N,85] required')
    r=np.zeros((len(before),14),np.float64)
    r[:,0]=.002*(after[:,3]-before[:,3]);r[:,1]=.006*np.maximum(after[:,1]-before[:,1],0)
    r[:,2]=.5*(fraction(after[:,6],after[:,7])-fraction(before[:,6],before[:,7]))
    r[:,3]=health(fraction(after[:,4],after[:,5]))-health(fraction(before[:,4],before[:,5]))
    r[:,4]=-.12*(after[:,12]-before[:,12]);r[:,5]=.1*(after[:,13]-before[:,13])
    r[:,6]=1.5*(after[:,14]-before[:,14]);r[:,7]=-3*(after[:,15]-before[:,15])
    r[:,8]=1.5*(tower(fraction(after[:,8],after[:,9]))-tower(fraction(before[:,8],before[:,9])))
    r[:,9:12]=(after[:,16:19]-before[:,16:19])*np.array([-.3,-.1,-.1])
    for actor in range(len(before)):
        first,end=int(before[actor,25]),int(after[actor,25])
        if end-first not in range(9):raise ValueError('Flask break ring overflow or episode boundary inside transition')
        for event in range(first,end):
            hp,maximum,hp_after,remaining=after[actor,28+(event%8)*6:32+(event%8)*6]
            r[actor,12]+=max(float(health(fraction(hp_after,maximum))-health(fraction(hp+after[actor,84]*remaining,maximum))),-.4)
    decay=np.where(after[:,0]>=360,.6,1.)
    for component in set(range(13))-{6,7}:r[:,component]*=decay
    r[:,13]=np.where((before[:,21]<=0)&(after[:,21]>0),5*after[:,22],0)
    enemy=-r[np.arange(len(r))^1].copy() if zero_sum else np.zeros_like(r)
    enemy[:,[6,7,13]]=0
    result=np.concatenate((r,enemy),axis=1)
    return result,result.sum(axis=1)


def read_record(path):
    with gzip.open(path,'rb') as stream:
        size=struct.unpack('<I',stream.read(4))[0]
        if not 0<size<1_000_000:raise ValueError('bad reward header')
        header=json.loads(stream.read(size))
        if header['format']!='chio-reward-v2':raise ValueError('wrong reward contract')
        data=np.frombuffer(stream.read(),dtype='<f4').reshape(header['shape'])
    return header,data


def audit_record(path,tolerance=5e-5):
    header,data=read_record(path);worst=0.;checked=0
    for step in range(header['frames']):
        frame=data[:,step];parts,total=components(frame[:,:SNAP],frame[:,SNAP:2*SNAP],header['zero_sum'])
        valid=frame[:,-1]>.5
        if valid.any():
            error=max(np.max(np.abs(parts[valid]-frame[valid,2*SNAP:2*SNAP+28])),np.max(np.abs(total[valid]-frame[valid,-2])))
            worst=max(worst,float(error));checked+=int(valid.sum())
    return dict(checked=checked,max_error=worst,tolerance=tolerance,passed=checked>0 and worst<=tolerance)
