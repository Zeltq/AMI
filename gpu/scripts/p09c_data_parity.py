"""Require exact reproduction of old demonstrations before using new telemetry."""
import argparse,json
from pathlib import Path
import numpy as np
from ggd_dataset import read_block
from p09b_run import read,save

def compare(old,new,out):
    old=Path(old);new=Path(new);a=read(old/'dataset.json');b=read(new/'dataset.json')
    assert len(a['files'])==len(b['files'])
    frames=0;files=0;old_fields=read(old/'reward-metadata.json')['fields']
    for x,y in zip(a['files'],b['files']):
        assert x['file']==y['file'];h,o,s=read_block(old/x['file'],x['sha256']);h2,o2,s2=read_block(new/y['file'],y['sha256'])
        assert h['actors']==h2['actors'];assert np.array_equal(o,o2),('observations',x['file']);assert np.array_equal(s,s2),('signals',x['file'])
        name=x['file'].replace('.ggd.gz','.reward.npz')
        with np.load(old/name) as z,np.load(new/name) as zz:
            fields=zz['fields'].tolist();ix=[fields.index(f) for f in old_fields]
            assert np.array_equal(z['state'],zz['state'][...,ix]),('telemetry',name)
            assert np.array_equal(z['valid'],zz['valid']);frames+=int(z['valid'].sum())
        files+=1
    assert a['game_metrics']==b['game_metrics'],'Per-match gameplay metrics differ'
    result=dict(passed=True,files_equal=files,files_total=len(a['files']),valid_frames_equal=frames,matches_equal=len(a['game_metrics']),old_fields_equal=len(old_fields),new_exact_fields=fields[26:],old=str(old),new=str(new))
    save(out,result);print(json.dumps(result))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('old');p.add_argument('new');p.add_argument('out');a=p.parse_args();compare(a.old,a.new,a.out)
