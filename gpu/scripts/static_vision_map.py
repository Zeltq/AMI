"""Compile measured building-only base sight; no live engine or policy calls."""
import argparse
import hashlib
import json
from pathlib import Path
import struct

from lane_conformance import atomic

PHASES=('pregame','day','night')


def compile_capture(capture):
    if not capture.get('complete')or not capture.get('static_only'):
        raise ValueError('Complete static-only capture required')
    events=capture.get('static_interventions',[])
    if not events or any(u['day']or u['night']for e in events for u in e['units']):
        raise ValueError('Hero/courier sight intervention is missing')
    rows=capture.get('atlas_rows',[])
    keys=[(r['team'],r['phase'],r['row'])for r in rows]
    expected={(t,p,y)for t in(2,3)for p in PHASES for y in range(128)}
    if len(keys)!=len(set(keys))or set(keys)!=expected:
        raise ValueError('Incomplete or duplicate phase/team rows')
    words=[0]*(512*512)
    for row in rows:
        origin=-8192 if row['team']==2 else 4096
        if row['start']!=origin or row['step']!=32:
            raise ValueError('Unexpected spatial support')
        if any(len(row[k])!=128 or set(row[k])-{'0','1'}for k in('center','plus','minus')):
            raise ValueError('Malformed visibility row')
        if row['center']!=row['plus']or row['center']!=row['minus']:
            raise ValueError('Visibility varies inside a32unit cell; refine capture before use')
        base=(origin+8192)//32
        bit=(row['team']-2)*3+PHASES.index(row['phase'])
        for x,value in enumerate(row['center']):
            words[(base+row['row'])*512+base+x]|=(1<<(bit+6))|(int(value)<<bit)
    return words


def main():
    p=argparse.ArgumentParser();p.add_argument('--capture',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--engine',required=True,type=Path);p.add_argument('--engine-output',required=True,type=Path)
    args=p.parse_args();raw=args.capture.read_bytes();capture=json.loads(raw);words=compile_capture(capture)
    args.output.write_bytes(struct.pack('<'+str(len(words))+'I',*words))
    engine=json.loads(args.engine.read_text());engine['static_vision']=str(args.output.resolve())
    engine['approximations'].append('Measured building-only static visibility replaces rays in ownbase4096square. Base buildings persist before terminal; hero/creep vision stays dynamic. Tree-cut changes to cachedbase static sight are not modeled.')
    atomic(args.engine_output,engine)
    report={'source':str(args.capture),'source_sha256':hashlib.sha256(raw).hexdigest(),'output_sha256':hashlib.sha256(args.output.read_bytes()).hexdigest(),
            'format':'512x512 little-endianu32; phase/team bits0..5, knownbits6..11;32unitcells,origin-8192',
            'coverage':'Each side ownbase [-8192,-4096) or[4096,8192) square; outside uses existing rays',
            'measured_phase_cells':len(capture['atlas_rows'])*128,'subcell_offsets_validated':len(capture['atlas_rows'])*128*2,
            'subcell_disagreements':0,'training_eligible':False}
    atomic(args.output.with_suffix('.json'),report);print(json.dumps(report))


if __name__=='__main__':main()
