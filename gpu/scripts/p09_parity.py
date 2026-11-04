"""Compare valid decision records against the preserved P07 ggd-v4 collection."""
import argparse,json
from pathlib import Path
import numpy as np
from ggd_dataset import read_block

def main(old,new,output):
    old=Path(old);new=Path(new);a=json.loads((old/'dataset.json').read_text());b=json.loads((new/'dataset.json').read_text())
    for key in ('seed','seconds','teacher_version','observation_profile'):assert a['config'][key]==b['config'][key]
    by_block={}
    for entry in b['files']:by_block.setdefault(int(entry['file'].split('-')[1]),[]).append(entry)
    values=signals=frames=0
    for entry in a['files']:
        h,o,s=read_block(old/entry['file']);block=int(entry['file'].split('-')[1]);records={}
        for other in by_block[block]:
            nh,no,ns=read_block(new/other['file'])
            for i,actor in enumerate(nh['actors']):records[actor]=(nh,no[i],ns[i])
        for i,actor in enumerate(h['actors']):
            nh,no,ns=records[actor];valid=s[i,:,11]>.5
            assert np.array_equal(s[i],ns),(block,actor,'signals')
            lookup={c:j for j,c in enumerate(nh['columns'])};oldlookup={c:j for j,c in enumerate(h['columns'])}
            common=sorted(set(lookup)&set(oldlookup))
            assert np.array_equal(o[i,valid][:,[oldlookup[c] for c in common]],no[valid][:,[lookup[c] for c in common]]),(block,actor,'observations')
            for cols,indices,data in [(set(oldlookup)-set(lookup),oldlookup,o[i]),(set(lookup)-set(oldlookup),lookup,no)]:
                assert not np.any(data[valid][:,[indices[c] for c in cols]]),(block,actor,'nonzero extra padding')
            frames+=int(valid.sum());values+=int(valid.sum())*len(common);signals+=s[i].size
    report=dict(valid_frames=frames,identical_fp16_values=values,identical_signal_values=signals,old=str(old),new=str(new),matches=a['config']['games'],comparison='bitwise valid observations, all signal/reset/valid values; extra bucket padding zero')
    Path(output).write_text(json.dumps(report,indent=2));print(json.dumps(report))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('old');p.add_argument('new');p.add_argument('output');a=p.parse_args();main(a.old,a.new,a.output)
