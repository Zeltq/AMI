"""Prepare a two-match v4 fixture: whole matches remain in separate splits."""
import gzip,hashlib,json,struct
from pathlib import Path
from ggd_dataset import inspect
from p09_run import RUN,REPORT,save

def main():
    source=Path('C:/AMI/runs/p07/collect-v4');target=RUN/'training-fixture';target.mkdir(exist_ok=False)
    manifest=json.loads((source/'dataset.json').read_text())
    for entry in manifest['files']:
        with gzip.open(source/entry['file'],'rb') as f:
            magic=f.read(8);h=json.loads(f.read(struct.unpack('<I',f.read(4))[0]));body=f.read()
        h['actors']=[a if a<2 else 12+a%2 for a in h['actors']];entry['actors']=h['actors']
        header=json.dumps(h).encode()
        with gzip.open(target/entry['file'],'wb',compresslevel=1) as f:f.write(magic+struct.pack('<I',len(header))+header+body)
        entry['sha256']=hashlib.sha256((target/entry['file']).read_bytes()).hexdigest();entry['bytes']=(target/entry['file']).stat().st_size
    manifest['fixture_note']='P07 two-match fixture only; match 1 reassigned to validation match ID 6; no fixture weights reused'
    save(target/'dataset.json',manifest)
    save(REPORT/'training-fixture.json',inspect(target,RUN/'training-fixture.npz'))
    save(REPORT/'train-smoke.json',dict(dataset=str(target),epochs=1,batch_sequences=8,learning_rate=1e-4,seed=20261102,gradient_norm=1.,attack_weight=4.,timing_honest=True,timing_oracle=False,model_name='ZOg-smoke',action_weights={'6':8.,'7':8.,'8':8.},hero_attack_weight=5.))

if __name__=='__main__':main()
