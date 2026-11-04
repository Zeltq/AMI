"""Reuse verified frozen ZOg-d1 features; replace every target with event-v2 MC."""
import argparse,json
from pathlib import Path
import numpy as np
from p09b_run import read,save,sha
from p09c_reward import targets
from ggd_dataset import read_block
from dagger_data import eligible

def main():
    p=argparse.ArgumentParser();p.add_argument('output');a=p.parse_args();out=Path(a.output)
    previous=Path('C:/AMI/runs/p09/P09b/critic');config=read(previous/'config.json')
    initial=Path('C:/AMI/runs/p09/P09b/ZOg-d1/last.pt')
    assert sha(initial)==read(Path(__file__).resolve().parents[2]/'reports/P09b/selection.json')['checkpoint_sha256']
    assert config['checkpoint'].replace('/mnt/c/','C:/').replace('\\','/')==str(initial).replace('\\','/')
    rows=[]
    for index,name in enumerate(('demo','d1','d2')):
        root=Path('C:/AMI/runs/p09/P09c')/(name+'-data')
        proof=read(Path(__file__).resolve().parents[2]/'reports/P09c'/(name+'-parity.json'));assert proof['passed']
        target=out/f'targets-{index}'
        if not (target/'targets.json').exists():targets(root,target)
        tm=read(target/'targets.json');assert tm['version']==3 and tm['metadata_sha256']==sha(root/'reward-metadata.json')
        lookup={x['demonstration']:x['file'] for x in tm['files']}
        dest=out/f'features-{index}';dest.mkdir(parents=True,exist_ok=True);frames=0;count=0
        for entry in read(root/'dataset.json')['files']:
            h,obs,sig=read_block(root/entry['file'],entry['sha256']);source=h.get('source','teacher-demo')
            ids=[i for i,actor in enumerate(h['actors']) if eligible(actor,'train',source) or eligible(actor,'validation',source)]
            if not ids:continue
            valid=sig[ids,:,11]>.5
            if not valid.any():continue
            file=Path(entry['file']).name+'.npz';old=previous/f'features-{index}'/file
            with np.load(target/lookup[entry['file']]) as z:
                assert z['actors'].tolist()==h['actors'];assert np.array_equal(valid,z['valid'][ids])
                y=z['returns'][ids][valid];post=z['post0'][ids][valid]
            validation=np.broadcast_to(np.array([h['actors'][i]//2%8==6 for i in ids])[:,None],valid.shape)[valid]
            with np.load(old) as z:
                assert z['h'].shape[0]==len(y);assert np.array_equal(z['post'],post);assert np.array_equal(z['validation'],validation)
                np.savez(dest/file,h=z['h'],y=y,post=post,validation=validation)
            frames+=len(y);count+=1
            if count%32==0:save(out/'cache-status.json',dict(dataset=name,files=count,frames=frames))
        rows.append(dict(dataset=name,files=count,frames=frames,parity=proof,targets_sha256=sha(target/'targets.json')))
    save(out/'cache-reuse.json',dict(actor_sha256=config['actor_sha256'],source_config_sha256=sha(previous/'config.json'),source=str(previous),datasets=rows,reason='Identical observations/signals/actor; cached frozen recurrent features reused byte-for-byte, all MC targets replaced. Value head is retrained.'))

if __name__=='__main__':main()
