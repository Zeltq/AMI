"""Copy immutable 005 assets and create the separate CHiO environment."""
import hashlib
import json
from pathlib import Path
import shutil

SOURCE=Path('C:/AMI/runs/p07/dota-calibration-005')
TARGET=Path('C:/AMI/runs/p15/dota-calibration-005-chio')
ASSETS=('geometry','navigation','movement_geometry','fow_blockers','static_vision')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def create(source=SOURCE,target=TARGET):
    old=json.loads((source/'engine.json').read_text(encoding='utf-8'))
    target.mkdir(parents=True,exist_ok=False)
    new=dict(old,calibration=target.as_posix(),decision_ticks=6,chio_direct=True)
    assets={}
    for key in ASSETS:
        if old.get(key):
            src=source/old[key];dst=target/old[key]
            dst.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(src,dst)
            assets[key]=dict(file=old[key],source_sha256=sha(src),copy_sha256=sha(dst))
            assert assets[key]['source_sha256']==assets[key]['copy_sha256']
    changed={key for key in new if new[key]!=old.get(key)}
    assert changed=={'calibration','decision_ticks','chio_direct'}
    (target/'engine.json').write_text(json.dumps(new,indent=2)+'\n',encoding='utf-8')
    tables={key:hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
            for key,value in old.items() if key not in changed}
    return dict(source=str(source),target=str(target),changed_keys=sorted(changed),assets=assets,
                identical_table_sha256=tables,source_engine_sha256=sha(source/'engine.json'),
                target_engine_sha256=sha(target/'engine.json'))


def verify(source=SOURCE,target=TARGET):
    old=json.loads((source/'engine.json').read_text(encoding='utf-8'));new=json.loads((target/'engine.json').read_text(encoding='utf-8'))
    assert new['decision_ticks']==6 and new['offset_ticks']==6 and new['chio_direct']
    unchanged={k:hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest() for k,v in old.items() if k not in ('calibration','decision_ticks')}
    assert all(old[k]==new[k] for k in unchanged)
    original={k:dict(source_sha256=sha(source/old[k]),copy_sha256=sha(target/new[k])) for k in ASSETS if old.get(k)}
    assert all(v['source_sha256']==v['copy_sha256'] for v in original.values())
    additions={k:dict(file=new[k],sha256=sha(target/new[k]),bytes=(target/new[k]).stat().st_size) for k in ('chio_ground','chio_teacher_navigation','chio_live_grid') if new.get(k)}
    return dict(source=str(source),target=str(target),identical_table_sha256=unchanged,assets=original,chio_only_assets=additions,source_engine_sha256=sha(source/'engine.json'),target_engine_sha256=sha(target/'engine.json'))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--verify',action='store_true');p.add_argument('--install-ground',type=Path);a=p.parse_args()
    if a.install_ground:
        assert a.install_ground.stat().st_size==512*512*4
        shutil.copy2(a.install_ground,TARGET/'chio-ground32.f32');path=TARGET/'engine.json';engine=json.loads(path.read_text(encoding='utf-8'));assert engine['chio_direct']
        engine['chio_ground']='chio-ground32.f32';path.write_text(json.dumps(engine,indent=2)+'\n',encoding='utf-8',newline='\n')
    result=verify() if a.verify or a.install_ground else create()
    path=Path(__file__).resolve().parents[2]/'reports/P15/environment.json';path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n');print(json.dumps(result))
