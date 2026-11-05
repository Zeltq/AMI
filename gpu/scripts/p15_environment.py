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


if __name__=='__main__':
    result=create()
    path=Path(__file__).resolve().parents[2]/'reports/P15/environment.json'
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(result))
