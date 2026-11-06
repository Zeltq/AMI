"""CPU smoke: checkpoint resume must match uninterrupted recurrent IL exactly."""
import json,shutil
from pathlib import Path
import torch
from train_chio import run
ROOT=Path(__file__).resolve().parents[2];OUT=Path('/mnt/c/AMI/runs/p16/python-resume-check');OUT.mkdir(exist_ok=False)
source=Path('/mnt/c/AMI/runs/p15/collect-008-aim');dataset=OUT/'dataset';dataset.mkdir()
m=json.loads((source/'dataset.json').read_text());m['files']=m['files'][:1];m['partial']=True
for entry in m['files']:shutil.copyfile(source/entry['file'],dataset/entry['file'])
(dataset/'dataset.json').write_text(json.dumps(m))
cfg=json.loads((ROOT/'reports/P16/configs/imitation.json').read_text());cfg.update(dataset=str(dataset),device='cpu',batch_sequences=1,checkpoint_updates=1)
config=OUT/'config.json';config.write_text(json.dumps(cfg))
run(config,OUT/'full',max_updates=3)
run(config,OUT/'resumed',max_updates=2)
run(config,OUT/'resumed',max_updates=3,resume=str(OUT/'resumed/checkpoint.pt'))
a=torch.load(OUT/'full/checkpoint.pt',weights_only=False,map_location='cpu');b=torch.load(OUT/'resumed/checkpoint.pt',weights_only=False,map_location='cpu')
equal=sum(torch.equal(v,b['state_dict'][k]) for k,v in a['state_dict'].items());assert equal==len(a['state_dict'])
assert a['optimizer_steps']==b['optimizer_steps']==3
result=dict(passed=True,equal_tensors=equal,total_tensors=len(a['state_dict']),updates=3,history_equal=[r['loss'] for r in a['history']]==[r['loss'] for r in b['history']])
assert result['history_equal'];(OUT/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
