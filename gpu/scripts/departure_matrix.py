"""Four predeclared short closed-loop comparisons; no training or model selection."""
import argparse,hashlib,json,os,shutil,subprocess,time
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def write(path,value):
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,ensure_ascii=True)+'\n',encoding='utf-8');tmp.replace(path)
def main():
 p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--binary',required=True);a=p.parse_args();os.chdir(ROOT)
 run=Path(a.run).resolve();run.mkdir(parents=True,exist_ok=False);exe=run/'sf-duel-gpu.exe';shutil.copy2(a.binary,exe)
 pilot=ROOT/'runs/gpu/pilot-mia2-25m-001';base=ROOT/'runs/gpu/mgia-series-001/baseline';baseline=base/'burn-weights';failed=pilot/'frozen-checkpoint/model.bpk'
 identities={str(path):hashlib.sha256(path.read_bytes()).hexdigest()for path in (base/'MIA2-original.pt',failed)}
 template=json.loads((pilot/'training-config.json').read_text());template.update(matches=2,seed=20261221,opponents=[str(baseline)])
 results=[]
 write(run/'question.json',{'question':'Does argmax cause AFK/stuck relative to sampling under identical frozen models/memory inGPU?','matrix':['MIA2-MIA2 argmax','MIA2-MIA2 sampling','failed-pilot-MIA2 argmax','failed-pilot-MIA2 sampling'],'matches_each':2,'model_sides':[2,3],'stop':'Both agents reach mid+10s, early true terminal, or90s game time. Not a strength evaluation.','training_updates':0,'source_weights':identities})
 for policy,checkpoint in [('baseline',baseline),('failed-pilot',failed)]:
  config=dict(template,initial_checkpoint=str(checkpoint));config_path=run/(policy+'-config.json');write(config_path,config)
  for selection in ('argmax','sampling'):
   label=policy+'-'+selection;output=run/label;argv=[str(exe),'check-departure',str(config_path),str(output)]+(['--argmax']if selection=='argmax'else [])
   with (run/(label+'.stdout.log')).open('w')as out,(run/(label+'.stderr.log')).open('w')as err:
    child=subprocess.Popen(argv,cwd=ROOT,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
    write(run/'status.json',{'phase':'checking','pid':os.getpid(),'child_pid':child.pid,'case':label,'command':argv,'completed':len(results),'total':4,'updated_utc':datetime.now(timezone.utc).isoformat()})
    deadline=time.monotonic()+180
    while child.poll()is None:
     if time.monotonic()>deadline:
      child.terminate();child.wait(timeout=15);write(run/'status.json',{'phase':'failed','case':label,'reason':'180second wall cap; do not infer behavior from incomplete prefix'});raise TimeoutError(label)
     time.sleep(2)
    if child.returncode:
     write(run/'status.json',{'phase':'failed','case':label,'exit_code':child.returncode});raise RuntimeError(f'{label} failed; inspect logs')
   report=json.loads((output/'departure.json').read_text());results.append({'policy':policy,'selection':selection,'report':str(output/'departure.json'),'matches':report['matches']});write(run/'partial-results.json',results)
 if any(hashlib.sha256(Path(path).read_bytes()).hexdigest()!=sha for path,sha in identities.items()):raise RuntimeError('Frozen weights changed')
 write(run/'results.json',{'complete':True,'training_updates':0,'cases':results,'source_weights_unchanged':True});write(run/'status.json',{'phase':'complete','pid':os.getpid(),'child_pid':None,'completed':4,'total':4,'training_updates':0});print(json.dumps({'complete':True,'results':str(run/'results.json')}),flush=True)
if __name__=='__main__':main()
