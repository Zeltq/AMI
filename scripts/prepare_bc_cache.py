import argparse,json,os,sys,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from sf_duel.bc_data import prepare_match

def main():
 p=argparse.ArgumentParser();p.add_argument('dataset');p.add_argument('--output',required=True);p.add_argument('--watch',action='store_true');p.add_argument('--workers',type=int,default=2);a=p.parse_args();base=Path(a.dataset);dest=Path(a.output);dest.mkdir(exist_ok=True,parents=True)
 pending={};done={int(x.stem.split('-')[1]) for x in dest.glob('match-*.json')}
 with ProcessPoolExecutor(max_workers=a.workers) as pool:
  while True:
   for path in base.glob('match-*/result.json'):
    r=json.loads(path.read_text());i=r['index']
    if r.get('eligible') and i not in done and i not in pending:pending[i]=pool.submit(prepare_match,r['run'],str(dest),i)
   for i,f in list(pending.items()):
    if f.done():f.result();done.add(i);del pending[i];print(json.dumps({'prepared_matches':len(done),'last_match':i}),flush=True)
   manifest=base/'dataset.json'
   if manifest.exists():
    d=json.loads(manifest.read_text())
    if len(done)==d['eligible_matches'] and not pending:
     d['cache']=str(dest);(dest/'dataset.json').write_text(json.dumps(d,indent=2));return
   if not a.watch and not pending:return
   time.sleep(3)
if __name__=='__main__':main()
