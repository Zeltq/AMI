"""A pool of isolated native Dota servers; each match has a separate Bot VM."""
import argparse
import os
for variable in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[variable]='1'
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import json
from pathlib import Path
import random
import multiprocessing
import msvcrt
from queue import Empty
import shutil
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from collect_teacher import collect
from prepare_teacher import prepare
from sf_duel.live.install import isolated_content


def worker(slot,base,source,args,jobs,stop):
    base=Path(base);content,bots=isolated_content(ROOT/'workers'/f'{slot:02d}',source)
    result=[]
    while not stop.is_set():
        try:index=jobs.get_nowait()
        except Empty:break
        run=base/f'match-{index:04d}'
        try:
            item=collect(run,bots,args,index,content_root=content,game_port=27155+slot*10,stop_event=stop)
        except BaseException:
            stop.set();raise
        # Keep lossless engine evidence without retaining hundreds of GB of text.
        with (run/'dota.log').open('rb') as src,gzip.open(run/'dota.log.gz','wb',compresslevel=1) as dst:
            shutil.copyfileobj(src,dst,1024*1024)
        (run/'dota.log').unlink()
        item.update(index=index,run=str(run))
        result.append(item)
        status=base/f'worker-{slot:02d}.json'
        temp=status.with_suffix('.tmp')
        temp.write_text(json.dumps({'slot':slot,'completed':len(result),'last_index':index,
                                   'eligible':sum(r['eligible'] for r in result)},indent=2))
        temp.replace(status)
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True)
    parser.add_argument('--matches',type=int,required=True);parser.add_argument('--workers',type=int,required=True)
    parser.add_argument('--hz',type=int,default=15);parser.add_argument('--timescale',type=float,default=50)
    parser.add_argument('--until',type=float,default=600);parser.add_argument('--wall-timeout',type=float,default=600)
    parser.add_argument('--validation-matches',type=int,default=64);parser.add_argument('--replay-prefix')
    parser.add_argument('--engine-threads',type=int)
    parser.add_argument('--recorder',choices=('python','go'),default='python')
    parser.add_argument('--record-first',action='store_true');args=parser.parse_args()
    if args.hz<=0 or 30%args.hz or not 1<=args.workers<=min(20,args.matches):
        parser.error('hz must divide 30; pool size must be 1..20 and not exceed matches')
    base=Path(args.run).resolve()
    if base.exists():parser.error('Use a new directory')
    base.mkdir(parents=True);source=prepare()
    snapshot=base/'teacher_bots';shutil.copytree(source,snapshot)
    if args.recorder=='go':
        collector=base/'collector';collector.mkdir()
        for path in (ROOT/'tools/nativecollect').glob('*.go'):shutil.copy2(path,collector/path.name)
        for name in ('go.mod','go.sum','nativecollect.exe'):shutil.copy2(ROOT/'tools/nativecollect'/name,collector/name)
        args.recorder_exe=str(collector/'nativecollect.exe')
    started=time.monotonic();results=[]
    (ROOT/'workers').mkdir(exist_ok=True)
    with (ROOT/'workers/pool.lock').open('a+b') as lock,multiprocessing.Manager() as manager:
      lock.seek(0);lock.write(b'1');lock.flush();lock.seek(0)
      msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
      jobs=manager.Queue();stop=manager.Event()
      for index in range(1,args.matches+1):jobs.put(index)
      with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(worker,slot,str(base),str(snapshot),args,jobs,stop) for slot in range(args.workers)]
        while not all(f.done() for f in futures):
            for f in futures:
                if f.done() and f.exception():
                    print(json.dumps({'worker_error':str(f.exception())}),flush=True)
                    raise f.exception()
            count=sum(json.loads(p.read_text())['completed'] for p in base.glob('worker-*.json'))
            print(json.dumps({'elapsed_seconds':round(time.monotonic()-started,1),
                              'completed':count,'target':args.matches,'workers':args.workers}),flush=True)
            time.sleep(10)
        for future in futures:results.extend(future.result())
    results.sort(key=lambda r:r['index'])
    valid=[r['index'] for r in results if r['eligible']]
    validation=set(random.Random(20261030).sample(valid,min(len(valid),args.validation_matches)))
    elapsed=time.monotonic()-started
    for r in results:r['split']='validation' if r['index'] in validation else 'train'
    dataset={'matches':results,'eligible_matches':len(valid),'validation_matches':len(validation),
        'train_matches':len(valid)-len(validation),'split_seed':20261030,'split_unit':'whole match, both perspectives',
        'wall_seconds':elapsed,'matches_per_hour':len(valid)/elapsed*3600,'training_started':False,
        'teacher_manifest_sha256':hashlib.sha256((snapshot/'ADAPTER.json').read_bytes()).hexdigest()}
    if args.recorder=='go':dataset['collector_sha256']=hashlib.sha256(Path(args.recorder_exe).read_bytes()).hexdigest()
    (base/'dataset.json').write_text(json.dumps(dataset,indent=2))
    print(json.dumps({k:v for k,v in dataset.items() if k!='matches'}),flush=True)

if __name__=='__main__':main()
