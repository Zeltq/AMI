"""Create an exclusive diagnostic ZIP; never remove or modify source artifacts."""
import argparse,hashlib,json,os,zipfile
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--output',required=True);a=p.parse_args()
 run=Path(a.run).resolve();output=Path(a.output).resolve();output.parent.mkdir(parents=True,exist_ok=True)
 if output.exists():raise FileExistsError(output)
 temporary=output.with_suffix('.partial.zip');entries=[]
 sources=[(file,'pilot/'+file.relative_to(run).as_posix())for file in sorted(run.rglob('*'))if file.is_file()]
 baseline=ROOT/'runs/gpu/mgia-series-001/baseline'
 sources += [(file,'baseline/'+file.relative_to(baseline).as_posix())for file in sorted(baseline.rglob('*'))if file.is_file()]
 source_names=['src/sf_duel/model.py','src/sf_duel/observations.py','src/sf_duel/native_data.py','src/sf_duel/native_policy_server.py','src/sf_duel/spec.py','scripts/selfplay_cycle.py','scripts/run_native_teacher.py','gpu/scripts/diagnose_native_start.py','gpu/scripts/behavior_report.py','gpu/scripts/pilot_run.py','gpu/scripts/evaluate_dota.py']
 sources += [(ROOT/name,'diagnostic-source/'+name)for name in source_names]
 with zipfile.ZipFile(temporary,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True)as archive:
  for index,(source,name)in enumerate(sources):
   sha=hashlib.sha256();info=zipfile.ZipInfo.from_file(source,name);info.compress_type=zipfile.ZIP_STORED if source.suffix.lower()in ('.gz','.dem','.zip')else zipfile.ZIP_DEFLATED
   with source.open('rb')as inp,archive.open(info,'w',force_zip64=True)as out:
    while chunk:=inp.read(1024*1024):sha.update(chunk);out.write(chunk)
   entries.append({'path':name,'bytes':source.stat().st_size,'sha256':sha.hexdigest()})
   if index%100==0:print(f'Archived{index+1}/{len(sources)}',flush=True)
  archive.writestr('ARCHIVE-MANIFEST.json',json.dumps({'created_utc':datetime.now(timezone.utc).isoformat(),'diagnostic_only':True,'prohibited_use':'This pilot checkpoint must not initialize future training. Baseline is original immutable MIA2.','source_run':str(run),'files':entries},indent=2))
 with zipfile.ZipFile(temporary)as archive:
  bad=archive.testzip()
  if bad is not None:raise ValueError('Archive CRC failure:'+bad)
  critical=[entry for entry in entries if entry['path'].startswith('pilot/frozen-checkpoint/')or entry['path']in ('pilot/pilot-checkpoint.pt','pilot/pilot-summary.json','baseline/MIA2-original.pt')]
  for entry in critical:
   if hashlib.sha256(archive.read(entry['path'])).hexdigest()!=entry['sha256']:raise ValueError('Archive SHA failure:'+entry['path'])
 if output.exists():raise FileExistsError(output)
 os.rename(temporary,output)
 sha=hashlib.sha256()
 with output.open('rb')as stream:
  while chunk:=stream.read(1024*1024):sha.update(chunk)
 pointer={'archive':str(output),'sha256':sha.hexdigest(),'bytes':output.stat().st_size,'files':len(entries),'crc_verified_all_files':True,'critical_sha_verified':True,'diagnostic_only':True,'training_initialization_prohibited':True,'source_run':str(run)}
 folder=ROOT/'runs/gpu/diagnostic-archive';folder.mkdir(exist_ok=True)
 (folder/(run.name+'.json')).write_text(json.dumps(pointer,indent=2)+'\n',encoding='utf-8');print(json.dumps(pointer),flush=True)
if __name__=='__main__':main()
