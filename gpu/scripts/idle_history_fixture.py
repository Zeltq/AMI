"""Export selected complete reset-aligned blocks from the recorded idle episode."""
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from sf_duel.native_data import examples
from sf_duel.observations import numpy_batch

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--output',required=True);p.add_argument('--team',type=int,default=2);a=p.parse_args()
 starts=(641,769,2817);wanted={i for start in starts for i in range(start,start+128)};selected=[]
 for row in examples(a.run):
  if row['team']==a.team and row['sequence']in wanted:selected.append(row)
  if len(selected)==len(wanted):break
 assert len(selected)==384 and {row['sequence']for row in selected}==wanted
 # These blocks precede the first movement, avoiding any target-index remap.
 assert all(row['action']['action']==0 for row in selected)
 arrays=numpy_batch([row['encoded'].arrays for row in selected]);arrays={k:v.astype(np.float16).astype(np.float32)if np.issubdtype(v.dtype,np.floating)else v for k,v in arrays.items()}
 arrays['labels']=np.asarray([[row['action'][k]for k in ('action','buy','item','target','x','y')]for row in selected],dtype=np.int64)
 output=Path(a.output);output.parent.mkdir(parents=True,exist_ok=True);np.savez(output,**arrays)
 metadata={'source':str(Path(a.run).resolve()),'team':a.team,'steps':384,'block_starts':starts,'sequence':[row['sequence']for row in selected],'times':[row['encoded'].snapshot['time']for row in selected],'rounding':'float32->float16->float32; bool masks unchanged','memory_reset':'Each selected block starts at sequence1 mod128; previous actual action is preserved across periodic memory resets, matching native contract. First two blocks are contiguous.','scope':'Recorded observations and actual previous actions, not a closed-loop counterfactual game or simulation-state parity.'}
 output.with_suffix('.json').write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf-8');print(json.dumps({k:v for k,v in metadata.items()if k not in ('sequence','times')}))
if __name__=='__main__':main()
