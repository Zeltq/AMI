"""Reuse the P02 trace counters with the actual P08 model and batch labels."""
import argparse
import json
from pathlib import Path
from p02_trace_analysis import analyze

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('trace_directory',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--model',required=True)
    args=parser.parse_args()
    result=analyze(args.trace_directory)
    result['model']=args.model
    result['limitations'][-1]='The first three matches of the same 64-match sampling evaluation are analyzed; this is a small diagnostic subset.'
    args.output.write_bytes((json.dumps(result,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))
    print(json.dumps(result['totals']))
