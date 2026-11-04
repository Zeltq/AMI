"""Repeat P07's real-cast onset/effect alignment on P10's fresh capture."""
from pathlib import Path
import shutil
import p07_results

RUN=Path('C:/AMI/runs/p10')
if __name__=='__main__':
    p07_results.OUT=RUN/'cast-parity';p07_results.OUT.mkdir(exist_ok=True)
    shutil.copy2(RUN/'gpu-spell-parity.json',p07_results.OUT/'gpu-parity.json')
    p07_results.actual_parity(RUN/'spell-probe/dota.log')
    shutil.copy2(p07_results.OUT/'actual-cast-parity.json',p07_results.ROOT/'reports/P10/actual-cast-parity.json')
