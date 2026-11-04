"""Repeat the P07 public spell capture in P10's isolated dedicated server."""
from pathlib import Path
import p07_probe
from sf_duel.live.install import isolated_content

p07_probe.runner.isolated_content=lambda root,source: isolated_content(
    Path('C:/AMI/runs/p10/spell-runtime')/Path(root).name,source)
# P10's job supervisor owns the shared status; retain the probe's local status.
p07_probe.runner.save=p07_probe.save

if __name__=='__main__':
    p07_probe.runner.main()
