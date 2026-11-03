"""P07 controlled Dota scenario; Bot API samples are separate from server truth."""
import importlib.util
import json
import sys
import time
from p07_status import publish
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('p03_measure', ROOT / 'reports/P03/measure_dota.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
isolate = runner.isolated_content
runner.isolated_content = lambda root, source: isolate(Path('C:/AMI/runs/p07/workers') / Path(root).name, source)
runner.CALLBACK = (ROOT / 'reports/P07/bot_probe.lua').read_text(encoding='utf-8')
if '--miss' in sys.argv:
    sys.argv.remove('--miss')
    runner.CALLBACK = '''local json=require('game/dkjson')
local last=nil
function Think()
 local second=math.floor(DotaTime())
 if second~=last then last=second;print('SF_P03_HERO|'..json.encode({time=DotaTime(),team=GetTeam()})) end
end
'''
save = runner.save
last_publish=0.
def save_status(path, value):
    global last_publish
    save(path, value)
    if path.name == 'status.json' and (time.monotonic()-last_publish>=10 or not value.get('background_work_remaining')):
        publish(path.parent.name,value);last_publish=time.monotonic()
runner.save = save_status
scenario_flags=[]
scenario_kind=None
if '--case-kind' in sys.argv:
    at=sys.argv.index('--case-kind');scenario_kind=sys.argv[at+1];del sys.argv[at:at+2]
    if scenario_kind not in ('sf','melee','ranged','tower'):raise ValueError('unsupported fixture kind')
for flag,name in [('--impact','SF_P07_IMPACT'),('--nonhero','SF_P07_NONHERO'),('--tower-target','SF_P07_TOWER_TARGET')]:
    if flag in sys.argv:sys.argv.remove(flag);scenario_flags.append(name)
if scenario_flags or scenario_kind:
    class ImpactGame(runner.Game):
        def __enter__(self):
            script=self.bot_root/'p03_events.lua'
            prefix=''.join('_G.'+name+'=true\n' for name in scenario_flags)
            if scenario_kind:prefix+='_G.SF_P07_KIND="'+scenario_kind+'"\n'
            script.write_text(prefix+script.read_text(encoding='utf-8'),encoding='utf-8')
            return super().__enter__()
    runner.Game=ImpactGame

if __name__ == '__main__':
    runner.main()
