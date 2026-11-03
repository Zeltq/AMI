"""Read-only per-tick Bot API timing probe using the existing isolated Dota runner."""
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('p03_measure', ROOT / 'reports/P03/measure_dota.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
# Steam updates can replace files underlying old worker hardlinks. Give this
# probe its own fresh runtime instead of repairing another task's worker.
isolate = runner.isolated_content
runner.isolated_content = lambda root, source: isolate(Path('C:/AMI/runs/p06/workers') / Path(root).name, source)
probe = (ROOT / 'reports/P06/timing_probe.lua').read_text(encoding='utf-8')
runner.CALLBACK = 'local timing_probe=(function()\n' + probe + '\nend)()\n' + runner.CALLBACK
runner.CALLBACK = runner.CALLBACK.replace('local agent=require("bots/agent")', 'settings.observation_profile="ggd-v3"\nlocal agent=require("bots/agent")')
runner.CALLBACK = runner.CALLBACK.replace('  controller.Think()', '  timing_probe.tick(bot)\n  controller.Think()')

# Install the state-transition observer while still in hero selection.
class ProbeGame(runner.Game):
    def wait_log(self, needle, timeout=100):
        if needle == 'DOTA_GAMERULES_STATE_PRE_GAME':
            self.command('script_reload_code bots/p03_events')
        return super().wait_log(needle, timeout)

runner.Game = ProbeGame
if '--tower-scenario' in sys.argv:
    sys.argv.remove('--tower-scenario')
    runner.CALLBACK = runner.CALLBACK.replace('  controller.Think()',
        '  bot:Action_MoveToLocation(GetTower(GetOpposingTeam(),TOWER_MID_1):GetLocation())')

if __name__ == '__main__':
    runner.main()
