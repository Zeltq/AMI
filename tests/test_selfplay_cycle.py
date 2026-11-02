import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch
from contextlib import nullcontext
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from sf_duel.selfplay.config import Config

spec=importlib.util.spec_from_file_location('selfplay_cycle',Path(__file__).resolve().parents[1]/'scripts/selfplay_cycle.py')
cycle=importlib.util.module_from_spec(spec);spec.loader.exec_module(cycle)


class CycleTests(unittest.TestCase):
    def test_cpu_service_can_launch_locally_with_native_paths_and_records_host(self):
        import json,sys
        for host in ('wsl','windows'):
            with self.subTest(host=host),TemporaryDirectory()as temp:
                root=Path(temp);service=root/'service'
                def launch(command,**kwargs):
                    (service/'ready.json').write_text('{}');(service/'stats.json').write_text('{"errors":[]}')
                    return SimpleNamespace(pid=123,poll=lambda:None,wait=lambda timeout:0)
                with patch.object(cycle.subprocess,'Popen',side_effect=launch)as process:
                    with cycle.policy_service(root/'current.pt',root,Config(workers=2),[root/'old.pt'],record=False,inference_device='cpu',selection='argmax',reset_origin=-3.3,inference_host=host):pass
                command=process.call_args.args[0];record=json.loads((service/'launch.json').read_text())
                self.assertEqual(record['inference_host'],host);self.assertEqual(record['pid'],123)
                self.assertIn('--reset-origin',command);self.assertNotIn('--stochastic',command)
                self.assertTrue((service/'stop').exists())
                if host=='windows':
                    self.assertEqual(command[0],sys.executable)
                    self.assertEqual(command[command.index('--checkpoint')+1],str(root/'current.pt'))
                    self.assertEqual(command[command.index('--opponent')+1],str(root/'old.pt'))
                    self.assertIn(str(cycle.ROOT/'src'),process.call_args.kwargs['env']['PYTHONPATH'])
                else:self.assertEqual(command[0],'wsl')

    def test_windows_host_rejects_cuda_before_launch(self):
        with TemporaryDirectory()as temp,patch.object(cycle.subprocess,'Popen')as process:
            with self.assertRaisesRegex(ValueError,'Windows CPU'):
                with cycle.policy_service(Path(temp)/'model.pt',Path(temp),Config(),[],inference_host='windows'):pass
            process.assert_not_called()

    def test_evaluation_refill_preserves_missing_indices_and_sides(self):
        def play(base,index,slot,cfg,routing,until,teacher,demo):
            side=2 if index%2 else 3
            self.assertEqual(routing[str(side)],'current')
            self.assertEqual(routing[str(5-side)],'0')
            return {'index':index,'model_team':side,'outcome':{'winner':side}}
        with TemporaryDirectory() as temp:
            root=Path(temp);(root/'workers').mkdir()
            with patch.object(cycle,'ROOT',root), patch.object(cycle,'policy_service',return_value=nullcontext()), patch.object(cycle,'play',side_effect=play),patch.object(cycle.shutil,'disk_usage',return_value=SimpleNamespace(free=64*2**30)):
                games=cycle.collect(root/'model.pt',root/'eval',Config(workers=2),['old.pt'],matches=3,versus_previous=True,match_indices=[2,5,6])
            self.assertEqual(sorted((m['index'],m['model_team']) for m in games),[(2,3),(5,2),(6,3)])

    def test_low_disk_still_prevents_refilling(self):
        def play(base,index,slot,cfg,routing,until,teacher,demo):
            return {'index':index,'model_team':2 if index%2 else 3,'outcome':{'winner':2}}
        with TemporaryDirectory()as temp:
            root=Path(temp);(root/'workers').mkdir()
            with patch.object(cycle,'ROOT',root),patch.object(cycle,'policy_service',return_value=nullcontext()),patch.object(cycle,'play',side_effect=play),patch.object(cycle.shutil,'disk_usage',return_value=SimpleNamespace(free=19*2**30)):
                games=cycle.collect(root/'model.pt',root/'eval',Config(workers=2),['old.pt'],matches=3,versus_previous=True)
            self.assertEqual(sorted(m['index']for m in games),[1,2])

    def test_archive_and_teacher_schedule(self):
        cfg=Config()
        self.assertEqual(cycle.archive_paths({},[],'initial'),['initial'])
        self.assertEqual(cycle.archive_paths({'archive':['old0','old1']},[],'current'),['old0','old1'])
        self.assertEqual(cycle.archive_paths({'archive':['old0']},['selected'],'current'),['selected'])
        self.assertTrue(cycle.teacher_required(1,4,cfg))
        self.assertFalse(cycle.teacher_required(2,4,cfg))
        self.assertTrue(cycle.teacher_required(2,5,cfg))

    def test_score_respects_sides_and_draw_half_credit(self):
        matches=[]
        for i in range(10):
            team=2 if i%2==0 else 3
            matches.append({'model_team':team,'outcome':{'winner':team if i<8 else None if i==8 else 5-team}})
        score=cycle.evaluation_score(matches)
        self.assertEqual((score['wins'],score['draws'],score['losses']),(8,1,1))
        self.assertAlmostEqual(score['score'],.85)
