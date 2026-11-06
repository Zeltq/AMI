import json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'gpu/scripts'))
import p16_run as runner

class P16Supervisor(unittest.TestCase):
    def test_windows_paths_for_wsl(self):
        self.assertEqual(runner.linux('C:/AMI/runs/p16/test'),'/mnt/c/AMI/runs/p16/test')
        self.assertEqual(runner.linux(Path('C:/AMI-wt/P16')),'/mnt/c/AMI-wt/P16')

    def test_sequential_evaluation_waits_for_native_exit_then_resumes_same_stage(self):
        base=Path('C:/AMI/runs/p16');base.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base) as temporary:
            root=Path(temporary);run=root/'run';run.mkdir();reports=root/'reports';reports.mkdir();calls=[];evaluations=[];completed=[];live=False
            def job(name,cmd,artifact,on_tick):
                nonlocal live
                folder=artifact.parent;folder.mkdir(exist_ok=True);checkpoint=folder/'checkpoint-000001'
                runner.save(folder/'latest.json',{'path':str(checkpoint)})
                cfg=json.loads(Path(cmd[2]).read_text());calls.append(cfg);live=True
                seconds=7300 if len(calls)==1 else 14401
                on_tick({'training_seconds':seconds})
                if len(calls)==1:self.assertTrue((folder/'stop.json').exists())
                live=False
                return {'checkpoint':str(checkpoint),'training_seconds':seconds,'blocks':len(calls)}
            def evaluate(*args):self.assertFalse(live);evaluations.append(args)
            def finish(name,result,folder):completed.append(result)
            with patch.multiple(runner,RUN=run,REPORT=reports),patch.object(runner,'cached',return_value=None),patch.object(runner,'run_job',side_effect=job),patch.object(runner,'queue_evaluation',side_effect=evaluate),patch.object(runner,'done',side_effect=finish):
                runner.native_training('self-play',{'minutes':240,'schedule_minutes':240,'initial_weights':'fixed-anchor'},parallel=False)
            self.assertEqual(len(calls),2);self.assertEqual(len(evaluations),1);self.assertEqual(completed[0]['training_seconds'],14401)
            self.assertEqual(calls[0]['initial_weights'],calls[1]['initial_weights']);self.assertEqual(calls[1]['schedule_minutes'],240)
            self.assertIn('resume_checkpoint',calls[1])

if __name__=='__main__':unittest.main()
