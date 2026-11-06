import json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'gpu/scripts'))
import p16_run as runner

class P16Supervisor(unittest.TestCase):
    def test_live_status_retries_partial_json_but_rejects_persistent_corruption(self):
        with patch.object(Path,'exists',return_value=True),patch.object(Path,'read_text',side_effect=['','{"step":','{"step":42}']),patch.object(runner.time,'sleep') as sleep:
            self.assertEqual(runner.read('status.json'),{'step':42});self.assertEqual(sleep.call_count,2)
        with patch.object(Path,'exists',return_value=True),patch.object(Path,'read_text',return_value='{'),patch.object(runner.time,'sleep') as sleep:
            with self.assertRaises(json.JSONDecodeError):runner.read('status.json')
            self.assertEqual(sleep.call_count,19)

    def test_launch_adopts_existing_job_without_spawning(self):
        with patch.object(Path,'exists',return_value=True),patch.object(runner.subprocess,'Popen') as spawn:
            runner.launch('nav-imitation-w',['trainer'])
            spawn.assert_not_called()

    def test_new_sampling_evaluation_does_not_reuse_argmax(self):
        archived=[dict(label='nav-imitation',opponent='v10',complete=True)]
        captured=[]
        def config(name,value):
            captured.append((name,value));raise LookupError('captured config')
        with patch.object(runner,'read',return_value=archived),patch.object(runner,'current_engine',return_value='engine'),patch.object(runner,'config',side_effect=config):
            with self.assertRaisesRegex(LookupError,'captured config'):runner.evaluation('nav-imitation','weights','v10')
        name,cfg=captured[0]
        self.assertEqual(name,'eval-nav-imitation-v10-sampling');self.assertTrue(cfg['sampling'])
        self.assertEqual((cfg['matches'],cfg['seed']),(128,20261103))

    def test_sampling_start_rule_ties_and_low_farm_are_not_stop_conditions(self):
        from p16_sampling_pipeline import choose_start
        def summary(farm):return {'controllers':{'CHiO':{'lh':{'numerator':farm},'deny':{'numerator':0},'actors':128}}}
        weights=[1,1,4,4,4,4,4,2.93,4,1,1,1,1]
        for left,right,label in [(1,1,'nav-imitation-w'),(3,2,'nav-imitation'),(2,3,'nav-imitation-w')]:
            d=choose_start(summary(left),summary(right),weights)
            self.assertEqual(d['start'],label);self.assertTrue(d['below_three'])
            self.assertEqual(d['dagger_class_weights'][4],1)
            self.assertEqual(d['dagger_class_weights'][2],4 if label.endswith('-w') else 1)
        self.assertEqual(weights[4],4)

    def test_navigation_gate_requires_a_strict_majority(self):
        for count,action in [(65,'continue-policy'),(64,'scripted-navigation-restart')]:
            writes=[]
            def read(path,default=None):
                if Path(path).name=='eval-dagger-1-v10.json':return {'controllers':{'CHiO':{'mid_arrivals':{'numerator':count,'denominator':128}}}}
                return {'classification':'distribution-shift'}
            with patch.object(runner,'read',side_effect=read),patch.object(runner,'save',side_effect=lambda p,v:writes.append(v)),patch.object(runner,'publish'),patch.object(runner,'state',side_effect=RuntimeError('gate reached')):
                if count==65:runner.navigation_gate()
                else:
                    with self.assertRaisesRegex(RuntimeError,'gate reached'):runner.navigation_gate()
            self.assertEqual(writes[0]['action'],action)

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
