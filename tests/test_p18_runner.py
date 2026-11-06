import datetime
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import p18_run as run

class Recovery(unittest.TestCase):
    def test_interval_resume_preserves_anchor_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);run.save(root/'plan.json',{'deadline_utc':(datetime.datetime.now(datetime.timezone.utc)+datetime.timedelta(hours=1)).isoformat()})
            configs=[];evals=[];done=[]
            def job(name,cmd,artifact,on_tick):
                folder=artifact.parent;folder.mkdir(exist_ok=True)
                if len(configs)==1:
                    on_tick({'training_seconds':7201})
                    return {'training_seconds':7201,'checkpoint':'checkpoint-one'}
                return {'training_seconds':14401,'checkpoint':'checkpoint-two'}
            def config(name,cfg):configs.append(dict(cfg));return root/'config.json'
            def cached(name):return done[-1] if done else None
            with patch.multiple(run,RUN=root,config=config,run_job=job,cached=cached,queue_evaluation=lambda *a:evals.append(a),done=lambda *a:done.append(a)):
                cfg={'minutes':240,'initial_weights':'anchor','schedule_minutes':240}
                run.native_training('self-play',cfg)
            self.assertEqual(len(configs),2);self.assertEqual(configs[1]['resume_checkpoint'],'checkpoint-one')
            self.assertEqual(configs[1]['initial_weights'],'anchor');self.assertEqual(configs[1]['schedule_minutes'],240)
            self.assertEqual(evals[0][0],'self-play-120m');self.assertNotIn('resume_checkpoint',cfg)

if __name__=='__main__':unittest.main()
