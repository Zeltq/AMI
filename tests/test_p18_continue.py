import datetime,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import p18_continue as c

class Continuation(unittest.TestCase):
    def test_retry_preserves_new_stage_anchor_and_schedule(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);c.r.save(root/'continuation-plan.json',{'deadline_utc':(datetime.datetime.now(datetime.timezone.utc)+datetime.timedelta(hours=10)).isoformat()})
            configs=[];finished=[]
            def config(name,cfg):configs.append(dict(cfg));return root/'config.json'
            def job(name,cmd,artifact,on_tick):
                artifact.parent.mkdir(exist_ok=True)
                if len(configs)==1:
                    c.r.save(artifact.parent/'latest.json',{'path':'resume-checkpoint'});raise RuntimeError('simulated process exit')
                return {'training_seconds':600,'checkpoint':'final'}
            with patch.multiple(c.r,RUN=root,config=config,run_job=job,cached=lambda name:finished[-1] if finished else None,done=lambda *a:finished.append(a),incident=lambda *a,**kw:None):
                cfg=dict(initial_weights='new-anchor',bootstrap_checkpoint='360m',schedule_minutes=600,minutes=600,pool_weights=[45,45,10,0]);c.train(cfg)
            self.assertEqual(configs[1],dict(cfg,resume_checkpoint='resume-checkpoint'));self.assertNotIn('resume_checkpoint',cfg)

    def test_wall_deadline_requests_graceful_stop(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);c.r.save(root/'continuation-plan.json',{'deadline_utc':(datetime.datetime.now(datetime.timezone.utc)-datetime.timedelta(seconds=1)).isoformat()});finished=[]
            def job(name,cmd,artifact,on_tick):
                artifact.parent.mkdir(exist_ok=True);on_tick({'training_seconds':0});self.assertTrue((artifact.parent/'stop.json').exists());return {'training_seconds':1}
            with patch.multiple(c.r,RUN=root,config=lambda *a:root/'config.json',run_job=job,cached=lambda name:finished[-1] if finished else None,done=lambda *a:finished.append(a)):
                c.train(dict(minutes=600))
if __name__=='__main__':unittest.main()
