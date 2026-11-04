import json,sys,tempfile,unittest
from pathlib import Path
import numpy as np
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'gpu/scripts'))
from p09b_reward import potentials,rewards,targets,GAMMA

FIELDS=['own_lh','enemy_lh','own_deny','enemy_deny','own_hp_fraction','enemy_hp_fraction','own_mana_fraction','enemy_mana_fraction','terminal','result','game_seconds']
class RewardTests(unittest.TestCase):
    def test_discounted_shaping_telescopes_with_nonzero_initial_potential(self):
        states=np.zeros((4,len(FIELDS)));states[:,0]=[2,3,4,4];states[:,4]=[1,.8,.6,0];states[:,5]=1;states[-1,8]=1;states[-1,9]=-1
        transitions=np.stack([states[:-1],states[1:]],axis=1)
        r=rewards(transitions,FIELDS);phi=potentials(states,FIELDS)
        np.testing.assert_allclose((r[:,1:]*GAMMA**np.arange(3)[:,None]).sum(0),-phi[0],atol=1e-12)
        np.testing.assert_allclose(r[:,1:].sum(0)+(1-GAMMA)*phi[1:].sum(0),-phi[0],atol=1e-12)
        self.assertEqual(r[-1,0],-10)
        self.assertEqual(r[:-1,0].sum(),0)

    def test_mc_returns_cross_blocks_and_discount_invalid_steps_to_native_terminal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);files=[]
            for block in range(2):
                s=np.zeros((1,128,2,len(FIELDS)));v=np.zeros((1,128),bool);v[0,127 if block==0 else 0]=True
                s[:,:,:,0]=1
                s[0,:,:,10]=block
                if block==1:
                    s[0,2,1,8]=1;s[0,2,1,9]=1
                    s[0,3:,:,8]=1;s[0,3:,:,9]=1
                name=f'b{block}.npz';np.savez(root/name,state=s,valid=v,actors=[0],fields=FIELDS,match_outcome=np.ones((1,128),np.int8))
                files.append(dict(file=name,demonstration=f'b{block}.ggd.gz'))
            (root/'reward-metadata.json').write_text(json.dumps(dict(files=files)))
            result=targets(root,root/'out');self.assertEqual(result['actors'],1)
            self.assertEqual(result['terminals_outside_imitation_mask'],1)
            with np.load(root/'out/b1.npz') as z:self.assertAlmostEqual(float(z['returns'][0,0]),-.5+.99**2*10,places=5)
            with np.load(root/'out/b0.npz') as z:self.assertAlmostEqual(float(z['returns'][0,127]),-.5+.99**3*10,places=5)

    def test_episode_curves_exclude_incomplete_frozen_prefix_and_keep_denominators(self):
        import p09b_results as report
        import xml.etree.ElementTree as ET
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);run=root/'rl-main';run.mkdir();out=root/'reports';out.mkdir()
            (run/'config.json').write_text(json.dumps({'learning_rate':3e-5}))
            update=dict(block=1,training_seconds=60,learning=dict(warmup=False,value_loss=1,ev_post0=.2,rollout_kl_m2=.1,entropy=.4,clip_fraction=.3,kl_coefficient=.05,valid_decisions=256))
            (run/'history.jsonl').write_text(json.dumps(update)+'\n')
            episode=dict(training_seconds=60,reward=9.6,components=[10,.1,.2,-.3,-.4],initialization_prefix_decisions=0,game=dict(lane_creeps_spawned=8,sides=[dict(controller='GGD',lh=3,deny=1,first_death_game_seconds=None)]))
            prefix={**episode,'initialization_prefix_decisions':128,'reward':999}
            (run/'episodes.jsonl').write_text('\n'.join(json.dumps(e) for e in [prefix]+[episode]*100)+'\n')
            with patch.object(report,'RUN',root),patch.object(report,'REPORT',out):updates,bins=report.curves()
            self.assertEqual(len(bins),1);self.assertEqual(bins[0]['matches'],100)
            self.assertAlmostEqual(bins[0]['reward_sum'],960);self.assertEqual(bins[0]['farm'],400);self.assertEqual(bins[0]['spawn'],800)
            self.assertEqual(json.loads((out/'rl-curve-coverage.json').read_text())['excluded_frozen_prefix_episodes'],1)
            ET.parse(out/'rl-curves.svg')

    def test_interrupted_controller_accounts_time_and_refuses_live_child(self):
        import p09b_run as runner
        from datetime import datetime,timezone
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);history=[dict(stage='complete',exit_code=0,minutes=2,gpu_budget=True),dict(stage='interrupted',pid=123,started_utc='2026-11-04T00:00:00+00:00',gpu_budget=True)]
            runner.save(root/'commands.json',history)
            with patch.object(runner,'RUN',root),patch.object(runner,'pid_alive',return_value=True):
                with self.assertRaisesRegex(RuntimeError,'still alive'):runner.recover_interrupted_jobs(datetime(2026,11,4,0,10,tzinfo=timezone.utc))
            self.assertEqual(runner.read(root/'commands.json'),history)
            with patch.object(runner,'RUN',root),patch.object(runner,'pid_alive',return_value=False):runner.recover_interrupted_jobs(datetime(2026,11,4,0,10,tzinfo=timezone.utc))
            recovered=runner.read(root/'commands.json');self.assertEqual(recovered[0],history[0]);self.assertEqual(recovered[1]['minutes'],10);self.assertEqual(recovered[1]['exit_code'],'controller_interrupted')

    def test_rl_resume_keeps_schedule_and_refuses_extra_training_after_eight_hours(self):
        import p09b_rl as rl
        from p09b_run import save
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);reports=root/'reports';reports.mkdir();checkpoint=root/'rl-main-001/checkpoint-000100';(checkpoint/'weights').mkdir(parents=True)
            for name in ('model.bpk','optimizer.bpk','weights/manifest.json'):(checkpoint/name).write_text('{}')
            progress=dict(training_seconds=7200,last_evaluation=7200);save(checkpoint/'progress.json',progress)
            for name in ('smoke-check','resume-check'):save(reports/(name+'.json'),{'passed':True})
            save(reports/'warmup.json',dict(actor_weights={'changed':[]},result={'latest':str(root/'warmup-checkpoint')}))
            save(root/'commands.json',[dict(stage='rl-main-003',exit_code=1)])
            with patch.object(rl,'RUN',root),patch.object(rl,'REPORT',reports),patch.object(rl,'native',return_value=({'training_seconds':28800},root/'final')) as native,patch.object(rl,'publish'):
                rl.main_training();name,cfg=native.call_args.args
                self.assertEqual(name,'rl-main-004');self.assertEqual(cfg['resume_checkpoint'],str(checkpoint));self.assertEqual(cfg['minutes'],480);self.assertEqual(cfg['schedule_minutes'],480);self.assertTrue(cfg['evaluate'])
                progress['training_seconds']=28800;save(checkpoint/'progress.json',progress);native.reset_mock()
                with self.assertRaisesRegex(RuntimeError,'already trained'):rl.main_training()
                native.assert_not_called()

    def test_verified_binary_seal_rejects_changed_native_source(self):
        import p09b_rl as rl
        from p09b_run import save,sha
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);binary=root/'native.exe';binary.write_bytes(b'test fixture, not executable')
            save(root/'verified-gpu.json',dict(passed=True,sources={'native.rs':'old'},executable_sha256=sha(binary)))
            with patch.object(rl,'REPORT',root),patch.object(rl,'BINARY',binary),patch.object(rl,'rust_sources',return_value={'native.rs':'old'}):rl.verified()
            with patch.object(rl,'REPORT',root),patch.object(rl,'BINARY',binary),patch.object(rl,'rust_sources',return_value={'native.rs':'changed'}):
                with self.assertRaisesRegex(AssertionError,'needs verification'):rl.verified()

    def test_failed_critic_checkpoint_write_keeps_previous_checkpoint(self):
        import p09b_critic as critic
        import torch
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'last.pt';critic.save_checkpoint(path,{'value':torch.tensor(1)})
            def interrupted(data,target):
                Path(target).write_bytes(b'incomplete checkpoint');raise OSError('simulated interrupted write')
            with patch.object(critic.torch,'save',side_effect=interrupted):
                with self.assertRaisesRegex(OSError,'interrupted write'):critic.save_checkpoint(path,{'value':torch.tensor(2)})
            self.assertEqual(torch.load(path,weights_only=True)['value'].item(),1)
            critic.save_checkpoint(path,{'value':torch.tensor(2)})
            self.assertEqual(torch.load(path,weights_only=True)['value'].item(),2)

if __name__=='__main__':unittest.main()
