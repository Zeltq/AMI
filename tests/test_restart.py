import copy
from dataclasses import replace
import gzip
import struct
import tempfile
from pathlib import Path
import unittest
import threading
import time
from queue import Queue
import numpy as np
import torch
from sf_duel.protos.duel_pb2 import Record
from sf_duel.selfplay.quality import audit_match
from sf_duel.selfplay.ppo import optimize,states_at,refresh,warm_value_head
from sf_duel.selfplay.config import Config
from sf_duel.model import Policy
from test_selfplay import synthetic_episode


class CollectionRecoveryTests(unittest.TestCase):
    def recover(self,root):
        import sys
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
        from run_mia_restart import collection_parts
        return collection_parts(root)

    def test_drained_parts_keep_all_accepted_games(self):
        import json
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for name,matches in [('collection',['a','b']),('collection-attempt-2',['c'])]:
                p=root/name;p.mkdir();(p/'collection.json').write_text(json.dumps({'matches':[{'run':m} for m in matches],'quarantine':[]}))
            (root/'collection/drain').touch()
            self.assertEqual([m['run'] for p in self.recover(root) for m in p['matches']],['a','b','c'])

    def test_quarantine_needs_repair_and_duplicate_matches_are_rejected(self):
        import json
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p=root/'collection';p.mkdir();(p/'collection.json').write_text(json.dumps({'matches':[{'run':'a'}],'quarantine':[{'run':'bad'}]}))
            with self.assertRaisesRegex(RuntimeError,'diagnosis'):self.recover(root)
            (p/'quality-repair.json').write_text('{}');self.assertEqual(len(self.recover(root)),1)
            p=root/'collection-attempt-2';p.mkdir();(p/'collection.json').write_text(json.dumps({'matches':[{'run':'a'}]}))
            with self.assertRaisesRegex(ValueError,'Duplicate'):self.recover(root)


class RolloutStartupTests(unittest.TestCase):
    def test_first_append_defers_filesystem_and_metadata_precedes_shard(self):
        from sf_duel.selfplay.rollouts import RolloutWriter
        from sf_duel.observations import encode
        from fixtures import snapshot
        import json
        release=threading.Event()
        class PausedWriter(RolloutWriter):
            def _work(self):
                release.wait(3);super()._work()
        with tempfile.TemporaryDirectory() as tmp:
            writer=PausedWriter(Path(tmp)/'rollouts')
            try:
                writer.append('match',2,'hash',True,{'observation':encode(snapshot()).arrays,
                    'action':dict(action=0,buy=0,item=0,target=0,x=64,y=64),'sequence':1})
                self.assertEqual(list(writer.root.iterdir()),[])
            finally:
                release.set();writer.close()
            folder=next(writer.root.iterdir())
            self.assertEqual(json.loads((folder/'actor.json').read_text())['policy_sha256'],'hash')
            with np.load(folder/'00000001.npz') as data:self.assertEqual(data['sequence'].tolist(),[1])


class TelemetryTests(unittest.TestCase):
    def test_rpc_timing_is_bounded_and_separates_queue_from_service(self):
        from sf_duel.native_policy_server import Service
        service=Service.__new__(Service);service.rpc_timings=Queue(64)
        r=Record();r.episode='match';r.decision.team=2;r.decision.sequence=7
        service.record_rpc_timing(r,0,.001,.002,.010,.011)
        self.assertTrue(service.rpc_timings.empty())
        for _ in range(70):service.record_rpc_timing(r,0,.010,.070,.090,.100)
        self.assertEqual(service.rpc_timings.qsize(),64)
        event=service.rpc_timings.get_nowait()
        self.assertAlmostEqual(event['queue_ms'],60)
        self.assertAlmostEqual(event['service_ms'],20)
        self.assertAlmostEqual(event['total_ms'],100)
        self.assertEqual(event['sequence'],7)

    def test_blocked_disk_does_not_block_new_snapshots(self):
        from sf_duel.native_policy_server import Service
        with tempfile.TemporaryDirectory() as tmp:
            service=Service.__new__(Service);service.out=Path(tmp);service.stats={'requests':1};service.batch_wait_ms=2.
            service.telemetry_queue=Queue(1);service.telemetry_error=None;service.telemetry_ms=0.
            entered=threading.Event();release=threading.Event();saved=[]
            def slow_write(payload):
                entered.set();release.wait(3);saved.append(payload)
            service.write_stats=slow_write
            service.telemetry_thread=threading.Thread(target=service.telemetry_work,daemon=True);service.telemetry_thread.start()
            try:
                service.queue_telemetry();self.assertTrue(entered.wait(2))
                start=time.monotonic()
                for n in range(2,20):service.stats['requests']=n;service.queue_telemetry()
                self.assertLess(time.monotonic()-start,.2)
            finally:
                release.set();service.finish_telemetry()
            import json
            self.assertEqual(json.loads(saved[-1])['requests'],19)
            self.assertIsNone(service.telemetry_error)


class AdmissionTests(unittest.TestCase):
    def check(self,delay=6,n=3,missing=False,terminal=.9,gap=False,reason='death',prefix=None):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with gzip.open(root/'native-decisions.pb.gz','wb') as f:
                for team in (2,3):
                    for i in range(n):
                        r=Record();d=r.decision;d.team=team;d.sequence=i+1;d.observed_tick=100+i*4+(4 if gap and i==1 else 0);d.execute_tick=d.observed_tick+6;d.observed_dota_time=i*4/30;r.observation.team_id=team
                        data=r.SerializeToString();f.write(struct.pack('<I',len(data))+data)
                        if not (missing and i==n-1):
                            r=Record();r.execution.team=team;r.execution.sequence=i+1;r.execution.tick=100+i*4+(delay if i==0 else 6);r.execution.accepted=True
                            data=r.SerializeToString();f.write(struct.pack('<I',len(data))+data)
            return audit_match({'run':str(root),'time':terminal,'reason':reason,'outcome':{'winner':2}},output=False,prefix_seconds=prefix)

    def test_explicit_prefix_stop_keeps_all_transport_thresholds(self):
        self.assertFalse(self.check(reason='diagnostic_cut')['passed'])
        self.assertTrue(self.check(reason='diagnostic_cut',prefix=.9)['passed'])
        self.assertFalse(self.check(reason='diagnostic_cut',prefix=.8)['passed'])
        self.assertFalse(self.check(reason='diagnostic_cut',prefix=.9,delay=12)['passed'])
        self.assertFalse(self.check(reason='diagnostic_cut',prefix=.9,missing=True)['passed'])

    def test_clock_thresholds_and_rare_lateness(self):
        self.assertTrue(self.check(delay=8)['passed'])
        self.assertFalse(self.check(delay=12)['passed'])
        self.assertFalse(self.check(delay=9)['passed'])
        self.assertTrue(self.check(delay=9,n=1001,terminal=200)['passed'])

    def test_gap_and_missing_ack_vs_terminal(self):
        self.assertFalse(self.check(gap=True)['passed'])
        self.assertFalse(self.check(missing=True)['passed'])
        result=self.check(missing=True,terminal=.4);self.assertTrue(result['passed']);self.assertEqual(result['terminal_pending'],2)
        self.assertTrue(self.check(missing=True,terminal=.5)['passed'])


class RestartPPOTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(13);self.m=Policy().eval();self.e=synthetic_episode(self.m,n=14)
        self.cfg=replace(Config(),gae_lambda=.95,burn_in=3,bptt=5,batch_sequences=2,value_warmup_epochs=2,ppo_epochs=1,kl_check_batches=1,learning_rate=1e-6,fresh_memory=True)

    def test_current_memory_matches_full_replay_after_weights_change(self):
        with torch.no_grad():self.m.memory.bias_ih_l0.add_(.01)
        full=refresh(self.m,self.e,self.cfg);selected=states_at(self.m,self.e,[0,2,7])
        for point in selected:
            for a,b in zip(selected[point],full['states'][point]):torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-5)

    def test_adam_is_continued_and_coverage_counts_committed_frames(self):
        result,opt=optimize(self.m,[self.e],self.cfg,warmup=False)
        self.assertEqual(result['frame_coverage'],1);self.assertEqual(result['unique_frames_retained'],14)
        next_episode=synthetic_episode(self.m,n=14);before=result['optimizer_steps_after']
        result2,_=optimize(self.m,[next_episode],self.cfg,warmup=False,optimizer_state=opt.state_dict())
        self.assertEqual(result2['optimizer_steps_before'],before)
        self.assertEqual(result2['optimizer_steps_after'],before+result2['steps_retained'])

    def test_accumulation_matches_effective_batch(self):
        a=copy.deepcopy(self.m);b=copy.deepcopy(self.m)
        ra,_=optimize(a,[self.e],replace(self.cfg,batch_sequences=1,gradient_accumulation=2),warmup=False)
        rb,_=optimize(b,[self.e],self.cfg,warmup=False)
        self.assertEqual(ra['steps_retained'],rb['steps_retained'])
        for key,value in a.state_dict().items():torch.testing.assert_close(value,b.state_dict()[key],rtol=1e-5,atol=1e-6)

    def test_holdout_is_separate_and_actor_unchanged(self):
        val=copy.deepcopy(self.e);val['rewards'][-1]=-1
        original={k:v.clone() for k,v in self.m.state_dict().items() if not k.startswith('value_head.')}
        r=warm_value_head(self.m,[self.e],self.cfg,validation_episodes=[val])
        self.assertTrue(r['validation_is_independent']);self.assertLessEqual(r['validation_mse_after'],r['validation_mse_before']+1e-7)
        for k,v in original.items():torch.testing.assert_close(v,self.m.state_dict()[k],rtol=0,atol=0)

    def test_routes_are_prefetched_before_first_inference(self):
        import json
        from sf_duel.native_policy_server import Service
        with tempfile.TemporaryDirectory() as tmp:
            service=Service.__new__(Service);service.out=Path(tmp);(service.out/'registrations').mkdir()
            service.models={'current':None,'0':None};service.route_cache={};service.registered_files=set()
            p=service.out/'registrations/1.json';p.write_text(json.dumps({'episode':'C:\\match','routing':{'2':'current','3':'0'}}))
            service.prefetch_routes();self.assertEqual(service.route_cache['C:\\match']['3'],'0')
            service.prefetch_routes();self.assertEqual(len(service.registered_files),1)
