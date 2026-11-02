from dataclasses import replace
import hashlib
import gzip
import struct
import json
import copy
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from fixtures import snapshot
from sf_duel.observations import encode,batch
from sf_duel.model import Policy
from sf_duel.selfplay.config import Config
from sf_duel.selfplay.rewards import potential,transition,gae
from sf_duel.selfplay.rollouts import pack,RolloutWriter,load_actor,prepare_rollouts
from sf_duel.protos.duel_pb2 import Record
from sf_duel.selfplay.ppo import verify_behavior,refresh,warm_value_head,optimize,batch_segments,normalized_advantages


def synthetic_episode(model,n=14):
    rows=[];labels=[];hs=[];cs=[];logs=[];values=[];state=None;device=next(model.parameters()).device
    for i in range(n):
        s=snapshot();s['self']['health']=600-i;s['creeps']=[dict(s['enemy'],id=j+20,type='creep',x=j*20) for j in range(i%5)]
        e=encode(s);b=batch([e],device);state=model.initial_state(1) if state is None else state
        hs.append(state[0][0,0].detach().cpu().numpy().copy());cs.append(state[1][0,0].detach().cpu().numpy().copy())
        with torch.no_grad():a,lp,_,v,state=model.decide(b,state)
        a={k:int(t[0]) for k,t in a.items()}
        # Convert padded batch target back into canonical observation indexing.
        pc=b['creeps'].shape[1];pt=b['trees'].shape[1];nc=len(e.arrays['creeps']);nt=len(e.arrays['trees']);target=a['target']
        if target>=6+pc+pt:a['target']=6+nc+nt
        elif target>=6+pc:a['target']=6+nc+target-(6+pc)
        elif target>=2+pc:a['target']=2+nc+target-(2+pc)
        rows.append(e.arrays);labels.append(a);logs.append(float(lp[0]));values.append(float(v[0]))
    result=pack(rows,labels);result.update(h=np.stack(hs),c=np.stack(cs),old_log_prob=np.asarray(logs,np.float32),old_value=np.asarray(values,np.float32),rewards=np.zeros(n,np.float32),terminals=np.zeros(n,bool))
    result['rewards'][-1]=1;result['terminals'][-1]=True
    return result


class RewardTests(unittest.TestCase):
    def test_completed_dataset_joins_authoritative_ticks_and_excludes_old_opponent(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);run=root/'game';run.mkdir();writer=RolloutWriter(root/'rollouts')
            for team in (2,3):
                for i in range(3):
                    e=encode(snapshot());label=dict(action=0,buy=0,item=0,target=0,x=64,y=64)
                    writer.append(str(run),team,'current' if team==2 else 'archive',team==2,dict(observation=e.arrays,action=label,h=np.zeros(256,np.float32),c=np.zeros(256,np.float32),sequence=i+1,tick=i*4,time=-1+i/7.5,old_log_prob=np.float32(-1),old_value=np.float32(0),reward_state=np.array([i,0,600,600,1800,1800],np.float32)))
            writer.close()
            with gzip.open(run/'native-decisions.pb.gz','wb') as f:
                for i in range(3):
                    for team in (2,3):
                        r=Record();r.reward_frame.team=team;r.reward_frame.tick=i*4;r.reward_frame.values.extend([i,0,600,600,1800,1800])
                        raw=r.SerializeToString();f.write(struct.pack('<I',len(raw)));f.write(raw)
            report={'run':str(run),'time':1,'reason':'death','outcome':{'winner':2},'offset_ticks':6}
            result=prepare_rollouts(root/'rollouts',[report],root/'dataset','current')
            self.assertEqual(len(result['episodes']),1)
            path=root/'dataset'/result['episodes'][0]['path']
            self.assertAlmostEqual(float(np.load(path/'rewards.npy').sum()),1)
            self.assertTrue(np.load(path/'terminals.npy')[-1])
            self.assertFalse(np.load(path/'execution_known.npy').any())
            archive_meta=next(p for p in (root/'rollouts').glob('*/actor.json') if json.loads(p.read_text())['team']==3)
            original_meta=json.loads(archive_meta.read_text());same=dict(original_meta,policy_sha256='current');archive_meta.write_text(json.dumps(same))
            identical=prepare_rollouts(root/'rollouts',[report],root/'identical-policy','current')
            self.assertEqual(len(identical['episodes']),2)
            archive_meta.write_text(json.dumps(original_meta))
            def rewrite_missing(missing):
                with gzip.open(run/'native-decisions.pb.gz','wb') as f:
                    for i in range(3):
                        for team in (2,3):
                            if (team,i) in missing:continue
                            r=Record();r.reward_frame.team=team;r.reward_frame.tick=i*4;r.reward_frame.values.extend([i,0,600,600,1800,1800])
                            raw=r.SerializeToString();f.write(struct.pack('<I',len(raw)));f.write(raw)
            rewrite_missing({(3,0)})
            aligned=prepare_rollouts(root/'rollouts',[report],root/'aligned','current');folder=root/'aligned'/aligned['episodes'][0]['path']
            self.assertEqual(aligned['episodes'][0]['masked_startup_frames'],1)
            np.testing.assert_array_equal(np.load(folder/'loss_valid.npy'),[False,True,True])
            self.assertEqual(len(np.load(folder/'labels.npy')),3)
            self.assertAlmostEqual(float(np.load(folder/'rewards.npy')[1:].sum()),1)
            rewrite_missing({(3,1)})
            with self.assertRaisesRegex(ValueError,'same-tick'):
                prepare_rollouts(root/'rollouts',[report],root/'interior-gap','current')
            with self.assertRaisesRegex(ValueError,'true completed'):
                prepare_rollouts(root/'rollouts',[{**report,'reason':'diagnostic_cut'}],root/'rejected','current')

    def test_antisymmetry_caps_and_terminal_telescoping(self):
        a=np.array([20,5,400,600,1500,1800.]);b=np.array([10,2,300,700,1400,1800.])
        p=float(potential(a,b));self.assertAlmostEqual(p,-float(potential(b,a)))
        extreme=a.copy();extreme[:2]=1e6;self.assertLessEqual(abs(potential(extreme,b)),.66)
        for result in (-1,0,1):
            total=transition(0,p)+transition(p,.2)+transition(.2,999,result,True)
            self.assertAlmostEqual(total,result)
        self.assertAlmostEqual(transition(p,0,1,True)+transition(-p,0,-1,True),0)

    def test_healing_max_hp_and_unknown(self):
        full=np.array([0,0,600,600,1800,1800.]);hurt=full.copy();hurt[2]=300
        p0=potential(full,full);p1=potential(full,hurt)
        self.assertGreater(p1,p0);self.assertAlmostEqual(transition(p0,p1)+transition(p1,p0),0)
        bigger=full.copy();bigger[2:4]=1200
        self.assertEqual(potential(bigger,full),0)
        bigger[2]=600;self.assertAlmostEqual(potential(bigger,full),-.05)
        broken=full.copy();broken[2]=np.nan
        with self.assertRaises(ValueError):potential(broken,full)

    def test_gae_terminal_bootstrap_and_undiscounted_delayed_result(self):
        n=4500;r=np.zeros(n);r[-1]=1;v=np.full(n,.2);done=np.zeros(n,bool);done[-1]=True
        a,g=gae(r,v,done);np.testing.assert_allclose(a,.8);np.testing.assert_allclose(g,1)
        a,g=gae([0,0],[.1,.2],[False,False],bootstrap=.7);np.testing.assert_allclose(g,.7)
        with self.assertRaises(ValueError):gae([1],[0],[True],bootstrap=1)


class RecurrentPPOTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(12);self.model=Policy().eval()
        self.cfg=replace(Config(),burn_in=3,bptt=5,batch_sequences=2,value_warmup_epochs=1,ppo_epochs=1,kl_check_batches=1)
        self.e=synthetic_episode(self.model)

    def test_log_prob_full_history_and_input_states(self):
        result=verify_behavior(self.model,[self.e],self.cfg)
        self.assertLess(result['max_log_probability_error'],1e-4)
        broken=dict(self.e);broken['old_log_prob']=self.e['old_log_prob']+.01
        with self.assertRaises(ValueError):verify_behavior(self.model,[broken],self.cfg)

    def test_burn_padding_masks_and_global_advantages(self):
        refs=[refresh(self.model,self.e,self.cfg)]
        b,y,mask,state,*_=batch_segments([self.e],[(0,5,2),(0,10,7)],refs,[np.ones(14)],[np.ones(14)],self.cfg,'cpu')
        self.assertEqual(mask[:8].sum(),5);self.assertEqual(mask[8:].sum(),4)
        self.assertGreater(state[0].abs().sum(),0)
        a=normalized_advantages([np.array([0,1]),np.array([2,3])]);self.assertAlmostEqual(float(np.concatenate(a).mean()),0,places=6)
        self.assertLess(a[0].mean(),0)
        masked=normalized_advantages([np.array([999.,1.,3.])],[np.array([False,True,True])])[0]
        np.testing.assert_allclose(masked,[0,-1,1])

    def test_value_warmup_freezes_actor_and_ppo_is_finite(self):
        self.e['loss_valid'][:2]=False
        result=warm_value_head(self.model,[self.e],self.cfg);self.assertTrue(result['actor_unchanged'])
        self.assertLessEqual(result['value_mse_after'],result['value_mse_before']+1e-7)
        self.assertTrue(result['shuffled_across_episodes'])
        result,_=optimize(self.model,[self.e],self.cfg,warmup=False)
        self.assertTrue(np.isfinite(result['final_kl']));self.assertGreater(result['steps_attempted'],0)
        self.assertTrue(all(torch.isfinite(p).all() for p in self.model.parameters()))

    def test_critic_minibatch_mixes_opposite_episode_results(self):
        # Identical histories with opposite results have optimal value zero.
        # Episode-at-a-time updates would instead produce alternating +/-2
        # bias gradients and forget whichever result was processed first.
        other=copy.deepcopy(self.e);other['rewards'][-1]=-1
        with torch.no_grad():
            for parameter in self.model.value_head.parameters():parameter.zero_()
        gradients=[]
        handle=self.model.value_head[-1].bias.register_hook(lambda g:gradients.append(float(g)))
        try:result=warm_value_head(self.model,[self.e,other],self.cfg)
        finally:handle.remove()
        self.assertEqual(len(gradients),1)
        self.assertLess(abs(gradients[0]),1e-6)
        self.assertAlmostEqual(result['value_mse_after'],1,places=6)

    def test_kl_limit_rolls_back_to_last_verified_weights(self):
        cfg=replace(self.cfg,learning_rate=.5,target_kl=1e-8,stop_kl=2e-8)
        original={k:v.clone() for k,v in self.model.state_dict().items()}
        result,_=optimize(self.model,[self.e],cfg,warmup=False)
        self.assertTrue(result['early_stop'])
        self.assertEqual(result['steps_retained'],0)
        for k,v in original.items():torch.testing.assert_close(v,self.model.state_dict()[k],rtol=0,atol=0)

    def test_normalized_entropy_scale_and_float32_shards(self):
        b=batch([encode(snapshot())])
        with torch.no_grad():r=self.model.sequence_decide(b,1,1,return_metrics=True)
        entropy=float(r[-1]['normalized_entropy'][0]);self.assertGreaterEqual(entropy,0);self.assertLessEqual(entropy,1.000001)
        with tempfile.TemporaryDirectory() as directory:
            writer=RolloutWriter(Path(directory)/'rollouts')
            for i in range(130):
                s=snapshot();s['creeps']=[dict(s['enemy'],id=j+20,type='creep') for j in range(20 if i>=128 else 1)]
                e=encode(s);label=dict(action=3,buy=0,item=0,target=2,x=64,y=64)
                writer.append('episode',2,'hash',True,dict(observation=e.arrays,action=label,h=np.zeros(256,np.float32),c=np.zeros(256,np.float32),sequence=i+1,tick=i*4,time=i/7.5,old_log_prob=np.float32(-1),old_value=np.float32(0),reward_state=np.array([0,0,600,600,1800,1800],np.float32)))
            writer.close();folder=next((Path(directory)/'rollouts').iterdir());loaded=load_actor(folder)
            self.assertEqual(len(loaded['labels']),130);self.assertTrue((loaded['labels'][:,3]==2).all());self.assertEqual(loaded['heroes'].dtype,np.float32)

    def test_policy_service_routes_versions_and_rejects_privileged_reward_input(self):
        from sf_duel.native_policy_server import Service
        service=Service.__new__(Service)
        service.actors={('a',2):{'policy':'current'},('b',3):{'policy':'archive'},('c',2):{'policy':'current'}}
        calls=[]
        def decide_group(requests,policy):
            calls.append((policy,len(requests)));return [(policy+':'+r.episode).encode() for r in requests]
        service.decide_group=decide_group
        requests=[]
        for episode,team in [('a',2),('b',3),('c',2)]:
            r=Record(episode=episode);r.decision.team=team;requests.append(r)
        self.assertEqual(service.decide(requests),[b'current:a',b'archive:b',b'current:c'])
        self.assertEqual(calls,[('current',2),('archive',1)])
        calls.clear();service.canonical_policy={'current':'current','archive':'current'}
        self.assertEqual(service.decide(requests),[b'current:a',b'current:b',b'current:c'])
        self.assertEqual(calls,[('current',3)])
        self.assertEqual(service.actors[('b',3)]['policy'],'archive')
        requests[0].reward_frame.team=2
        with self.assertRaisesRegex(ValueError,'Privileged'):service.decide(requests)
