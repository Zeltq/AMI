import socket
import unittest
from unittest.mock import patch,Mock
from sf_duel.native_policy_server import exact,policy_contract,Service,memory_clock_group


class NativeRPCTests(unittest.TestCase):
    def test_fallback_route_path_uses_current_host_filesystem(self):
        from sf_duel.protos.duel_pb2 import Record
        for host,expected in [('nt','C:/AMI/episode'),('posix','/mnt/c/AMI/episode')]:
            with self.subTest(host=host):
                service=Service.__new__(Service);service.actors={};service.models={'current':Mock()};service.stats={};service.decide_group=Mock(return_value=[b'reply'])
                request=Record(episode='C:\\AMI\\episode');request.decision.team=2
                with patch('sf_duel.native_policy_server.os.name',host),patch('sf_duel.native_policy_server.Path')as constructor:
                    (constructor.return_value/'policy-routing.json').exists.return_value=False
                    self.assertEqual(service.decide([request]),[b'reply'])
                    self.assertEqual(constructor.call_args.args[0],expected)

    def test_game_clock_reset_does_not_depend_on_rpc_start_or_count(self):
        self.assertEqual(memory_clock_group(-3.30000305,-3.3),0)
        self.assertEqual(memory_clock_group(-3.3-512/30,-3.3),-1)
        self.assertEqual(memory_clock_group(-3.3-1/30,-3.3),-1)
        self.assertEqual(memory_clock_group(-3.3+511/30,-3.3),0)
        self.assertEqual(memory_clock_group(-3.3+512/30,-3.3),1)
        with self.assertRaises(ValueError):memory_clock_group(float('nan'),0)

    def test_clock_resets_each_actor_only_when_its_game_clock_crosses_boundary(self):
        import torch
        from sf_duel.protos.duel_pb2 import Record
        for seq,old_group,time,expected in [(5,-1,-3.3,True),(128,0,0,False),(5,0,14,True),(5,None,0,True)]:
            with self.subTest(seq=seq,old_group=old_group,time=time):
                service=Service.__new__(Service);service.reset_origin=-3.3;service.contracts={'current':(128,'f16')}
                model=Mock();model.initial_state.return_value=(torch.zeros(1,1,2),torch.zeros(1,1,2));service.models={'current':model}
                service.stats={'episodes':{}};key=('clock',2)
                actor={'seq':seq,'clock_group':old_group,'previous':None,'state':model.initial_state.return_value,'adapter':Mock(),'shots':set(),'cooldowns':{}}
                service.actors={key:actor};r=Record(episode='clock');r.decision.team=2;r.decision.sequence=seq+1;r.observation.dota_time=time
                r.observation.units.add(name='npc_dota_hero_nevermore',team_id=2)
                with patch('sf_duel.native_policy_server.encode',return_value=Mock()),patch('sf_duel.native_policy_server.batch',side_effect=InterruptedError('after clock preparation')):
                    with self.assertRaises(InterruptedError):service.decide_group([r],'current')
                self.assertEqual(model.initial_state.called,expected)
                self.assertEqual(actor['clock_group'],memory_clock_group(time,-3.3))

    def test_new_life_resets_gpu_contract_memory_and_previous_but_keeps_legacy_contract(self):
        import torch
        from sf_duel.protos.duel_pb2 import Record
        for interval,tick,deaths,expected in [(128,140,1,True),(128,104,0,False),(128,140,0,False),(128,104,1,True),(0,140,1,False)]:
            with self.subTest(interval=interval,tick=tick,deaths=deaths):
                service=Service.__new__(Service);service.contracts={'current':(interval,'f32')}
                model=Mock();model.initial_state.return_value=(torch.zeros(1,1,2),torch.zeros(1,1,2));service.models={'current':model}
                service.stats={'episodes':{}};key=('life',2);prior={'old':'action'}
                actor={'seq':5,'tick':100,'previous':prior,'state':(torch.ones(1,1,2),torch.ones(1,1,2)),'adapter':Mock(),'shots':set(),'cooldowns':{}}
                service.actors={key:actor};r=Record(episode='life');r.decision.team=2;r.decision.sequence=6;r.decision.observed_tick=tick
                r.observation.units.add(name='npc_dota_hero_nevermore',team_id=2)
                r.observation.players.add(team_id=2,hero_id=11,deaths=deaths)
                # Stop at batching: this check exercises request validation,
                # encoding and life masks without loading a checkpoint/server.
                with patch('sf_duel.native_policy_server.encode',return_value=Mock()) as encoded,patch('sf_duel.native_policy_server.batch',side_effect=InterruptedError('after life preparation')):
                    with self.assertRaisesRegex(InterruptedError,'after life preparation'):service.decide_group([r],'current')
                self.assertEqual(model.initial_state.called,expected)
                self.assertEqual(encoded.call_args.args[1],None if expected else prior)
                self.assertEqual(service.stats.get('life_resets',0),int(expected))
                self.assertEqual(actor['tick'],tick)

    def test_gpu_and_legacy_checkpoint_contracts_remain_distinct(self):
        self.assertEqual(policy_contract({}),(0,'f32'))
        self.assertEqual(policy_contract({'gpu_pipeline':{'recurrent_reset_interval':128,'observation_storage':'f16'}}),(128,'f16'))
        for pipeline in ({'recurrent_reset_interval':64},{'observation_storage':'i8'}):
            with self.assertRaises(ValueError):policy_contract({'gpu_pipeline':pipeline})

    def test_fragmented_request_acknowledges_before_waiting_for_more(self):
        class ProxySocket:
            chunks=[b'ab',b'cd',b'ef'];acked=True
            def recv(self,n):
                if not self.acked:raise AssertionError('proxy is waiting for ACK before forwarding next fragment')
                self.acked=False
                return self.chunks.pop(0)
            def setsockopt(self,level,option,value):
                assert level==socket.IPPROTO_TCP and option==12 and value==1
                self.acked=True
        with patch.object(socket,'TCP_QUICKACK',12,create=True):
            self.assertEqual(exact(ProxySocket(),6),b'abcdef')
