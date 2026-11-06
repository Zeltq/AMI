import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'gpu/scripts'))
from chio_mid_phase import MidPhase
class MidPhaseTests(unittest.TestCase):
    def test_exact_clock_handoff_masks_loss_and_resets_previous(self):
        entry={'file':'block-000028-shard-000.ggd.gz','actors':[640,641]};manifest={'files':[{'file':'block-000027-shard-000.ggd.gz','actors':[640,641]},entry],'game_metrics':[{'game':320,'sides':[{'mid_arrival_elapsed_seconds':26.3999958},{'mid_arrival_elapsed_seconds':26.5999966}]}]}
        phase=MidPhase(manifest);signals=np.zeros((2,128,13),np.float32);b={'loss_valid':np.ones(256,bool),'previous':np.full((256,55),.5,np.float32)}
        reset,allowed,handoff=phase.apply(entry,[640,641],b,signals)
        self.assertEqual(allowed.sum(axis=1).tolist(),[124,123]);self.assertEqual(np.argwhere(handoff).tolist(),[[0,4],[1,5]])
        self.assertEqual(reset.sum(),2);self.assertEqual(b['previous'][4].sum(),2);self.assertEqual(b['previous'][4,1],1);self.assertEqual(b['previous'][4,49],1)
    def test_never_arrived_is_never_trainable_and_later_blocks_stay_enabled(self):
        m={'files':[{'file':'block-000000-shard-000.ggd.gz','actors':[0,1]}],'game_metrics':[{'game':0,'sides':[{'mid_arrival_elapsed_seconds':None},{'mid_arrival_elapsed_seconds':10.}]}]};phase=MidPhase(m)
        b={'loss_valid':np.ones(256,bool),'previous':np.zeros((256,55),np.float32)};reset,allowed,handoff=phase.apply({'file':'block-000010-shard-000.ggd.gz'},[0,1],b,np.zeros((2,128,13),np.float32))
        self.assertEqual(allowed.sum(axis=1).tolist(),[0,128]);self.assertFalse(handoff.any());self.assertEqual(reset.sum(),0)

    def test_native_mask_is_not_recomputed_after_empty_premid_shards_were_dropped(self):
        phase=MidPhase({'scripted_base_to_mid':True});signals=np.zeros((1,128,13),np.float32);signals[0,4:,11]=1;signals[0,4,10]=1
        b={'loss_valid':signals[:,:,11].reshape(-1)>.5,'previous':np.ones((128,55),np.float32)}
        reset,allowed,handoff=phase.apply({'file':'block-000001-shard-000.ggd.gz'},[0],b,signals)
        self.assertEqual(allowed.sum(),124);self.assertEqual(reset[0,4],1);self.assertFalse(handoff.any());self.assertEqual(b['previous'].sum(),128*55)
