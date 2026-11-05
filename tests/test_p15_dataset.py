import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'gpu/scripts'))
from ggd_dataset import layout,decode

class ChioInputs(unittest.TestCase):
    def test_legacy_clarity_columns_are_reserved_zero_and_item_rows_excluded(self):
        groups,width=layout(5);obs=np.zeros((1,1,width),np.float16);sig=np.zeros((1,1,13),np.float32)
        header={'version':5,'buckets':[16,32,16],'columns':list(range(width))}
        for name in ('heroes','creeps','towers'):
            offset,shape,_=groups[name]
            obs[0,0,offset:offset+np.prod(shape)].reshape(shape)[:,[38,86]]=1
        for name in ('items','enemy_items'):
            offset,shape,_=groups[name]
            for i in range(8):obs[0,0,offset:offset+np.prod(shape)].reshape(shape)[i]=i
        b=decode(header,obs,sig)
        for name in ('heroes','creeps','towers'):self.assertTrue((b[name][:,:,[38,86]]==0).all())
        for name in ('items','enemy_items'):np.testing.assert_equal(b[name][0,:,0],[0,1,2,5,4,6])

if __name__=='__main__':unittest.main()
