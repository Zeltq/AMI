import gzip,json,struct,sys,tempfile,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'gpu/scripts'))
from train_chio import returns
from sf_duel.chio_reward import SNAP

class CriticTargets(unittest.TestCase):
    def test_returns_cross_blocks_but_stop_at_terminal(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for block,reward,terminal in [(0,1.,0),(1,2.,1)]:
                data=np.zeros((2,128,2*SNAP+30),np.float32);data[:,0,-1]=1;data[:,0,-2]=reward;data[:,0,SNAP+21]=terminal
                header=json.dumps({'format':'chio-reward-v2','shape':list(data.shape),'actors':[0,1],'frames':1,'zero_sum':True}).encode()
                with gzip.open(root/f'chio-reward-{block:06}.f32.gz','wb') as f:f.write(struct.pack('<I',len(header)));f.write(header);f.write(data.tobytes())
            targets=returns(root,.997,True)
            self.assertAlmostEqual(float(targets[0,0][0]),1+.997*2,places=6)
            self.assertEqual(targets[1,0][0],2);self.assertEqual(np.count_nonzero(targets[0,0]),1)
            with self.assertRaisesRegex(ValueError,'zero_sum'):returns(root,.997,False)

if __name__=='__main__':unittest.main()
