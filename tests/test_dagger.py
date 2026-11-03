import sys,unittest,tempfile
from pathlib import Path
import numpy as np
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'gpu/scripts'))
from dagger_data import eligible,pack,mixed_batches
from ggd_dataset import layout
from train_ggd import gameplay_score,selection_score
from run_dagger import Runner,write


class DaggerTests(unittest.TestCase):
    def test_resume_reuses_completed_stage_and_stop_never_launches(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);runner=Runner({},out)
            with patch.object(runner,'status'),patch.object(runner,'archive'),patch('run_dagger.subprocess.Popen') as popen:
                child=popen.return_value;child.pid=123;child.returncode=0;child.poll.return_value=0
                target=runner.job('collect',lambda t:['test'],lambda t:write(t/'report.json',{'complete':True}))
                self.assertEqual(popen.call_count,1)
            resumed=Runner({},out)
            with patch('run_dagger.subprocess.Popen') as popen:
                self.assertEqual(resumed.job('collect',lambda t:['must-not-run']),target)
                write(out/'stop.json',{})
                with self.assertRaises(InterruptedError):resumed.job('next',lambda t:['must-not-run'])
                popen.assert_not_called()

    def test_partial_dataset_is_never_a_completed_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner=Runner({},Path(tmp))
            with patch.object(runner,'status'),patch.object(runner,'archive'),patch('run_dagger.subprocess.Popen') as popen:
                child=popen.return_value;child.pid=123;child.returncode=0;child.poll.return_value=0
                with self.assertRaisesRegex(RuntimeError,'Incomplete'):
                    runner.job('collect',lambda t:['test'],lambda t:write(t/'report.json',{'partial':True}))
            self.assertNotIn('collect',runner.state['stages'])

    def test_new_pass_is_exact_and_terminal_padding_never_makes_replay_majority(self):
        header={'actors':[0,3,4,7,8],'source':'dagger-round-1'}
        obs=np.zeros((5,128,1),np.float16);signals=np.zeros((5,128,13),np.float32)
        for i in range(5):signals[i,:i+1,11]=1;signals[i,:,0]=i+1
        old=np.zeros((128,13),np.float32);old[:,11]=1
        manifest={'new_source':'dagger-round-1','files':[{'source':'dagger-round-1','file':'new','sha256':'x','actors':header['actors']}]}
        with patch('dagger_data.read_block',return_value=(header,obs,signals)),patch('dagger_data.OldSequences') as sample,patch('dagger_data.pack',side_effect=lambda s:s):
            sample.return_value.draw.return_value=({},obs[0],old)
            batches=list(mixed_batches(Path('.'),manifest,8,7))
        self.assertEqual([len(b) for b in batches],[8,2])
        labels=[]
        for batch in batches:
            n=len(batch)//2;fresh=batch[:n];replay=batch[n:]
            labels.extend(int(s[0,0]) for _,_,s in fresh)
            self.assertGreaterEqual(sum((s[:,11]>.5).sum() for _,_,s in fresh),sum((s[:,11]>.5).sum() for _,_,s in replay))
        self.assertEqual(sorted(labels),[1,2,3,4,5])
        self.assertEqual(old[:,11].sum(),128,'replay cache must not be mutated')

    def test_heldout_and_opponent_never_train(self):
        actors=[a for a in range(512) if eligible(a,'train','dagger-round-1')]
        self.assertEqual(len(actors),192)
        self.assertTrue(all(a%2==(a//2)%2 and a//2%8<6 for a in actors))
        self.assertEqual(sum(eligible(a,'train','teacher-demo') for a in range(512)),384)

    def test_gameplay_selection_prioritizes_early_death_then_own_farm(self):
        def report(death,lh):
            return {'complete':True,'per_match':[{'sides':[
                {'controller':'GGD','first_death_game_seconds':death,'lh':lh,'deny':3},
                {'controller':'teacher','first_death_game_seconds':None,'lh':1000,'deny':1000}]}]}
        self.assertLess(gameplay_score(report(None,1)),gameplay_score(report(119,500)))
        self.assertLess(gameplay_score(report(120,50)),gameplay_score(report(None,1)))
        self.assertEqual(selection_score({'validation':{'nll':0}}),(float('inf'),float('inf')))

    def test_union_bucket_padding_keeps_targets_and_previous(self):
        groups,width=layout()
        def sequence(pc,pt,pp,target):
            columns=[]
            for name,(at,shape,_) in groups.items():
                ix=np.arange(at,at+int(np.prod(shape))).reshape(shape)
                if name in ('creeps','creep_combat','creep_valid','creep_ally'):ix=ix[:pc]
                if name in ('trees','tree_valid'):ix=ix[:pt]
                if name in ('projectiles','projectile_combat','projectile_valid'):ix=ix[:pp]
                if name in ('attack_mask','tree_mask','self_mask','tp_mask'):ix=np.r_[ix[:2+pc],ix[258:262+pt],ix[-1:]]
                columns.extend(ix.reshape(-1).tolist())
            h={'columns':columns,'buckets':[pc,pt,pp]}
            o=np.ones((128,len(columns)),np.float16);s=np.zeros((128,13),np.float32)
            s[:,0]=3;s[:,3]=target;s[:,11]=1;s[0,10]=1
            return h,o,s
        arrays,labels,valid,n,resets=pack([sequence(16,32,16,258),sequence(32,64,32,518)])
        self.assertEqual(n,2);self.assertEqual(labels[0,3],34);self.assertEqual(labels[128,3],102)
        self.assertTrue(valid.all());self.assertEqual(resets.sum(),2)
        self.assertTrue((arrays['previous']==1).all())
        self.assertTrue((arrays['creeps'][:128,16:]==0).all())

if __name__=='__main__':unittest.main()
