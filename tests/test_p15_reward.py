import unittest
import numpy as np
from sf_duel.chio_reward import components,health,tower,SNAP


def baseline():
    a=np.zeros((2,SNAP));a[:,4]=600;a[:,5]=1000;a[:,6]=200;a[:,7]=500;a[:,8:12]=1000;a[:,84]=30
    return a


class RewardContract(unittest.TestCase):
    def test_every_event_coefficient_and_positive_gold_only(self):
        before=baseline();after=before.copy();after[0,3]+=100;after[0,1]+=50;after[0,2]-=100
        after[0,12:16]+=[1,2,1,1];after[0,16:19]+=[1,2,3]
        c,total=components(before,after,False)
        np.testing.assert_allclose(c[0,:14],[.2,.3,0,0,-.12,.2,1.5,-3,0,-.3,-.2,-.3,0,0])
        self.assertAlmostEqual(total[0],c[0].sum());self.assertTrue((c[:,14:]==0).all())
        after[0,1]=-1;self.assertEqual(components(before,after,False)[0][0,1],0)

    def test_hp_mana_and_tower_fraction_changes_both_directions(self):
        before=baseline();after=before.copy();after[0,4]=500;after[0,6]=100;after[0,8]=900
        c,_=components(before,after,False)
        self.assertAlmostEqual(c[0,2],-.1);self.assertAlmostEqual(c[0,3],health(.5)-health(.6))
        self.assertAlmostEqual(c[0,8],1.5*(tower(.9)-tower(1)))
        reverse,_=components(after,before,False);np.testing.assert_allclose(reverse[0,[2,3,8]],-c[0,[2,3,8]])
        after[0,7]=1000;self.assertAlmostEqual(components(before,after,False)[0][0,2],-.15)

    def test_six_minute_decay_excludes_kills_deaths_and_terminal_outcome(self):
        before=baseline();after=before.copy();after[:,0]=359.999;after[0,1:4]+=[10,10,10]
        after[0,12:19]+=1;after[0,21]=1;after[0,22]=1
        normal,_=components(before,after,False);after[:,0]=360;late,_=components(before,after,False)
        for c in range(14):self.assertAlmostEqual(late[0,c],normal[0,c]*(1 if c in (6,7,13) else .6))
        self.assertEqual(late[0,13],5)

    def test_zero_sum_subtracts_enemy_shaping_without_enemy_kill_death_outcome(self):
        before=baseline();after=before.copy();after[0,14]=1;after[1,15]=1;after[1,1]=10
        after[:,21]=1;after[:,22]=[1,-1]
        c,total=components(before,after,True)
        self.assertAlmostEqual(total[0],6.5-.06);self.assertAlmostEqual(total[1],-8+.06)
        self.assertTrue((c[:,[20,21,27]]==0).all())
        self.assertAlmostEqual(components(before,after,False)[1][0],6.5)

    def test_terminal_once_for_win_loss_draw_and_all_reasons(self):
        for reason in (1,2,3):
            for result in (-1,0,1):
                before=baseline();after=before.copy();after[:,0]=600;after[0,21:24]=[1,result,reason]
                self.assertEqual(components(before,after,False)[0][0,13],5*result)
                self.assertEqual(components(after,after,False)[0][0,13],0)

    def test_flask_break_replays_every_event_with_environment_heal_rate_and_floor(self):
        before=baseline();after=before.copy();after[0,25]=1;after[0,28:34]=[400,1000,350,2,100,0]
        expected=health(.35)-health(.46)
        self.assertAlmostEqual(components(before,after,False)[0][0,12],expected)
        after[0,84]=20
        self.assertAlmostEqual(components(before,after,False)[0][0,12],health(.35)-health(.44))
        after[0,84]=30
        after[0,31]=8;self.assertEqual(components(before,after,False)[0][0,12],-.4)
        before[0,25]=8;after[0,25]=10;after[0,34:40]=[200,500,100,8,101,1]
        self.assertAlmostEqual(components(before,after,False)[0][0,12],-.8)
        after[0,25]=17
        with self.assertRaisesRegex(ValueError,'overflow'):components(before,after)


if __name__=='__main__':unittest.main()
