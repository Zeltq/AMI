import unittest
from sf_duel.rules import Referee

class RulesTests(unittest.TestCase):
    def test_first_terminal_packet_waits_for_other_sides_tick(self):
        r=Referee();r.add({'team':2,'reason':'death','time':50.})
        self.assertFalse(r.ready({2:49.9,3:50.}))
        self.assertTrue(r.ready({3:50.033}))
        r.add({'team':3,'reason':'death','time':50.})
        self.assertTrue(r.ready({}));self.assertIsNone(r.result()['winner'])
    def test_first_death_is_a_terminal_loss(self):
        judge=Referee();judge.add(dict(team=2,time=100.,reason='death'))
        self.assertEqual(judge.result()['winner'],3)
    def test_tower_then_later_death_preserves_first_outcome(self):
        judge=Referee();judge.add(dict(team=3,time=80.,reason='tower'))
        judge.add(dict(team=2,time=81.,reason='death'))
        self.assertEqual(judge.result()['winner'],2)
    def test_ten_minutes_compares_last_hits_not_gold_or_denies(self):
        judge=Referee();judge.add(dict(team=2,time=600.,reason='timeout',last_hits=30,gold=99999,denies=40))
        self.assertIsNone(judge.result())
        judge.add(dict(team=3,time=600.,reason='timeout',last_hits=31,gold=0,denies=0))
        self.assertEqual(judge.result()['winner'],3)
    def test_equal_last_hits_are_draws(self):
        judge=Referee()
        for t in (2,3):judge.add(dict(team=t,time=600.,reason='timeout',last_hits=40))
        self.assertIsNone(judge.result()['winner'])
    def test_same_tick_deaths_are_draws(self):
        judge=Referee()
        for t in (2,3):judge.add(dict(team=t,time=100.,reason='death'))
        self.assertIsNone(judge.result()['winner'])
