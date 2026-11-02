import unittest
from sf_duel.live.game import duel_clients

class ClientIdentityTests(unittest.TestCase):
    def test_sourcetv_does_not_change_which_heroes_are_kept(self):
        log="""0 BOT 0 0 active 0 'SourceTV'
1 BOT 0 0 active 0 'SF_DUEL_2_1 (Bot)'
2 BOT 0 0 active 0 'SF_DUEL_2_2 (Bot)'
6 BOT 0 0 active 0 'SF_DUEL_3_1 (Bot)'
7 BOT 0 0 active 0 'SF_DUEL_3_2 (Bot)'"""
        clients=duel_clients(log)
        self.assertEqual(clients[(2,1)],1)
        self.assertEqual(clients[(3,1)],6)
        self.assertEqual({value for (_,index),value in clients.items() if index!=1},{2,7})

    def test_native_server_without_sourcetv(self):
        self.assertEqual(duel_clients("0 BOT 0 0 active 0 'SF_DUEL_2_1'\n5 BOT 0 0 active 0 'SF_DUEL_3_1'"),
            {(2,1):0,(3,1):5})
