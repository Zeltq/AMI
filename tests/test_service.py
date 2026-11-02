import copy
from pathlib import Path
import tempfile
import unittest
from fixtures import snapshot
from sf_duel.server import Service

class ServiceTests(unittest.TestCase):
    def test_retry_is_idempotent_and_stale_sequence_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            service=Service('probe',directory,device='cpu')
            try:
                s=snapshot();first=service.decide([s])[0]
                self.assertEqual(service.decide([s])[0],first)
                self.assertEqual(service.stats['requests'],1)
                newer=copy.deepcopy(s);newer['seq']=2;newer['time']=.1
                service.decide([newer])
                with self.assertRaisesRegex(ValueError,'Out-of-order'):service.decide([s])
                other=copy.deepcopy(s);other['match']='other-game'
                service.decide([other]);self.assertEqual(service.stats['requests'],3)
            finally:
                service.snapshots.close();service.records.close()
