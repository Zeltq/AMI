import json
from pathlib import Path
import tempfile
import unittest
from sf_duel.live.bridge import LogPackets,publish

class BridgeTests(unittest.TestCase):
    def test_capture_marker_excludes_setup_packets(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'log'
            path.write_text('SF_OBS|1|1|1|1|{"player":1,"seq":1}\nSF_CAPTURE_READY\n'
                            'SF_OBS|1|2|1|1|{"player":1,"seq":2}\n')
            reader=LogPackets(path,start_marker='SF_CAPTURE_READY')
            self.assertEqual(reader.read(),[{'player':1,'seq':2}])

    def test_interleaved_fragments_and_partial_lines(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'log';path.touch();reader=LogPackets(path)
            a=json.dumps({'player':1,'seq':4,'text':'test'}).encode()
            b=json.dumps({'player':5,'seq':2}).encode()
            path.write_bytes(b'SF_OBS|1|4|1|2|'+a[:15]+b'\nSF_OBS|5|2|1|1|'+b+b'\nSF_OBS|1|4|2|2|'+a[15:])
            self.assertEqual(reader.read(),[json.loads(b)])
            with path.open('ab') as f:f.write(b'\nSF_END|{"team":2,"reason":"death"}\n')
            self.assertEqual(reader.read(),[json.loads(a)])
            self.assertEqual(reader.events,[{'team':2,'reason':'death'}])
            self.assertEqual(reader.read(),[])
            with path.open('ab') as f:f.write(b'SF_RULE_ERROR|{"team":2,"reason":"attribute_bonus_upgrade_rejected_by_bot_api"}\n')
            self.assertEqual(reader.read(),[])
            self.assertEqual(reader.errors[0]['reason'],'attribute_bonus_upgrade_rejected_by_bot_api')

    def test_identity_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'log';path.write_text('SF_OBS|1|4|1|1|{"player":2,"seq":4}\n')
            with self.assertRaisesRegex(ValueError,'identity'):LogPackets(path).read()

    def test_response_is_namespaced_by_match_and_player(self):
        with tempfile.TemporaryDirectory() as directory:
            response={'match':'a'*32,'seq':7,'action':0}
            publish(1,response,directory)
            output=list(Path(directory).iterdir())
            self.assertEqual([p.name for p in output],['response_'+'a'*32+'_1.lua'])
            self.assertIn('["seq"]=7',output[0].read_text())
            response['match']='../unsafe'
            with self.assertRaises(ValueError):publish(1,response,directory)
