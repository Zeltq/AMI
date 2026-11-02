import importlib.util,socket,struct,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('play_human_teacher',ROOT/'scripts/play_human_teacher.py')
play=importlib.util.module_from_spec(spec);spec.loader.exec_module(play)

def packet(id,kind,text):
 data=struct.pack('<ii',id,kind)+text.encode()+b'\0\0';return struct.pack('<i',len(data))+data

class FakeSocket:
 def __init__(self,data):self.data=data;self.sent=[]
 def settimeout(self,value):pass
 def sendall(self,data):self.sent.append(data)
 def recv(self,n):
  if not self.data:raise socket.timeout()
  result=self.data[:min(n,3)];self.data=self.data[len(result):];return result
 def close(self):pass

class HumanTeacherSetupTests(unittest.TestCase):
 def test_only_extra_bot_userids_are_removed(self):
  text=" 5 BOT active 'SF_DUEL_2_1'\n 6 BOT active 'SF_DUEL_3_1'\n 7 BOT active 'SF_DUEL_3_2'\n 9 STEAM active 'Human player'\n"
  self.assertEqual(sorted(play.extra_bot_ids(text,2)),[5,7]);self.assertEqual(sorted(play.extra_bot_ids(text,3)),[6,7])
 def test_source_rcon_handles_fragmented_authentication_and_response(self):
  sock=FakeSocket(packet(1,0,'')+packet(1,2,'')+packet(11,0,'status result'))
  with patch.object(play.socket,'create_connection',return_value=sock):
   connection=play.Rcon(27295,'local-password');self.assertEqual(connection.command('status'),'status result')
  self.assertEqual(struct.unpack('<iii',sock.sent[0][:12]),(len(sock.sent[0])-4,1,3))
 def test_rcon_rejects_authentication_failure(self):
  sock=FakeSocket(packet(-1,2,''))
  with patch.object(play.socket,'create_connection',return_value=sock):
   with self.assertRaisesRegex(RuntimeError,'authentication'):play.Rcon(27295,'wrong')
 def test_rcon_rejects_invalid_packet_size(self):
  sock=FakeSocket(struct.pack('<i',2))
  with patch.object(play.socket,'create_connection',return_value=sock):
   with self.assertRaisesRegex(ValueError,'packet'):play.Rcon(27295,'wrong')
 def test_manual_setup_verifies_composition_before_enabling_the_teacher(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);(root/'cfg').mkdir();log=root/'console.log';log.write_text('SF_V9_HUMAN_READY|2|0\n')
   replies=[" 5 BOT active 'SF_DUEL_2_1'\n 6 BOT active 'SF_DUEL_3_1'\n 7 BOT active 'SF_DUEL_3_2'\n"," 6 BOT active 'SF_DUEL_3_1'\n"]
   def reply(_):
    with log.open('a') as f:f.write(replies.pop(0))
   with patch('builtins.input',side_effect=reply):play.manual_setup(dict(content=str(root),human_team=2,teacher_team=3),log)
   self.assertEqual((root/'cfg/ami_teacher_cleanup.cfg').read_text(),'kickid 7\nkickid 5\nent_remove_all npc_dota_hero_lina\n')

if __name__=='__main__':unittest.main()
