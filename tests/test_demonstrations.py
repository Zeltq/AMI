import copy
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from fixtures import snapshot
from sf_duel.demonstrations import demonstration_label,TeacherWriter


def frame():
    s=snapshot();s['legal']['actions'][5]=False
    s['probe']={'allied_heroes':[{'name':'npc_dota_hero_nevermore'}]}
    s['demonstration']={'command':{'seq':1,'match':s['match'],'observed_time':0,
        'action':3,'buy':0,'item':0,'x':64,'y':64,'target_id':2,'target_type':'unit'},
        'execution':{'seq':1,'accepted':True},'decision_tick':0,'period_ticks':2,
        'issued_hero_orders':1,'purchase_receipts':[],'raw_actions':[]}
    return s


class DemonstrationTests(unittest.TestCase):
    def test_terminal_tick_is_removed_from_saved_demonstrations(self):
        with tempfile.TemporaryDirectory() as directory:
            writer=TeacherWriter(Path(directory)/'frames.gz',15)
            writer.append(frame())
            s=frame();s['seq']=2;s['time']=2/30
            s['demonstration']['decision_tick']=2
            s['demonstration']['command'].update(seq=2,observed_time=2/30)
            s['demonstration']['execution']['seq']=2
            writer.append(s);report=writer.close(2/30)
            with gzip.open(writer.path,'rt') as f:rows=[json.loads(line) for line in f]
            self.assertEqual(len(rows),1)
            self.assertEqual(report['frames'],1)
            self.assertEqual(rows[0]['observation']['time'],0)

    def test_exact_visible_target_and_previous_are_saved(self):
        s=frame();label,ref=demonstration_label(s)
        self.assertEqual(label['target'],1)
        self.assertEqual(ref['id'],2)
        self.assertEqual(label['target_offset'],[600/2048,0])
        with tempfile.TemporaryDirectory() as directory:
            writer=TeacherWriter(Path(directory)/'frames.gz',15)
            writer.append(s)
            second=copy.deepcopy(s);second['seq']=2;second['time']=2/30
            d=second['demonstration'];d['decision_tick']=2;d['command'].update(seq=2,observed_time=2/30)
            d['execution']['seq']=2
            writer.append(second);report=writer.close()
            with gzip.open(writer.path,'rt') as f:rows=[json.loads(line) for line in f]
            self.assertEqual(report['frames'],2)
            self.assertEqual(rows[1]['previous'],rows[0]['action'])

    def test_unknown_target_and_unavailable_item_are_rejected(self):
        s=frame();s['demonstration']['command']['target_id']=999
        with self.assertRaisesRegex(ValueError,'Target absent'):demonstration_label(s)
        s=frame();s['demonstration']['command'].update(action=10,item=5)
        s['legal']['actions'][10]=True
        with self.assertRaisesRegex(ValueError,'Item unavailable'):demonstration_label(s)

    def test_illegal_purchase_and_rejected_execution_are_rejected(self):
        s=frame();s['demonstration']['command']['buy']=4
        with self.assertRaisesRegex(ValueError,'Illegal'):demonstration_label(s)
        s=frame();s['demonstration']['execution']['accepted']=False
        with self.assertRaisesRegex(ValueError,'Executor'):demonstration_label(s)

    def test_missing_decision_and_off_clock_action_are_rejected(self):
        s=frame();s['demonstration']['decision_tick']=1;s['time']=1/30
        s['demonstration']['command']['observed_time']=s['time']
        with self.assertRaisesRegex(ValueError,'common clock'):demonstration_label(s)
        with tempfile.TemporaryDirectory() as directory:
            writer=TeacherWriter(Path(directory)/'frames.gz',15)
            writer.append(frame());s=frame();s['seq']=3
            try:
                with self.assertRaisesRegex(ValueError,'Missing'):writer.append(s)
            finally:writer.close()
