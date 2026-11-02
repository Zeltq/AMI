import copy
import unittest
import numpy as np
from fixtures import snapshot
from sf_duel.observations import encode
from sf_duel.telemetry import UNIT_COLUMNS,PROBE_SKILL_COLUMNS,INVENTORY_COLUMNS,unpack_snapshot


class TelemetryTests(unittest.TestCase):
    def test_columns_preserve_every_model_input_and_target(self):
        s=snapshot();s['probe']={'skill_slots':[{'slot':0,'name':'raze','level':1},{'slot':23}],
                                'inventory':[{'slot':1,'name':'item_tango'}]}
        s['creeps']=[dict(s['enemy'],id=13,type='creep')]
        wire=copy.deepcopy(s);wire['codec']='sf-columns-1'
        def pack(row,columns):return [row.get(key) for key in columns]
        for key in ('self','enemy'):wire[key]=pack(wire[key],UNIT_COLUMNS)
        for key in ('creeps','towers'):wire[key]=[pack(r,UNIT_COLUMNS) for r in wire[key]]
        wire['probe']['skill_slots']=[pack(r,PROBE_SKILL_COLUMNS) for r in wire['probe']['skill_slots']]
        wire['probe']['inventory']=[pack(r,INVENTORY_COLUMNS) for r in wire['probe']['inventory']]
        result=unpack_snapshot(wire)
        self.assertEqual(s,result)
        a,b=encode(s),encode(result)
        for key in a.arrays:np.testing.assert_array_equal(a.arrays[key],b.arrays[key])
        self.assertEqual(a.refs,b.refs)

    def test_bad_column_count_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'column count'):
            unpack_snapshot({'codec':'sf-columns-1','self':[]})
