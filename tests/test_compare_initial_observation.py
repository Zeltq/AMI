import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'gpu/scripts'))
from compare_initial_observation import observation_layout,field_summary
from sf_duel.spec import COMBAT_UNIT_FIELDS,COMBAT_PROJECTILE_FIELDS


class ObservationComparisonTests(unittest.TestCase):
    def test_ggd_layout_includes_both_combat_extensions(self):
        legacy,old_width=observation_layout()
        ggd,width=observation_layout('ggd-v2')
        self.assertEqual(old_width,37324)
        self.assertNotIn('hero_combat',legacy)
        self.assertEqual(ggd['hero_combat'],(2,2*len(COMBAT_UNIT_FIELDS)))
        self.assertEqual(ggd['projectile_combat'],(512,2*len(COMBAT_PROJECTILE_FIELDS)))
        self.assertEqual(width,51280)
        self.assertEqual(sum(int(np.prod(s))for s in ggd.values()),width)

    def test_one_percent_is_of_the_declared_field_scale(self):
        n=len(COMBAT_UNIT_FIELDS)
        a=np.zeros((1,2*n),np.float32);a[:,n:]=1
        b=a.copy();b[0,0]=.011;b[0,1]=.009
        rows=field_summary(np.abs(a-b),COMBAT_UNIT_FIELDS,a,b)
        self.assertEqual([r['field']for r in rows],['attack_target_dx'])
        self.assertAlmostEqual(rows[0]['max_field_error'],.011*2048,places=4)
        self.assertEqual(rows[0]['above_one_percent'],1)


if __name__=='__main__':unittest.main()
