import importlib.util
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'gpu/scripts'))
spec=importlib.util.spec_from_file_location('fit_sf_projectile_profile',Path(__file__).resolve().parents[1]/'gpu/scripts/fit_sf_projectile_profile.py')
fit=importlib.util.module_from_spec(spec);spec.loader.exec_module(fit)


class ProjectileProfileTests(unittest.TestCase):
    def test_same_phase_conflicting_poses_are_rejected_instead_of_averaged(self):
        with self.assertRaisesRegex(ValueError,'not determined'):
            fit.curve([[.4,0.,0.,0.],[.40001,100.,0.,0.],[.5,20.,0.,0.]])

    def test_curve_requires_distinct_covered_phases_and_preserves_height(self):
        with self.assertRaisesRegex(ValueError,'Insufficient'):fit.curve([[.4,1.,2.,3.]])
        curve,spread=fit.curve([[.4,1.,2.,3.],[.40001,1.,2.,3.],[.5,4.,5.,6.]])
        self.assertEqual([row[3]for row in curve],[3.,6.]);self.assertEqual(spread,0.)


if __name__=='__main__':unittest.main()
