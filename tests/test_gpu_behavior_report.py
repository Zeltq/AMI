import importlib.util
from pathlib import Path
import unittest
spec=importlib.util.spec_from_file_location('behavior_report',Path(__file__).resolve().parents[1]/'gpu/scripts/behavior_report.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

class BehaviorReportTests(unittest.TestCase):
 def test_completed_native_movement_is_idle_and_is_not_an_active_blocked_order(self):
  m=module.Motion();m.order=(83,0)
  for i in range(61):m.sample(i/5,0,0,True,moving=False,idle=True)
  self.assertEqual(m.maximum,0);self.assertEqual(m.stuck,0)
  self.assertAlmostEqual(m.idle_seconds,12)
  m.sample(12.2,0,0,True,moving=True,idle=False)
  self.assertEqual(m.idle_run,0)
 def test_mid_includes_negative_pregame_and_ignores_dead(self):
  m=module.Motion();m.sample(-60,0,0,False);self.assertIsNone(m.first_mid)
  m.sample(-59,0,0,True);m.sample(-58,5000,5000,True);self.assertEqual(m.first_mid,-59)
 def test_stuck_requires_movement_and_excludes_blocked_and_gaps(self):
  m=module.Motion()
  for i in range(101):m.sample(i/5,3000,0,True)
  self.assertEqual(m.maximum,0);m.order=(4000,0)
  for i in range(101,161):m.sample(i/5,3000,0,True)
  self.assertGreater(m.stuck,10);self.assertAlmostEqual(m.maximum,12)
  m.sample(32.4,3000,0,True,True);self.assertEqual(m.run,0)
  m.sample(35,3000,0,True);self.assertEqual(m.run,0);self.assertEqual(m.gaps,1)
 def test_reached_goal_or_real_displacement_is_not_stuck(self):
  m=module.Motion();m.order=(3010,0)
  for i in range(101):m.sample(i/5,3000,0,True)
  self.assertEqual(m.maximum,0);m.order=(4000,0)
  for i in range(101,201):m.sample(i/5,3000+2*(i-100),0,True)
  self.assertEqual(m.maximum,0)
 def test_passive_continue_at_spawn_is_reported_without_inventing_blocked_movement(self):
  m=module.Motion()
  for i in range(-100,76):m.sample(i/5,6000,6000,True)
  self.assertEqual(m.maximum,0);self.assertEqual(m.stuck,0)
  self.assertAlmostEqual(m.passive_max,15);self.assertAlmostEqual(m.passive_seconds,15)
  m.sample(15.2,0,0,True);self.assertEqual(m.passive_run,0);self.assertEqual(m.first_mid,15.2)
