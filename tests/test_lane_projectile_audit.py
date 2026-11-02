import copy
import importlib.util
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'gpu/scripts'))
spec=importlib.util.spec_from_file_location('lane_projectile_audit',Path(__file__).resolve().parents[1]/'gpu/scripts/lane_projectile_audit.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)


def fixture():
    units=[dict(id='radiant',kind='hero',name='npc_dota_hero_nevermore',team=2,x=0.,y=0.,z=0.,facing=0.,health=500.,max_health=500.,level=1,last_attack=0.,attack_hand=0.),
           dict(id='dire',kind='hero',name='npc_dota_hero_nevermore',team=3,x=0.,y=1000.,z=0.,facing=0.,health=500.,max_health=500.,level=1,last_attack=0.,attack_hand=0.),
           dict(id='victim',kind='melee',name='creep',team=3,x=64.,y=0.,z=0.,facing=0.,health=100.,max_health=100.,last_attack=0.,attack_hand=0.)]
    orders=[dict(tick=0,actor='radiant',action=3,target='victim')]
    case=dict(id='test',mode='controlled',family='attack',actor_team=2,duration_ticks=3,damage_mode='stock',units=copy.deepcopy(units),orders=orders)
    nf=[];gf=[];tracking=[]
    for tick in range(4):
        nu=copy.deepcopy(units);gu=copy.deepcopy(units)
        if tick>=1:nu[0]['last_attack']=100+1/30;gu[0]['attack_hand']=1.
        if tick>=3:nu[2]['health']=80.
        if tick>=2:gu[2]['health']=80.
        nf.append(dict(tick=tick,game_time=100+tick/30,units=nu))
        gs=[dict(source='radiant',target='victim',x=21.,y=-20.,z=None)]if tick==1 else[]
        gf.append(dict(tick=tick,units=gu,projectiles=gs))
        if tick in(1,2):tracking.append(dict(tick=3000+tick,targets=[dict(name='creep',team=3,x=64.,y=0.,available=True,
            shots=[dict(x=20.*tick,y=30.-10*tick,z=60.-10*tick,is_attack=True,caster_team=2,caster_name='npc_dota_hero_nevermore')])]))
    native=dict(complete=True,capture_passed=True,cases=[case],records=[dict(id='test',frames=nf,orders=orders)],tracking_projectile_samples=tracking)
    gpu=dict(records=[dict(id='test',frames=gf,orders=orders)])
    return native,gpu


class ProjectileAuditTests(unittest.TestCase):
    def test_timing_uses_direct_release_and_rejects_missing_events(self):
        n,g=fixture();nr=n['records'][0]
        nr['attack_events']=[dict(actor='radiant',event='start',tick=0),dict(actor='radiant',event='release',tick=1)]
        for f in g['records'][0]['frames']:f['units'][0]['windup']=1. if f['tick']==0 else 0.
        r=audit.timing_audit(n,g);self.assertTrue(r['timing_passed'])
        self.assertEqual(r['cases'][0]['release_source'],'direct callbacks')
        nr['attack_events'].append(dict(actor='radiant',event='release',tick=3))
        self.assertFalse(audit.timing_audit(n,g)['timing_passed'])

    def test_timing_reports_a_two_tick_shift_and_cannot_pass_empty_capture(self):
        n,g=fixture();nr=n['records'][0]
        nr['attack_events']=[dict(actor='radiant',event='release',tick=3)]
        self.assertFalse(audit.timing_audit(n,g)['timing_passed'])
        n,g=fixture()
        for f in n['records'][0]['frames']:f['units'][0]['last_attack']=0.;f['units'][2]['health']=100.
        for f in g['records'][0]['frames']:f['units'][0]['attack_hand']=0.;f['units'][2]['health']=100.
        r=audit.timing_audit(n,g);self.assertFalse(r['timing_passed']);self.assertFalse(r['cases'][0]['adequate'])

    def test_pairs_by_identity_and_preserves_signed_hand_difference(self):
        n,g=fixture();r=audit.audit(n,g)['cases'][0]['shots'][0]
        self.assertEqual(r['impact_delta_ticks'],-1)
        self.assertEqual(r['native_relative_first_visible']['lateral'],20.)
        self.assertEqual(r['gpu_relative_first_visible']['lateral'],-20.)
        self.assertEqual(r['first_visible_absolute_lateral_difference'],0.)

    def test_missing_capture_and_duplicate_native_shot_are_invalid(self):
        n,g=fixture();g['records'][0]['frames'][0].pop('projectiles')
        with self.assertRaisesRegex(ValueError,'Missing GPU'):audit.audit(n,g)
        n,g=fixture();n['tracking_projectile_samples'].append(copy.deepcopy(n['tracking_projectile_samples'][0]))
        with self.assertRaisesRegex(ValueError,'Ambiguous duplicate'):audit.audit(n,g)

    def test_target_ambiguity_and_wrong_command_are_rejected(self):
        n,g=fixture();frame=n['records'][0]['frames'][1];frame['units'].append(dict(frame['units'][2],id='other'))
        with self.assertRaisesRegex(ValueError,'ambiguous native'):audit.audit(n,g)
        n,g=fixture();g['records'][0]['orders']=[]
        with self.assertRaisesRegex(ValueError,'command log'):audit.audit(n,g)

    def test_unfinished_impact_is_censored_and_z_not_invented(self):
        n,g=fixture();n['records'][0]['frames'][3]['units'][2]['health']=100.
        r=audit.audit(n,g)['cases'][0]['shots'][0]
        self.assertTrue(r['right_censored']);self.assertIsNone(r['impact_delta_ticks'])
        self.assertIsNone(r['gpu_relative_first_visible']['height'])


if __name__=='__main__':unittest.main()
