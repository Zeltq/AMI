import importlib.util
from pathlib import Path
import copy
import json
import unittest
from unittest.mock import patch
import tempfile

spec=importlib.util.spec_from_file_location('lane_conformance',Path(__file__).resolve().parents[1]/'gpu/scripts/lane_conformance.py')
lane=importlib.util.module_from_spec(spec);spec.loader.exec_module(lane)


def fixture():
    case=lane.generate(30,42)['cases'][26]
    case['duration_ticks']=1;case['orders']=[]
    units=[dict(u,health=u.get('health',100),mana=10,z=0,level=u.get('level',3),alive=True,last_hits=0,denies=0,raze_stacks=0)for u in case['units']]
    native={'id':case['id'],'orders':[],'frames':[{'tick':i,'units':copy.deepcopy(units)}for i in range(2)]}
    return case,native,copy.deepcopy(native)


class LaneConformanceTests(unittest.TestCase):
    def test_raze_geometry_capture_requires_measured_hulls_and_spell_values(self):
        case,native,_=fixture();case['capture_raze_geometry']=True
        self.assertFalse(lane.admission(case,native)['passed'])
        for unit in native['frames'][0]['units']:
            unit['hull_radius']=24 if unit['kind']=='hero'else 16
            if unit['kind']=='hero':unit['raze_geometry']=[dict(distance=d,radius=250)for d in(200,450,700)]
        self.assertTrue(lane.admission(case,native)['passed'])
        native['frames'][0]['units'][0]['raze_geometry'][0]['radius']=float('nan')
        self.assertFalse(lane.admission(case,native)['passed'])

    def test_direct_event_capture_requires_actual_actor_event_triplets(self):
        case,native,_=fixture();case['capture_attack_events']=True;case['orders']=[dict(tick=0,actor='radiant',action=3,target='victim')];native['orders']=copy.deepcopy(case['orders'])
        self.assertFalse(lane.admission(case,native)['passed'])
        native['attack_events']=[dict(actor='radiant',event=event)for event in('start','release','landed')for _ in range(3)]
        self.assertTrue(lane.admission(case,native)['passed'])
        native['attack_events'][0]['actor']='dire'
        self.assertFalse(lane.admission(case,native)['passed'])

    def test_attachment_capture_cannot_admit_missing_or_nonfinite_points(self):
        case,native,_=fixture();case['capture_attachments']=True
        self.assertFalse(lane.admission(case,native)['passed'])
        for frame in native['frames']:
            for unit in frame['units']:
                if unit['kind']=='hero':unit['attachments']={key:dict(x=1.,y=2.,z=3.)for key in('attach_attack1','attach_attack2')}
        self.assertTrue(lane.admission(case,native)['passed'])
        native['frames'][1]['units'][0]['attachments']['attach_attack1']['z']=float('nan')
        self.assertFalse(lane.admission(case,native)['passed'])

    def test_manifest3000_reproducible_unique_and_balanced_by_family(self):
        a=lane.generate(3000,42);b=lane.generate(3000,42)
        self.assertEqual(a,b);lane.validate_manifest(a)
        self.assertEqual(len({c['id']for c in a['cases']}),3000)
        for family in (*lane.FAMILIES,'natural_wave'):
            counts=[sum(c['family']==family and c['actor_team']==t for c in a['cases'])for t in (2,3)]
            self.assertEqual(counts[0],counts[1]);self.assertGreater(counts[0],0)
        self.assertEqual({u['kind']for c in a['cases']for u in c['units']},{'hero','melee','ranged','siege'})
        self.assertEqual({u['level']for c in a['cases']if c['mode']=='controlled'for u in c['units']if u['kind']=='hero'},{1,3,6})
        b['cases'][0]['duration_ticks']+=1
        with self.assertRaisesRegex(ValueError,'checksum'):lane.validate_manifest(b)

    def test_missing_native_frame_and_late_order_are_invalid_not_environment_failure(self):
        case,native,gpu=fixture();native['frames'].pop()
        self.assertEqual(lane.compare_case(case,native,gpu)['status'],'invalid_native')
        case,native,gpu=fixture();case['orders']=[{'tick':0,'action':4,'actor':'radiant'}]
        native['orders']=[{'tick':1,'action':4,'actor':'radiant'}]
        self.assertFalse(lane.admission(case,native)['passed'])

    def test_wrong_target_is_detected_even_when_hp_and_positions_match(self):
        case,native,gpu=fixture()
        self.assertEqual(lane.compare_case(case,native,gpu)['status'],'pass')
        gpu['frames'][1]['units'][0]['attack_target']='victim'
        report=lane.compare_case(case,native,gpu)
        self.assertEqual(report['status'],'mismatch');self.assertEqual(report['first_divergence']['field'],'attack_target')

    def test_diagonal_motion_uses_euclidean_tolerance_and_missing_hp_cannot_pass(self):
        case,native,gpu=fixture();gpu['frames'][1]['units'][0]['x']+=12;gpu['frames'][1]['units'][0]['y']+=12
        self.assertEqual(lane.compare_case(case,native,gpu)['first_divergence']['field'],'position')
        case,native,gpu=fixture();del gpu['frames'][1]['units'][0]['health']
        self.assertEqual(lane.compare_case(case,native,gpu)['status'],'invalid_data')

    def test_natural_birth_identity_is_kept_after_units_cross(self):
        a={'id':'n1','team':2,'kind':'melee','x':0.,'y':0.,'alive':True}
        b=dict(a,id='n2',x=50.)
        native={'frames':[{'tick':0,'units':[a,b]},{'tick':1,'units':[dict(a,x=50),dict(b,x=0)]}]}
        gpu={'id':'case','frames':[{'tick':0,'units':[dict(b,id='g2'),dict(a,id='g1')]},
                                  {'tick':1,'units':[dict(a,id='g1',x=50),dict(b,id='g2',x=0)]}]}
        mapped,evidence=lane.map_natural_births(native,gpu)
        self.assertEqual([u['id']for u in mapped['frames'][1]['units']],['n1','n2'])
        self.assertEqual(len(evidence),1)
        self.assertEqual(gpu['frames'][0]['units'][0]['id'],'g2')

    def test_fixed_roll_is_explicit_and_does_not_modify_natural_waves(self):
        manifest=lane.generate(3000,42,'fixed_base_roll')
        self.assertTrue(all(c['damage_mode']==('stock'if c['mode']=='natural'else'fixed_base_roll')for c in manifest['cases']))
        case,native,gpu=fixture();case['mode']='natural'
        self.assertIn('missing declared genuine mid waves',lane.admission(case,native)['issues'])

    def test_fixed_roll_requires_measured_mean_preservation(self):
        case,native,gpu=fixture();case['damage_mode']='fixed_base_roll'
        self.assertFalse(lane.admission(case,native)['passed'])
        for row in native['frames'][0]['units']:
            row['damage']=49
            row['damage_control']=dict(mean_before=49,mean_after=49,reported_min=49,reported_max=49)
        self.assertTrue(lane.admission(case,native)['passed'])
        native['frames'][0]['units'][0]['damage']=79
        native['frames'][0]['units'][0]['damage_control']['mean_after']=79
        self.assertFalse(lane.admission(case,native)['passed'])

    def test_warm_start_slices_real_precombat_history_and_requires_provenance(self):
        case,native,gpu=fixture();case.update(mode='natural',duration_ticks=2)
        for row in native['frames'][0]['units']:row['max_health']=row['health']
        units=[u for u in native['frames'][0]['units']if u['kind']=='hero']
        creeps=[dict(units[0],id=f'creep-{i}',kind='melee')for i in range(8)]
        native['frames']=[dict(tick=i,dota_time=-1 if i==0 else i/30,units=copy.deepcopy(units+(creeps if i else[])))for i in range(3)]
        case['units']=[u for u in case['units']if u['kind']=='hero'];case['expected_waves']=1
        source=dict(complete=True,capture_passed=True,cases=[case],records=[native])
        result=lane.warm_start(source,'abc123',1,1)
        self.assertEqual([f['tick']for f in result['records'][0]['frames']],[0,1])
        self.assertEqual(result['records'][0]['frames'][0]['units'],native['frames'][1]['units'])
        self.assertTrue(lane.admission(result['cases'][0],result['records'][0])['passed'])
        result['records'][0].pop('derived_from_native_sha256')
        self.assertFalse(lane.admission(result['cases'][0],result['records'][0])['passed'])
        native['frames'][2]['units'][0]['attack_target']='creep-0'
        with self.assertRaisesRegex(ValueError,'before an observed attack'):lane.warm_start(source,'abc123',1,1)
        native['frames'][2]['units'][0]['health']-=10
        contact=lane.warm_start(source,'abc123',1,1,through_contact=True)
        self.assertTrue(contact['cases'][0]['warm_start']['through_contact'])
        self.assertEqual(contact['records'][0]['frames'][1]['units'],native['frames'][2]['units'])
        native['frames'][1]['units'][0]['attack_target']='creep-0'
        with self.assertRaisesRegex(ValueError,'before an observed attack'):lane.warm_start(source,'abc123',1,1,through_contact=True)

    def test_level3_requires_production_passive_and_raze_build(self):
        case,native,_=fixture()
        for row in case['units']:
            if row['kind']=='hero':row['level']=3
        self.assertFalse(lane.admission(case,native)['passed'])
        for row in native['frames'][0]['units']:
            if row['kind']=='hero':row['abilities']=dict(nevermore_shadowraze1=2,nevermore_shadowraze2=2,nevermore_shadowraze3=2,nevermore_requiem=0,nevermore_dark_lord=1)
        self.assertTrue(lane.admission(case,native)['passed'])
        native['frames'][0]['units'][0]['abilities']['nevermore_dark_lord']=0
        self.assertFalse(lane.admission(case,native)['passed'])

    def test_partial_reference_preserves_rejected_case_and_cannot_promote_incomplete_capture(self):
        case,native,_=fixture();bad=copy.deepcopy(case);bad['id']='bad'
        row=copy.deepcopy(native);row['id']='bad';row['frames'].pop()
        source={'complete':True,'error':None,'cases':[case,bad],'records':[native,row],'capture_passed':False}
        admitted=lane.admitted_subset(source)
        self.assertEqual([c['id']for c in admitted['cases']],[case['id']])
        self.assertEqual(admitted['excluded_cases'][0]['id'],'bad')
        self.assertFalse(source['capture_passed'])
        source['complete']=False
        with self.assertRaises(ValueError):lane.admitted_subset(source)

    def test_observed_release_and_credit_events_are_reported_without_inventing_damage(self):
        case,native,gpu=fixture();unit=gpu['frames'][1]['units'][0]
        unit.update(attack_hand=1,last_hits=1,health=90)
        events=lane.observed_events(gpu)
        self.assertEqual({e['event']for e in events},{'observed_attack_release','last_hits','observed_hp_drop'})
        self.assertTrue(all(e['tick']==1 for e in events))

    def test_public_reference_requires_both_teams_at_each_tick_and_preserves_target_identity(self):
        public=[]
        for tick in (100,101):
            for team in (2,3):
                own={'id':'radiant'if team==2 else'dire','name':'npc_dota_hero_nevermore','kind':'hero','team':team,'x':0,'y':team*100,'health':100}
                creep={'id':f'bot-{team}-1','name':f'npc_dota_creep_{team}_melee','kind':'melee','team':team,'x':team*10,'y':0,'health':20-(tick-100)*5}
                if team==2:creep['attack_target_ref']={'name':'npc_dota_creep_3_melee','team':3,'x':30,'y':0}
                public.append({'tick':tick,'team':team,'time':-1+(tick-100)/30,'game_time':tick/30,'units':[own,creep]})
        case={'id':'case','duration_ticks':1}
        r=lane.public_natural_reference(case,{'orders':[]},public)
        self.assertEqual(r['frames'][0]['units'][1]['attack_target'],'bot-3-1')
        self.assertEqual([u['health']for u in r['frames'][1]['units']if u['kind']=='melee'],[15,15])
        self.assertIn('aggro_target',r['unavailable_fields'])
        with self.assertRaisesRegex(ValueError,'Missing paired'):lane.public_natural_reference(case,{'orders':[]},public[:-1])

    def test_profile_covers_core_dimensions_and_deny_boundary_both_sides(self):
        manifest=lane.representative_profile(lane.generate(3000,42,'fixed_base_roll'))
        self.assertEqual(len(manifest['cases']),256);lane.validate_manifest(manifest)
        boundary=[c for c in manifest['cases']if c['id'].startswith('deny-boundary')]
        self.assertEqual(len(boundary),18)
        self.assertEqual({u['health_fraction']for c in boundary for u in c['units']if u['id']=='victim'},{.49,.5,.51})
        self.assertEqual({c['actor_team']for c in boundary},{2,3})


class AtomicSnapshotTests(unittest.TestCase):
    def test_transient_windows_reader_retries_the_same_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'progress.json'
            path.write_text('{"old":true}')
            replace=Path.replace
            attempts=[]
            def held_once(source,destination):
                attempts.append(source)
                if len(attempts)<3:raise PermissionError('reader holds destination')
                return replace(source,destination)
            with patch.object(Path,'replace',held_once),patch.object(lane.time,'sleep'):
                lane.atomic(path,{'new':True})
            self.assertEqual(json.loads(path.read_text()),{'new':True})
            self.assertEqual(len(attempts),3)

    def test_persistent_failure_keeps_the_previous_snapshot_and_is_bounded(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'progress.json';path.write_text('{"old":true}')
            with patch.object(Path,'replace',side_effect=PermissionError('held')) as replace,patch.object(lane.time,'sleep'):
                with self.assertRaises(PermissionError):lane.atomic(path,{'new':True})
            self.assertEqual(replace.call_count,5)
            self.assertEqual(json.loads(path.read_text()),{'old':True})


if __name__=='__main__':unittest.main()
