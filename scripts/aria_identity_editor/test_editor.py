import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from engine import Engine,Conflict,key,pair
from server import Project
from exporter import export


def make_engine():
    nodes=[];features=[]
    for frame in range(5):
        for i in range(2):
            x=20+i*100+frame
            nodes.append({'key':key(frame,i),'frame':frame,'index':i,'box':[x,20,x+30,80],'score':.9-i*.1})
            features.append([1,0] if i==0 else [0,1])
    e=Engine(nodes,np.array(features,float),np.tile(np.eye(3),(5,1,1)),np.eye(3))
    e.input={'nodes':nodes,'initial_labels':{},'frames':5,'fps':10,'width':256,'height':128,'video':'synthetic-no-video','config':e.config}
    return e


class EngineTests(unittest.TestCase):
    def test_appearance_geometry_and_determinism(self):
        e=make_engine();r=e.solve({});self.assertEqual(r['stats']['tracks'],2)
        self.assertEqual(r,e.solve({}));self.assertEqual(r['stats']['visible_boxes'],10)
        for f in range(5):self.assertEqual(r['track_ids'][key(f,0)],r['track_ids'][key(0,0)])
    def test_no_splits_and_cannot_reconnect_via_other_path(self):
        e=make_engine();a,b=key(1,0),key(2,0)
        r=e.solve({pair(a,b):'no'});self.assertNotEqual(r['track_ids'][a],r['track_ids'][b])
        self.assertEqual(r['stats']['visible_boxes'],10)
    def test_yes_overrides_score_and_displaces_old_edge(self):
        e=make_engine();a,b=key(1,0),key(2,1);r=e.solve({pair(a,b):'yes'})
        self.assertEqual(r['track_ids'][a],r['track_ids'][b]);self.assertTrue(any(l['a']==a and l['b']==b and l['kind']=='human_direct' for l in r['links']))
        for t in r['tracks']:self.assertEqual(len({e.nodes[n]['frame'] for n in t['nodes']}),len(t['nodes']))
    def test_contradictory_human_path_rejected(self):
        e=make_engine();a,b,c=key(0,0),key(1,0),key(2,0)
        with self.assertRaises(Conflict):e.solve({pair(a,b):'yes',pair(b,c):'yes',pair(a,c):'no'})
    def test_nonoverlapping_same_frame_positive_conflict(self):
        e=make_engine();a=key(0,0)
        with self.assertRaises(Conflict):e.solve({pair(a,key(1,0)):'yes',pair(a,key(1,1)):'yes'})
    def test_human_confirmed_duplicate_alias(self):
        e=make_engine();nodes=list(e.nodes.values());nodes[3]={**nodes[3],'box':nodes[2]['box'].copy()}
        e=Engine(nodes,e.features,e.R,e.K);a=key(0,0);r=e.solve({pair(a,key(1,0)):'yes',pair(a,key(1,1)):'yes'})
        self.assertEqual(r['stats']['duplicate_boxes'],1);self.assertEqual(r['aliases'][key(1,1)],key(1,0))
        self.assertEqual(r['track_ids'][key(1,1)],r['track_ids'][a])
    def test_unsure_not_constraint(self):
        e=make_engine();r=e.solve({pair(key(1,0),key(2,1)):'unsure'})
        self.assertEqual(r['track_ids'],e.solve({})['track_ids'])
    def test_invalid_rotation_explicit(self):
        e=make_engine();e.R[2]=np.nan;e._precompute()
        self.assertFalse(e.score(key(1,0),key(2,0))['rotation_valid'])
        self.assertFalse(e.score(key(0,0),key(3,0))['rotation_valid'])
    def test_gap_export_no_interpolation(self):
        e=make_engine();removed={key(2,0),key(3,0)};nodes=[n for n in e.nodes.values()if n['key']not in removed];feat=[e.features[e.index[n['key']]]for n in nodes]
        small=Engine(nodes,np.array(feat),e.R,e.K);small.input=e.input;r=small.solve({})
        self.assertEqual(r['track_ids'][key(1,0)],r['track_ids'][key(4,0)])
        with tempfile.TemporaryDirectory()as d:
            p=Path(d);(p/'input.json').write_text('{}');dest=Path(export(p,small,r,{},0));report=json.loads((dest/'manifest.json').read_text())
            self.assertEqual(report['visible_boxes_verified'],8);self.assertTrue(report['no_gap_interpolation_verified'])


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.directory=Path(self.temp.name);self.e=make_engine()
        blob=json.dumps(self.e.input).encode();(self.directory/'input.json').write_bytes(blob)
        (self.directory/'manifest.json').write_text(json.dumps({'input_sha256':hashlib.sha256(blob).hexdigest()}))
        self.p=Project(self.directory,self.e)
    def tearDown(self):self.temp.cleanup()
    def test_correction_immediate_export_and_undo(self):
        before=copy.deepcopy(self.p.result);a,b=key(1,0),key(2,0)
        result=self.p.apply({'revision':0,'a':a,'b':b,'value':'no'})
        self.assertGreater(result['changed_box_ids'],0);self.assertNotEqual(self.p.result['track_ids'][a],self.p.result['track_ids'][b])
        self.assertIsNone(result['export_error']);self.assertTrue((Path(result['export_path'])/'annotations.xml').exists())
        self.p.apply({'revision':1},True);self.assertEqual(self.p.result,before)
        reopened=Project(self.directory,self.e);self.assertEqual(reopened.result,before);self.assertEqual(reopened.revision,2)
    def test_conflict_does_not_write(self):
        a=key(0,0);self.p.apply({'revision':0,'a':a,'b':key(1,0),'value':'yes'})
        with self.assertRaises(Conflict):self.p.apply({'revision':1,'a':a,'b':key(1,1),'value':'yes'})
        self.assertEqual(self.p.revision,1);self.assertEqual(len(self.p.labels),1)
    def test_stale_revision(self):
        with self.assertRaises(Conflict):self.p.apply({'revision':9,'a':key(0,0),'b':key(1,0),'value':'yes'})
        self.assertEqual(self.p.revision,0)
    def test_imported_constraints_persist(self):
        # Separate fixture to avoid rewriting an initialized project.
        with tempfile.TemporaryDirectory()as d:
            e=make_engine();e.input['initial_labels']={pair(key(0,0),key(1,0)):'no'};p=Path(d);blob=json.dumps(e.input).encode()
            (p/'input.json').write_bytes(blob);(p/'manifest.json').write_text(json.dumps({'input_sha256':hashlib.sha256(blob).hexdigest()}))
            project=Project(p,e);self.assertNotEqual(project.result['track_ids'][key(0,0)],project.result['track_ids'][key(1,0)])
            self.assertEqual(list(project.origins.values()),['imported_review'])


if __name__=='__main__':unittest.main(verbosity=2)
