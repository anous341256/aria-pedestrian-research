import hashlib
import json
from pathlib import Path
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from urllib.request import Request, urlopen
from urllib.error import HTTPError
import numpy as np
from core import warp_box, box_scores, question_state, summarize, pair_key
from serve import Review, handler_for


def example():
    source=[10,10,30,50];candidates=np.array([[10,10,30,50],[35,10,55,50],[100,10,120,50]])
    a=box_scores(source,source,candidates)
    b=box_scores(source,[35,10,55,50],candidates)
    return {'id':'q1','frame':1,'previous_frame':0,'source_index':0,'source_box':source,'source_score':.8,
            'candidates':[{'index':i,'box':box.tolist(),'score':.8}for i,box in enumerate(candidates)],
            'cohort':'random','review_order':[0,1],'methods':{'none':a,'rgb':a,'aria':b},
            'seconds':.1,'time_block':0,'angle_deg':2.,'rgb_status':'ok'}


def fixture(directory):
    data=json.dumps({'version':1,'queries':[example()]})
    (directory/'dataset.json').write_text(data,encoding='utf-8')
    (directory/'manifest.json').write_text(json.dumps({'dataset_sha256':hashlib.sha256(data.encode()).hexdigest()}),encoding='utf-8')


class GeometryTests(unittest.TestCase):
    def test_identity(self):
        np.testing.assert_allclose(warp_box([2,3,12,23],np.eye(3)),[2,3,12,23])
    def test_translation(self):
        np.testing.assert_allclose(warp_box([2,3,12,23],[[1,0,5],[0,1,-2],[0,0,1]]),[7,1,17,21])
    def test_perspective_division(self):
        H=np.array([[1,0,0],[0,1,0],[.01,0,1]])
        result=warp_box([10,10,20,20],H)
        np.testing.assert_allclose(result,[10/1.1,10/1.2,20/1.2,20/1.1])
    def test_invalid_projection(self):
        self.assertIsNone(warp_box([1,1,2,2],-np.eye(3)))
    def test_same_denominator(self):
        c=np.array([[5,5,15,15],[19,9,21,11]])
        d=box_scores([0,0,20,20],[0,0,20,20],c)
        self.assertEqual(d['order'],[0,1]);self.assertEqual(d['iou_order'],[0,1])
        self.assertAlmostEqual(d['normalized_distances'][1],10/np.sqrt(800))


class ProtocolTests(unittest.TestCase):
    def test_blind_deduplicated_queue(self):
        q=example();self.assertEqual(question_state(q,{})[0],0)
        self.assertEqual(question_state(q,{'q1:0':'no'})[0],1)
        self.assertEqual(question_state(q,{'q1:0':'no','q1:1':'yes'}),(None,'reviewed'))
    def test_correction_not_automatic_identity(self):
        q=example();labels={'q1:0':'no','q1:1':'no'}
        self.assertEqual(question_state(q,labels)[0],2)
        s=summarize([q],labels)['random']
        self.assertEqual(s['confirmed_positive_queries'],0)
        self.assertIsNone(s['methods']['aria']['positive_rank_median'])
    def test_pending_not_incorrect(self):
        s=summarize([example()],{})['random']['methods']['aria']
        self.assertEqual(s['pending'],1);self.assertIsNone(s['judged_accuracy'])
        self.assertEqual((s['all_sample_lower'],s['all_sample_upper']),(0,1))
    def test_paired_gain(self):
        s=summarize([example()],{'q1:0':'no','q1:1':'yes'})['random']
        self.assertEqual(s['paired']['none']['gains'],1)
        self.assertEqual(s['methods']['none']['positive_rank_median'],2)
        self.assertIsNone(s['paired']['none']['block_bootstrap_95'])
    def test_multiple_positive_kept(self):
        s=summarize([example()],{'q1:0':'yes','q1:1':'yes'})['random']
        self.assertEqual(s['multiple_positive_queries'],1)
        self.assertEqual(s['methods']['aria']['yes'],1)
        self.assertEqual(s['methods']['none']['yes'],1)
    def test_cohort_separation(self):
        q=example();q['cohort']='disagreement'
        s=summarize([q],{'q1:0':'yes'})
        self.assertEqual(s['random']['queries'],0);self.assertEqual(s['disagreement']['queries'],1)


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='rotation-review-test-')
        self.path=Path(self.temp.name);fixture(self.path);self.review=Review(self.path)
    def tearDown(self):self.temp.cleanup()
    def test_persist_correct_undo(self):
        r=self.review;r.change({'revision':0,'query_id':'q1','candidate':0,'value':'yes'})
        r.change({'revision':1,'query_id':'q1','candidate':0,'value':'no'})
        r.change({'revision':2},undo=True)
        self.assertEqual(Review(self.path).snapshot()[0],{'q1:0':'yes'})
        r.change({'revision':3},undo=True)
        self.assertEqual(r.snapshot()[0],{})
        with r.connect() as c:self.assertEqual(c.execute('SELECT COUNT(*) FROM events').fetchone()[0],4)
    def test_stale_rejected(self):
        r=self.review;r.change({'revision':0,'query_id':'q1','candidate':0,'value':'yes'})
        with self.assertRaises(RuntimeError):r.change({'revision':0,'query_id':'q1','candidate':1,'value':'yes'})
        self.assertEqual(r.snapshot()[1],1)
    def test_invalid_rejected(self):
        with self.assertRaises(ValueError):self.review.change({'revision':0,'query_id':'q1','candidate':99,'value':'yes'})
        self.assertEqual(self.review.snapshot()[1],0)
    def test_http_and_csrf(self):
        server=ThreadingHTTPServer(('127.0.0.1',0),handler_for(self.review));t=threading.Thread(target=server.serve_forever,daemon=True);t.start()
        base=f'http://127.0.0.1:{server.server_port}'
        try:
            with urlopen(base+'/api/state') as response:state=json.load(response)
            body=json.dumps({'revision':0,'query_id':'q1','candidate':0,'value':'unsure'}).encode()
            with self.assertRaises(HTTPError) as caught:urlopen(Request(base+'/api/label',data=body,headers={'Content-Type':'application/json'}))
            self.assertEqual(caught.exception.code,403)
            with urlopen(Request(base+'/api/label',data=body,headers={'Content-Type':'application/json','X-Review-Token':state['token']})) as response:self.assertEqual(response.status,200)
            self.assertEqual(self.review.snapshot()[0],{'q1:0':'unsure'})
        finally:server.shutdown();server.server_close();t.join()


if __name__=='__main__':unittest.main(verbosity=2)
