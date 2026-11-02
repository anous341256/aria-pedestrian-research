import copy
import json
import sqlite3
import time
import unittest
from unittest.mock import patch
from engine import Conflict,key,pair
from test_editor import StoreTests


class SegmentTests(StoreTests):
    def risk(self):
        # The fixture is otherwise perfectly unambiguous; make exactly one link risky.
        next(l for l in self.p.result['links'] if l['kind']=='automatic')['cost']=.49
        self.p.segments.revision=-1
        self.p.segments.refresh()
        return self.p.segments.tasks[0]

    def submit(self,t,action='approve',**kwargs):
        return self.p.apply({'revision':self.p.revision,'task_id':t['id'],'fingerprint':t['fingerprint'],
                             'action':action,'watched_seconds':2.5,**kwargs},segment=True)

    def test_filter_not_all_links(self):
        self.assertEqual(self.p.segments.page()['total'],0)
        t=self.risk();self.assertEqual(len(t['risk_keys']),1)
        self.assertLessEqual(len(self.p.segments.page()['items']),20)

    def test_batch_single_solve_undo_export_and_scope(self):
        t=self.risk();before=copy.deepcopy(self.p.labels)
        with patch.object(self.e,'solve',wraps=self.e.solve) as solve:
            result=self.submit(t);self.assertEqual(solve.call_count,1)
        self.assertEqual(self.p.revision,1)
        self.assertEqual(len(self.p.labels),len(t['nodes'])-1)
        self.assertTrue(all('_d000' in k and '_d001' not in k for k in self.p.labels))
        exported=json.loads((self.directory/'exports/rev_000001/segment_reviews.json').read_text())
        self.assertEqual(exported[0]['scope'],[t['start'],t['end']])
        self.assertEqual(exported[0]['nodes'],t['nodes'])
        self.assertEqual(self.p.segments.metadata()['metrics']['resolved_risks'],1)
        self.assertEqual(self.p.segments.page()['total'],0)
        self.p.apply({'revision':1},True);self.assertEqual(self.p.labels,before)
        self.assertEqual(self.p.segments.records(),[])

    def test_preview_does_not_write_and_obeys_constraints(self):
        t=self.risk();before=self.p.read_labels()
        candidates=self.p.segments.candidates(t['id'],t['anchor'])
        self.assertLessEqual(len(candidates['alternatives']),3)
        for c in candidates['alternatives']:
            self.p.segments.constraints_for_path(c['nodes'])
            self.assertEqual(len(c['nodes']),len(set(c['nodes'])))
            self.assertTrue(all(self.e.nodes[n]['frame']<=candidates['end'] for n in c['nodes']))
        self.assertEqual(self.p.read_labels(),before)

    def test_defer_reopen_no_labels_and_undo(self):
        t=self.risk();self.submit(t,'defer')
        self.assertEqual(self.p.labels,{})
        self.assertEqual(self.p.segments.page()['total'],0)
        self.assertEqual(len(self.p.segments.page(deferred=True)['items']),1)
        opened=self.p.segments.reopen(t['id'])
        self.assertEqual(self.p.revision,1)
        self.submit(opened,'approve')
        self.assertEqual(self.p.segments.deferred_records(),[])
        self.p.apply({'revision':2},True)
        self.assertEqual(len(self.p.segments.deferred_records()),1)

    def test_conflict_batch_atomic_and_stale_fingerprint(self):
        t=self.risk();bad={**t,'fingerprint':'stale'}
        with self.assertRaises(Conflict):self.submit(bad)
        self.assertEqual(self.p.revision,0)
        self.p.labels[pair(key(0,0),key(1,0))]='no'
        with self.assertRaises(Conflict):self.submit(t)
        self.assertEqual(self.p.revision,0)

    def test_existing_label_survives_batch_undo(self):
        self.p.apply({'revision':0,'a':key(0,0),'b':key(1,0),'value':'yes'})
        t=self.risk();before=self.p.read_labels()[0]
        self.submit(t);self.p.apply({'revision':2},True)
        self.assertEqual(self.p.labels,before)


if __name__=='__main__':unittest.main()
