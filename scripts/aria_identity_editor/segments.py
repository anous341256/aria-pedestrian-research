"""Risk episodes and bounded, constraint-aware alternative paths (no training)."""
from collections import defaultdict
import hashlib
import json
import math
from engine import pair, Components, Conflict


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:20]


class SegmentService:
    def __init__(self, project):
        self.p = project
        self.revision = -1
        self.tasks = []
        self.by_id = {}
        self.previews = {}
        self.reopened = {}

    def records(self):
        with self.p.connect() as c:
            rows = c.execute('SELECT id,payload FROM events WHERE payload IS NOT NULL AND undo_of IS NULL AND id NOT IN (SELECT undo_of FROM events WHERE undo_of IS NOT NULL) ORDER BY id').fetchall()
        return [{'revision': i, **json.loads(payload)['review']} for i, payload in rows if json.loads(payload).get('review')]

    def refresh(self):
        p = self.p
        if self.revision == p.revision:
            return
        e, result = p.engine, p.result
        deferred = {k for r in self.deferred_records() for k in r.get('risk_keys', [])}
        frame = lambda n: e.nodes[n]['frame']
        per_track = defaultdict(list)
        links = {(l['a'], l['b']): l for l in result['links']}
        for q in result['queue']:
            if pair(q['a'], q['b']) in deferred:
                continue
            l = links.get((q['a'], q['b']))
            reasons = []
            if l:
                if l['cost'] >= .45: reasons.append('关联代价较高')
                if l.get('margin', 1) <= .10: reasons.append('竞争候选接近')
                if l['gap'] > 1: reasons.append('跨越缺失帧')
                if not l['rotation_valid']: reasons.append('旋转数据无效')
            else:
                reasons.append('断轨端点有续接候选')
            if reasons:
                per_track[result['track_ids'][q['a']]].append({**q, 'reasons': reasons})
        track_map = {t['id']: t for t in result['tracks']}
        width = max(2, round(e.input['fps'] * 5))
        overlap = max(1, round(e.input['fps']))
        fresh = []
        for tid, risks in per_track.items():
            risks.sort(key=lambda q: (frame(q['a']), q['a'], q['b']))
            remaining = risks[:]
            previous_end = -100000
            while remaining:
                first = remaining[0]
                start = max(0, frame(first['a']) - 2 * overlap)
                if start <= previous_end: start = max(0, previous_end - overlap + 1)
                end = min(e.frame_count - 1, start + width - 1)
                owned = [q for q in remaining if start <= frame(q['a']) <= end]
                if not owned:  # Defensive progress for unusual frame rates.
                    start = frame(first['a']); end = min(e.frame_count - 1, start + width - 1); owned = [first]
                keys = {pair(q['a'], q['b']) for q in owned}
                remaining = [q for q in remaining if pair(q['a'], q['b']) not in keys]
                ns = [n for n in track_map[tid]['nodes'] if start <= frame(n) <= end]
                task = {'id': digest([first['a'], first['b'], start, end]), 'anchor': first['a'],
                        'start': start, 'end': end, 'nodes': ns, 'risks': owned,
                        'risk_keys': sorted(keys), 'risk': max(q['risk'] for q in owned),
                        'reasons': sorted({s for q in owned for s in q['reasons']})}
                task['fingerprint'] = digest([ns, task['risk_keys']])
                fresh.append(task); previous_end = end
        # Keep surviving episode order/IDs; reuse a task when its risk ownership and window overlap.
        ordered = []
        for old in self.tasks:
            found = next((t for t in fresh if t['anchor'] == old['anchor'] and set(t['risk_keys']) & set(old['risk_keys'])), None)
            if found:
                found['id'] = old['id']; ordered.append(found); fresh.remove(found)
        fresh.sort(key=lambda t: (-t['risk'], t['start'], t['anchor']))
        self.tasks = ordered + fresh
        self.by_id = {t['id']: t for t in self.tasks}
        for task_id, snapshot in self.reopened.items():
            if task_id not in self.by_id:self.by_id[task_id]=self.rebuild_snapshot(snapshot)
        self.previews.clear()
        self.revision = p.revision

    def deferred_records(self):
        latest={}
        for r in self.records():latest[r['task_id']]=r
        return [r for r in latest.values() if r['action']=='defer']

    def rebuild_snapshot(self,snapshot):
        task=dict(snapshot)
        data=self.window(task['start'],task['end'],task['anchor'])
        task['nodes']=data['nodes'];task['risks']=[]
        task['fingerprint']=digest([task['nodes'],task['risk_keys']])
        return task

    def reopen(self,task_id):
        with self.p.lock:
            self.refresh()
            record=next((r for r in reversed(self.deferred_records()) if r['task_id']==task_id),None)
            if not record:raise Conflict('此稍后处理项已变化，请刷新稍后列表。')
            self.reopened[task_id]=record['task_snapshot']
            self.by_id[task_id]=self.rebuild_snapshot(record['task_snapshot'])
            return self.decorate(self.by_id[task_id])

    def metadata(self):
        p = self.p
        with p.lock:
            self.refresh()
            with p.connect() as c:
                can_undo = c.execute('SELECT 1 FROM events WHERE undo_of IS NULL AND id NOT IN (SELECT undo_of FROM events WHERE undo_of IS NOT NULL) LIMIT 1').fetchone() is not None
            records = self.records()
            return {'token': p.token, 'revision': p.revision, 'fps': p.engine.input['fps'],
                    'frames': p.engine.frame_count, 'width': p.engine.input['width'], 'height': p.engine.input['height'],
                    'can_undo': can_undo, 'export_path': p.export_path, 'export_error': p.export_error,
                    'pending': len(self.tasks), 'stats': p.result['stats'],
                    'metrics': {'review_actions': len(records), 'watched_seconds': round(sum(r.get('watched_seconds', 0) for r in records), 1),
                                'confirmed_relations': len({k for r in records if r['action'] != 'defer' for k in r.get('confirmed_pairs', [])}),
                                'resolved_risks': len({k for r in records for k in r.get('resolved_risk_keys', [])}),
                                'deferred_episodes': len(self.deferred_records())}}

    def page(self, after=None, deferred=False):
        with self.p.lock:
            self.refresh()
            if deferred:
                records = self.deferred_records()
                offset = int(after or 0)
                return {'items': records[offset:offset+20], 'next': str(offset+20) if offset+20 < len(records) else None, 'total': len(records)}
            i = next((i+1 for i,t in enumerate(self.tasks) if t['id'] == after), 0)
            items = self.tasks[i:i+20]
            return {'items': [{k:v for k,v in t.items() if k not in ('nodes','risks','risk_keys')} for t in items],
                    'next': items[-1]['id'] if i+20 < len(self.tasks) and items else None, 'total': len(self.tasks)}

    def detail(self, task_id):
        with self.p.lock:
            self.refresh()
            if task_id not in self.by_id: raise Conflict('该片段已经变化或退出待办，请刷新片段列表。')
            return self.decorate(self.by_id[task_id])

    def decorate(self, task):
        p = self.p
        nodes = [{**n, 'track_id': p.result['track_ids'][n['key']]} for n in p.engine.nodes.values()
                 if task['start'] <= n['frame'] <= task['end'] and p.result['aliases'][n['key']] == n['key']]
        return {**task, 'revision': p.revision, 'detections': nodes}

    def window(self,start,end,anchor):
        with self.p.lock:
            e=self.p.engine
            if anchor not in e.nodes or not 0<=start<=end<e.frame_count or end-start>round(e.input['fps']*10):raise ValueError('无效局部范围')
            tid=self.p.result['track_ids'][anchor]
            path=next(t['nodes'] for t in self.p.result['tracks'] if t['id']==tid)
            return self.decorate({'start':start,'end':end,'nodes':[n for n in path if start<=e.nodes[n]['frame']<=end]})

    def constraints_for_path(self, path):
        e = self.p.engine
        active, aliases, _, negatives, groups, _ = e._constraints(self.p.labels)
        c = Components(active, negatives)
        for g in groups:
            for a,b in zip(g,g[1:]): c.union(a,b)
        for a,b in zip(path,path[1:]):
            if e.nodes[a]['frame'] >= e.nodes[b]['frame']: raise Conflict('候选节点时间顺序错误')
            c.union(aliases[a], aliases[b])
        return c

    def candidates(self, task_id, anchor):
        with self.p.lock:
            self.refresh()
            t = self.by_id.get(task_id)
            if not t: raise Conflict('片段已过期')
            e = self.p.engine
            if anchor not in e.nodes or not t['start'] <= e.nodes[anchor]['frame'] <= t['end'] or self.p.result['aliases'][anchor] != anchor:
                raise ValueError('请在当前片段选择实际检测框作为锚点')
            start = e.nodes[anchor]['frame']; end = min(e.frame_count-1, start+round(e.input['fps']*5))
            # Build positive components once, not once per search branch.
            base=self.constraints_for_path([])
            roots={n:base.root(n) for n in base.parent}
            group_frames={r:set(fs) for r,fs in base.frames.items()}
            group_bans={r:{roots[n] for n in banned} for r,banned in base.banned.items()}
            aliases=self.p.result['aliases']
            def path_groups(path):return {roots[aliases[n]] for n in path}
            def compatible(path,b):
                chosen=path_groups(path);r=roots[b]
                if r in chosen:return True
                return all(group_frames[r].isdisjoint(group_frames[s]) and s not in group_bans[r] and r not in group_bans[s] for s in chosen)
            # Time-expanded beam: every frame charges either a transition or one unmatched penalty.
            beam = [(0., (anchor,))]
            for f in range(start+1,end+1):
                expanded = []
                for cost,path in beam:
                    expanded.append((cost+e.config['unmatched_cost'], path))
                    last = path[-1]; gap = f-e.nodes[last]['frame']
                    if gap > e.config['max_gap_frames']: continue
                    choices = [(b,e.score(last,b)) for b in e.by_frame[f] if self.p.result['aliases'][b] == b]
                    choices.sort(key=lambda x:(x[1]['cost'] if x[1]['cost'] is not None else 1e9,x[0]))
                    for b,score in choices[:12]:
                        if score['cost'] is None: continue
                        if compatible(path,b):expanded.append((cost+score['cost'],path+(b,)))
                unique = {}
                for cost,path in expanded:
                    if path not in unique or cost < unique[path]: unique[path]=cost
                ranked = sorted([(cost,path) for path,cost in unique.items()],key=lambda x:(x[0],x[1]))
                # Reserve beam capacity for different immediate continuations.
                beam=[];branch_counts=defaultdict(int)
                for item in ranked:
                    branch=item[1][1] if len(item[1])>1 else None
                    if branch_counts[branch]>=4:continue
                    beam.append(item);branch_counts[branch]+=1
                    if len(beam)==12:break
                if len(beam)<12:
                    kept={path for cost,path in beam}
                    beam.extend(item for item in ranked if item[1] not in kept)
                    beam=beam[:12]
            original_track = next(t for t in self.p.result['tracks'] if anchor in t['nodes'])
            original = [n for n in original_track['nodes'] if start <= e.nodes[n]['frame'] <= end]
            alternatives = []
            beam.sort(key=lambda x:(x[0],x[1]))
            # Show a different early continuation before near-identical late variations.
            firsts=[];rest=[];seen_branches=set()
            for item in beam:
                branch=item[1][1] if len(item[1])>1 else None
                (rest if branch in seen_branches else firsts).append(item);seen_branches.add(branch)
            for cost,path in firsts+rest:
                if len(path)<2 or list(path)==original: continue
                # Positive components may imply extra in-window nodes: include them explicitly in preview.
                members={n for r in path_groups(path) for n in base.members[r]}
                expanded = sorted([n for n in members if start <= e.nodes[n]['frame'] <= end],key=lambda n:e.nodes[n]['frame'])
                if expanded==original or any(x['nodes']==expanded for x in alternatives):continue
                item={'id':digest([self.p.revision,anchor,expanded]),'nodes':expanded,'cost':round(cost,4)}
                alternatives.append(item)
                if len(alternatives)==3:break
            data={'task_id':task_id,'anchor':anchor,'start':start,'end':end,'original':original,'alternatives':alternatives,'revision':self.p.revision}
            for item in alternatives:
                self.previews[item['id']]={'task_id':task_id,'anchor':anchor,'nodes':item['nodes'],'start':start,'end':end,'revision':self.p.revision}
            return {**data,'detections':self.decorate({'start':start,'end':end})['detections']}

    def prepare_review(self, body):
        self.refresh()
        t=self.by_id.get(body.get('task_id'))
        if not t:raise Conflict('该片段已变化，请重新打开；未保存任何判断。')
        if body.get('fingerprint')!=t['fingerprint']:raise Conflict('片段内容已变化，请重新审核。')
        action=body.get('action');path=[];updates={};scope=[t['start'],t['end']]
        if action=='approve':path=t['nodes']
        elif action=='candidate':
            preview=self.previews.get(body.get('candidate_id'))
            if not preview or preview['revision']!=self.p.revision or preview['task_id']!=t['id']:raise Conflict('候选已过期，请重新生成。')
            path=preview['nodes'];scope=[preview['start'],preview['end']]
        elif action=='manual':
            a,b=body.get('a'),body.get('b');value=body.get('value')
            if value not in ('yes','no'):raise ValueError('手动判断必须是同人或非同人')
            if a not in self.p.engine.nodes or b not in self.p.engine.nodes:raise ValueError('未知框')
            fa,fb=[self.p.engine.nodes[n]['frame'] for n in (a,b)]
            if not t['start']<=fa<=t['end'] or not fa<fb<=min(self.p.engine.frame_count-1,fa+round(self.p.engine.input['fps']*5)):raise ValueError('手动框必须位于当前局部范围并按时间先后排列')
            updates[pair(a,b)]=value;path=[a,b];scope=[fa,fb]
        elif action!='defer':raise ValueError('未知审核操作')
        if action in ('approve','candidate'):
            if len(path)<2:raise ValueError('这段只有一个检测框，无法确认身份连接；请选择续接候选或暂跳过。')
            updates={pair(a,b):'yes' for a,b in zip(path,path[1:])}
        for k,value in updates.items():
            old=self.p.labels.get(k)
            if old in ('yes','no') and old!=value:raise Conflict('与已有人工判断冲突，请先在旧框对入口核对并撤销该判断。')
        watched=body.get('watched_seconds',0)
        if not isinstance(watched,(int,float)) or not math.isfinite(watched) or not 0<=watched<=86400:raise ValueError('无效观看时长')
        review={'action':action,'task_id':t['id'],'fingerprint':t['fingerprint'],'nodes':path,'scope':scope,
                'risk_keys':t['risk_keys'],'watched_seconds':watched,'confirmed_pairs':[k for k,v in updates.items() if v=='yes'],
                'resolved_risk_keys':[],'task_snapshot':{k:v for k,v in t.items() if k!='risks'}}
        return updates,review
