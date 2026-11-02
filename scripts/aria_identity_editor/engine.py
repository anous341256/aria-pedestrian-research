"""Offline, constrained appearance + Aria-rotation association; not trained."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rotation_match_review'))
from core import warp_box

CONFIG={'max_gap_frames':5,'unmatched_cost':.55,'appearance_weight':.55,
        'geometry_weight':.35,'scale_weight':.10,'invalid_pose_appearance_weight':.75,
        'invalid_pose_geometry_weight':.15,'gap_penalty':.02,'duplicate_iou':.65}


def key(frame,index):return f'f{frame:04d}_d{index:03d}'
def pair(a,b):return '|'.join(sorted((a,b)))
def box_iou(a,b):
    a,b=np.asarray(a),np.asarray(b);inter=np.maximum(np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2]),0).prod()
    return float(inter/max((a[2:]-a[:2]).prod()+(b[2:]-b[:2]).prod()-inter,1e-8))


class Conflict(ValueError):pass


class Components:
    def __init__(self,nodes,negatives=()):
        self.parent={n:n for n in nodes};self.members={n:{n} for n in nodes}
        self.frames={n:{nodes[n]['frame']} for n in nodes};self.banned={n:set() for n in nodes}
        for a,b in negatives:self.banned[a].add(b);self.banned[b].add(a)
    def root(self,x):
        while self.parent[x]!=x:
            self.parent[x]=self.parent[self.parent[x]];x=self.parent[x]
        return x
    def allowed(self,a,b,check_frames=True):
        a,b=self.root(a),self.root(b)
        if a==b:return True
        return ((not check_frames or self.frames[a].isdisjoint(self.frames[b])) and
                self.banned[a].isdisjoint(self.members[b]) and self.banned[b].isdisjoint(self.members[a]))
    def union(self,a,b,check_frames=True):
        a,b=self.root(a),self.root(b)
        if a==b:return a
        if not self.allowed(a,b,check_frames):raise Conflict('身份约束冲突：同一轨迹出现同帧不同对象，或与“不是同一个人”矛盾。')
        if len(self.members[a])<len(self.members[b]):a,b=b,a
        self.parent[b]=a;self.members[a]|=self.members.pop(b);self.frames[a]|=self.frames.pop(b);self.banned[a]|=self.banned.pop(b)
        return a


class Engine:
    def __init__(self,nodes,features,rotations,K,config=None):
        self.nodes={n['key']:n for n in nodes};self.order=[n['key'] for n in nodes]
        self.index={n:i for i,n in enumerate(self.order)};self.config={**CONFIG,**(config or {})}
        f=np.asarray(features,dtype=np.float32);norm=np.linalg.norm(f,axis=1,keepdims=True)
        if len(f)!=len(nodes) or not np.isfinite(f).all() or np.any(norm<1e-8):raise ValueError('Invalid cached appearance features')
        self.features=f/norm;self.R=rotations;self.K=K;self.invK=np.linalg.inv(K)
        self.by_frame=defaultdict(list)
        for n in nodes:self.by_frame[n['frame']].append(n['key'])
        self.frame_count=len(rotations);self.costs={};self._precompute()

    @classmethod
    def load(cls,directory):
        directory=Path(directory);data=json.loads((directory/'input.json').read_text(encoding='utf-8'))
        with np.load(directory/'features.npz',allow_pickle=False) as arrays:
            obj=cls(data['nodes'],arrays['features'],arrays['rotations'],arrays['K'],data['config'])
        obj.input=data;return obj

    def _precompute(self):
        for f,aa in self.by_frame.items():
            R=np.eye(3);valid=True
            for gap in range(1,self.config['max_gap_frames']+1):
                t=f+gap
                if t>=self.frame_count:break
                if np.isfinite(self.R[t]).all():R=self.R[t]@R
                else:valid=False
                bb=self.by_frame[t]
                if not aa or not bb:continue
                sources=np.array([self.nodes[a]['box'] for a in aa]);targets=np.array([self.nodes[b]['box'] for b in bb])
                predicted=[];valid_rows=[]
                for box in sources:
                    p=warp_box(box,self.K@R@self.invK) if valid else None
                    valid_rows.append(p is not None);predicted.append(box if p is None else p)
                predicted=np.array(predicted);valid_rows=np.array(valid_rows)
                diagonal=np.maximum(np.linalg.norm(sources[:,2:]-sources[:,:2],axis=1),1)
                dist=np.linalg.norm(((predicted[:,:2]+predicted[:,2:])/2)[:,None,:]-((targets[:,:2]+targets[:,2:])/2)[None,:,:],axis=2)/diagonal[:,None]
                cosine=np.clip(self.features[[self.index[a] for a in aa]]@self.features[[self.index[b] for b in bb]].T,-1,1)
                app=np.minimum((1-cosine)/.5,1)
                size=np.mean(np.abs(np.log(np.maximum(targets[:,2:]-targets[:,:2],1)[None,:,:]/np.maximum(predicted[:,2:]-predicted[:,:2],1)[:,None,:])),axis=2)
                aw=np.where(valid_rows,self.config['appearance_weight'],self.config['invalid_pose_appearance_weight'])[:,None]
                gw=np.where(valid_rows,self.config['geometry_weight'],self.config['invalid_pose_geometry_weight'])[:,None]
                cost=aw*app+gw*np.minimum(dist/2,1)+self.config['scale_weight']*np.minimum(size/1.4,1)+self.config['gap_penalty']*(gap-1)
                self.costs[f,t]={'a':aa,'b':bb,'cost':cost,'cosine':cosine,'distance':dist,'rotation':valid_rows}

    def score(self,a,b):
        na,nb=self.nodes[a],self.nodes[b]
        entry=self.costs.get((na['frame'],nb['frame']))
        if entry is None:return {'cost':None,'cosine':float(self.features[self.index[a]]@self.features[self.index[b]]),'distance':None,'rotation_valid':False,'gap':nb['frame']-na['frame']}
        i,j=entry['a'].index(a),entry['b'].index(b)
        return {'cost':float(entry['cost'][i,j]),'cosine':float(entry['cosine'][i,j]),'distance':float(entry['distance'][i,j]),'rotation_valid':bool(entry['rotation'][i]),'gap':nb['frame']-na['frame']}

    def _constraints(self,labels):
        yes=[];no=[]
        for token,value in labels.items():
            a,b=token.split('|')
            if a not in self.nodes or b not in self.nodes or a==b:raise ValueError('Unknown detection reference')
            if self.nodes[a]['frame']>=self.nodes[b]['frame']:raise ValueError('Corrections must go forward in time')
            if value=='yes':yes.append((a,b))
            elif value=='no':no.append((a,b))
            elif value!='unsure':raise ValueError('Unknown correction value')
        raw=Components(self.nodes)
        for a,b in yes:raw.union(a,b,check_frames=False)
        for a,b in no:
            if raw.root(a)==raw.root(b):raise Conflict(f'“不是”与已有“同人”链矛盾：{a} / {b}。请撤销或修改冲突判断。')
        aliases={n:n for n in self.nodes};duplicates=[]
        for group in raw.members.values():
            perframe=defaultdict(list)
            for n in group:perframe[self.nodes[n]['frame']].append(n)
            for frame,items in perframe.items():
                if len(items)<2:continue
                if any(box_iou(self.nodes[a]['box'],self.nodes[b]['box'])<self.config['duplicate_iou'] for i,a in enumerate(items) for b in items[i+1:]):
                    raise Conflict(f'同人判断会把第 {frame} 帧多个不重叠的人合成一个 ID：{sorted(items)}。需要先纠正这组判断。')
                canonical=max(items,key=lambda n:(self.nodes[n]['score'],n))
                for n in items:aliases[n]=canonical
                duplicates.append({'frame':frame,'kept':canonical,'suppressed':sorted(n for n in items if n!=canonical),'reason':'human_same_identity_and_overlap_ge_0.65'})
        active={n:v for n,v in self.nodes.items() if aliases[n]==n}
        negatives=[(aliases[a],aliases[b]) for a,b in no]
        comp=Components(active,negatives)
        for a,b in yes:comp.union(aliases[a],aliases[b])
        groups=[sorted(g,key=lambda n:self.nodes[n]['frame']) for g in comp.members.values() if len(g)>1]
        direct={pair(aliases[a],aliases[b]) for a,b in yes}
        return active,aliases,duplicates,negatives,groups,direct

    def solve(self,labels):
        active,aliases,duplicates,negatives,groups,direct=self._constraints(labels)
        comp=Components(active,negatives);incoming={};outgoing={};links=[]
        def connect(a,b,kind):
            if a in outgoing or b in incoming:raise Conflict('Two incoming/outgoing links')
            comp.union(a,b);incoming[b]=a;outgoing[a]=b
            links.append({'a':a,'b':b,'kind':kind,**self.score(a,b)})
        for group in groups:
            for a,b in zip(group,group[1:]):connect(a,b,'human_direct' if pair(a,b) in direct else 'human_transitive')
        # Adjacent-frame assignment first; then reconnect only still-free endpoints over short gaps.
        for gap in range(1,self.config['max_gap_frames']+1):
            for f in range(self.frame_count-gap):
                t=f+gap;entry=self.costs.get((f,t))
                if entry is None:continue
                aa=[a for a in entry['a'] if a in active and a not in outgoing]
                bb=[b for b in entry['b'] if b in active and b not in incoming]
                if not aa or not bb:continue
                row=[entry['a'].index(a) for a in aa];col=[entry['b'].index(b) for b in bb]
                matrix=np.full((len(aa),len(bb)+len(aa)),1e6)
                matrix[:,:len(bb)]=entry['cost'][np.ix_(row,col)]
                for i,a in enumerate(aa):
                    matrix[i,len(bb)+i]=self.config['unmatched_cost']
                    for j,b in enumerate(bb):
                        if not comp.allowed(a,b):matrix[i,j]=1e6
                rr,cc=linear_sum_assignment(matrix)
                for i,j in zip(rr,cc):
                    if j>=len(bb) or matrix[i,j]>=self.config['unmatched_cost']:continue
                    a,b=aa[i],bb[j]
                    if comp.allowed(a,b):connect(a,b,'automatic')
        ids={};tracks=[]
        for members in sorted(comp.members.values(),key=lambda g:min(g)):
            ordered=sorted(members,key=lambda n:self.nodes[n]['frame'])
            anchor=ordered[0];n=self.nodes[anchor];tid=n['frame']*1000+n['index']+1
            for node in ordered:ids[node]=tid
            tracks.append({'id':tid,'anchor':anchor,'nodes':ordered,'start':self.nodes[ordered[0]]['frame'],'end':self.nodes[ordered[-1]]['frame']})
        for n,a in aliases.items():ids[n]=ids[a]
        for token,value in labels.items():
            a,b=token.split('|')
            if value=='yes' and ids[a]!=ids[b]:raise AssertionError('Confirmed identity not enforced')
            if value=='no' and ids[a]==ids[b]:raise AssertionError('Rejected identity reconnected indirectly')
        # Risk prioritization: alternatives close in cost, gaps, invalid pose. Never called confidence.
        for link in links:
            if link['kind']!='automatic':link['risk']=0.;continue
            e=self.costs[self.nodes[link['a']]['frame'],self.nodes[link['b']]['frame']]
            i=e['a'].index(link['a']);j=e['b'].index(link['b']);alternatives=np.delete(e['cost'][i],j)
            margin=float(np.min(alternatives)-link['cost']) if len(alternatives) else 1.
            link['margin']=margin;link['risk']=float(link['cost']+max(0,.15-margin)+.1*(link['gap']-1)+(.15 if not link['rotation_valid'] else 0))
        links.sort(key=lambda l:(self.nodes[l['b']]['frame'],l['a']))
        queue=[{'a':l['a'],'b':l['b'],'reason':'可疑自动连接','risk':l['risk']} for l in sorted(links,key=lambda x:-x['risk']) if l['kind']=='automatic' and pair(l['a'],l['b']) not in labels]
        # Offer broken endpoints too, even when no automatic score passed the cutoff.
        for a in active:
            if a in outgoing:continue
            proposals=self.candidates(a,limit=1,aliases=aliases)
            if proposals and pair(a,proposals[0]['b']) not in labels:
                b=proposals[0]['b']
                queue.append({'a':a,'b':b,'reason':'未连接的轨迹末端','risk':.8})
        queue.sort(key=lambda x:-x['risk'])
        return {'track_ids':ids,'tracks':tracks,'links':links,'aliases':aliases,'duplicates':duplicates,'queue':queue,
                'stats':{'input_boxes':len(self.nodes),'visible_boxes':len(active),'tracks':len(tracks),
                         'automatic_links':sum(l['kind']=='automatic' for l in links),
                         'human_links':sum(l['kind']!='automatic' for l in links),'duplicate_boxes':len(self.nodes)-len(active),
                         'yes':sum(v=='yes' for v in labels.values()),'no':sum(v=='no' for v in labels.values()),'unsure':sum(v=='unsure' for v in labels.values())}}

    def candidates(self,a,limit=12,aliases=None):
        aliases=aliases or {};frame=self.nodes[a]['frame'];choices=[]
        for gap in range(1,self.config['max_gap_frames']+1):
            entry=self.costs.get((frame,frame+gap))
            if entry is None:continue
            for b in entry['b']:
                if aliases.get(b,b)!=b:continue
                choices.append({'a':a,'b':b,**self.score(a,b)})
        # Prefer inspecting immediate next-frame alternatives before gap links.
        choices.sort(key=lambda x:(x['cost'],x['gap'],x['b']))
        return choices[:limit]
