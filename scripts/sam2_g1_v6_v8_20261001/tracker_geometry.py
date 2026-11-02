"""Auditable camera-aware motion state, candidate scoring, and identity assignment."""
from dataclasses import dataclass, field
import numpy as np

DEFAULTS = dict(min_iou=.40, min_presence=.35, min_appearance=.52,
                max_mahalanobis2=25., min_match_score=.66,
                memory_score=.76, memory_appearance=.62, memory_mahalanobis2=12.,
                memory_identity_margin=.025, cluster_iou=.65,
                cluster_containment=.88, cluster_area_ratio=.25,
                keyframe_stride=5, max_recent_memories=7, max_extra_references=2)


def project(points, matrix):
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    rays = np.c_[points, np.ones(len(points))] @ matrix.T
    valid = np.isfinite(rays).all(axis=1) & (rays[:, 2] > 1e-7)
    result = np.full((len(points), 2), np.nan)
    result[valid] = rays[valid, :2] / rays[valid, 2:]
    return result, valid


def transform_box(box, matrix):
    x0, y0, x1, y1 = np.asarray(box, dtype=float)
    corners = [[x0,y0], [x1,y0], [x1,y1], [x0,y1]]
    points, valid = project(corners, matrix)
    center, center_ok = project([[(x0+x1)/2, (y0+y1)/2]], matrix)
    if not valid.all() or not center_ok[0]:
        return None
    size = points.max(axis=0)-points.min(axis=0)
    if not np.isfinite(size).all() or (size < .01).any():
        return None
    return np.r_[center[0]-size/2, center[0]+size/2]


def box_to_z(box):
    box = np.asarray(box, dtype=float)
    return np.r_[(box[:2]+box[2:])/2, np.log(np.maximum(box[2:]-box[:2], 2))]


def z_to_box(z):
    size = np.exp(np.clip(np.asarray(z)[2:4], np.log(2), np.log(20000)))
    return np.r_[z[:2]-size/2, z[:2]+size/2]


def projection_jacobian(point, matrix):
    p = np.asarray(point, dtype=float)
    q = matrix @ np.r_[p, 1.]
    if q[2] <= 1e-7:
        return None
    return (matrix[:2,:2]*q[2]-q[:2,None]*matrix[2:3,:2])/(q[2]**2)


def clip_box(box, width, height):
    b = np.asarray(box, dtype=float).copy()
    b[[0,2]] = np.clip(b[[0,2]], 0, width-1)
    b[[1,3]] = np.clip(b[[1,3]], 0, height-1)
    return b


def visible_fraction(box, width, height):
    b = np.asarray(box, dtype=float)
    area = np.maximum(b[2:]-b[:2], 0).prod()
    c = clip_box(b, width, height)
    return float(np.maximum(c[2:]-c[:2], 0).prod()/max(area, 1.))


@dataclass
class Motion:
    x: np.ndarray
    P: np.ndarray
    last_time: float
    last_observation_frame: int
    seed_height: float
    missed: int = 0
    accepted_frames: list = field(default_factory=list)

    @classmethod
    def initial(cls, box, time, frame):
        z = box_to_z(box)
        w,h = np.exp(z[2:])
        std = np.array([max(10,.06*w), max(10,.04*h), .10, .10,
                        max(40,.6*h), max(40,.6*h), .4, .4])
        return cls(np.r_[z,np.zeros(4)], np.diag(std**2), time, frame, h,
                   accepted_frames=[frame])

    def predict(self, time):
        dt = float(time-self.last_time)
        if not 0 < dt <= 1:
            raise ValueError(f'Invalid frame interval: {dt}')
        F = np.eye(8)
        F[:4,4:] = np.eye(4)*dt
        accel = np.array([max(30,.3*self.seed_height)]*2+[.25,.25])
        Q = np.zeros((8,8))
        for i,a in enumerate(accel):
            Q[i,i] = dt**4*a*a/4
            Q[i,i+4] = Q[i+4,i] = dt**3*a*a/2
            Q[i+4,i+4] = dt**2*a*a
        if self.missed >= 5:
            self.x[4:] *= np.exp(-.6*dt)
        self.x = F@self.x
        self.P = F@self.P@F.T+Q
        self.x[2:4] = np.clip(self.x[2:4], np.log(2), np.log(20000))
        self.last_time = time

    def measurement_noise(self, z, uncertain=False):
        w,h = np.exp(np.clip(z[2:], np.log(2), np.log(20000)))
        std = np.array([max(8,.06*w), max(8,.04*h), .12, .12])
        return np.diag(std**2)*(4 if uncertain else 1)

    def innovation(self, box, current_to_anchor):
        anchor_box = transform_box(box, current_to_anchor)
        if anchor_box is None:
            return None
        z = box_to_z(anchor_box)
        error = z-self.x[:4]
        covariance = self.P[:4,:4]+self.measurement_noise(z)
        d2 = float(error[:2]@np.linalg.solve(covariance[:2,:2], error[:2]))
        return dict(z=z, delta=error, mahalanobis2=d2,
                    log_scale_delta=float(np.max(np.abs(error[2:]))))

    def update(self, z, frame, uncertain=False):
        noise = self.measurement_noise(z, uncertain)
        S = self.P[:4,:4]+noise
        gain = np.linalg.solve(S, self.P[:4,:]).T
        self.x += gain@(z-self.x[:4])
        observation = np.zeros((4,8)); observation[:4,:4] = np.eye(4)
        residual = np.eye(8)-gain@observation
        self.P = residual@self.P@residual.T+gain@noise@gain.T
        self.P = (self.P+self.P.T)/2
        self.x[4:6] = np.clip(self.x[4:6], -4*self.seed_height, 4*self.seed_height)
        self.x[6:] = np.clip(self.x[6:], -1.5, 1.5)
        self.last_observation_frame = frame
        self.accepted_frames.append(frame)
        self.missed = 0

    def image_prior(self, anchor_to_current, width, height):
        canonical_box = z_to_box(self.x)
        box = transform_box(canonical_box, anchor_to_current)
        jacobian = projection_jacobian(self.x[:2], anchor_to_current)
        result = dict(anchor_state=self.x.copy(), anchor_covariance=self.P.copy(),
                      last_observation_frame=self.last_observation_frame,
                      missed_frames=self.missed, box=box, search_box=None,
                      sigma_xy=None, visible_fraction=0., projection_valid=box is not None)
        if box is None or jacobian is None:
            return result
        covariance = jacobian@self.P[:2,:2]@jacobian.T
        sigma = np.sqrt(np.maximum(np.diag(covariance),1.))
        size = np.maximum(box[2:]-box[:2], 2)
        padding = np.minimum(np.maximum(2*sigma, size*np.array([.20,.12]))+8,
                             np.array([width*.30,height*.30]))
        search = clip_box(box+np.r_[-padding,padding],width,height)
        fraction = visible_fraction(box,width,height)
        result.update(sigma_xy=sigma, visible_fraction=fraction,
                      search_box=search if (search[2:]-search[:2] > 3).all() and fraction > .02 else None)
        return result


def appearance_similarity(descriptor, bank):
    if not bank:
        return 0.
    values=[]
    for reference in bank:
        feature=float(np.clip(descriptor['feature']@reference['feature'],0,1))
        color=float(np.clip(descriptor['color']@reference['color'],0,1))
        values.append(.65*feature+.35*color)
    return max(values)


def evaluate_candidate(candidate, motion, current_to_anchor, bank, config, width, height):
    reasons=[]
    if candidate['pixels'] < 200 or candidate['box'] is None:
        return dict(valid=False,reasons=['no_usable_mask'],score=0.,appearance=0.,innovation=None)
    innovation=motion.innovation(candidate['box'],current_to_anchor)
    appearance=appearance_similarity(candidate['descriptor'],bank)
    if candidate['iou_score'] < config['min_iou']: reasons.append('low_mask_score')
    if candidate['presence'] < config['min_presence']: reasons.append('low_object_presence')
    if appearance < config['min_appearance']: reasons.append('appearance_mismatch')
    if innovation is None:
        reasons.append('observation_projection_invalid')
        movement=0.
    else:
        movement=float(np.exp(-.125*min(innovation['mahalanobis2'],100)))
        if innovation['mahalanobis2'] > config['max_mahalanobis2']: reasons.append('motion_gate')
        b=candidate['box']
        edge=b[0]<5 or b[1]<5 or b[2]>width-5 or b[3]>height-5
        if innovation['log_scale_delta'] > 1.5 and not edge: reasons.append('scale_gate')
    score=.45*appearance+.25*np.clip(candidate['iou_score'],0,1)+.20*movement+.10*candidate['presence']
    if score < config['min_match_score']: reasons.append('low_match_score')
    return dict(valid=not reasons,reasons=reasons,score=float(score),appearance=appearance,innovation=innovation)


def cluster_candidates(candidates, config):
    parents=list(range(len(candidates)))
    def find(i):
        while parents[i]!=i:
            parents[i]=parents[parents[i]];i=parents[i]
        return i
    links=[]
    for i,a in enumerate(candidates):
        if a['pixels']<200:continue
        for j in range(i):
            b=candidates[j]
            if b['pixels']<200:continue
            # Low-resolution masks are sufficient for duplicate-candidate grouping.
            intersection=int(np.count_nonzero(a['small_mask'] & b['small_mask']))
            aa=int(a['small_mask'].sum());bb=int(b['small_mask'].sum())
            iou=intersection/max(1,aa+bb-intersection)
            containment=intersection/max(1,min(aa,bb))
            ratio=min(aa,bb)/max(1,max(aa,bb))
            if iou>config['cluster_iou'] or (containment>config['cluster_containment'] and ratio>config['cluster_area_ratio']):
                parents[find(i)]=find(j)
                links.append(dict(a=a['candidate_id'],b=b['candidate_id'],iou=iou,containment=containment))
    groups={}
    for i,c in enumerate(candidates):
        if c['pixels']>=200:groups.setdefault(find(i),[]).append(i)
    return list(groups.values()),links


def assign_candidates(identities,candidates,evaluations,groups,config):
    costs=np.full((len(identities),len(groups)+len(identities)),1e6)
    chosen={}
    for row,identity in enumerate(identities):
        costs[row,len(groups)+row]=1-config['min_match_score']
        for column,group in enumerate(groups):
            eligible=[i for i in group if candidates[i]['owner']==identity and evaluations[i]['valid']]
            if not eligible:continue
            best=max(eligible,key=lambda i:evaluations[i]['score'])
            costs[row,column]=1-evaluations[best]['score']
            chosen[(row,column)]=best
    # Five IDs each have at most two owned candidates plus an unmatched choice.
    # Exact enumeration stays tiny and avoids changing the existing SAM2 runtime.
    best_cost=float('inf');best_columns=None
    options=[np.flatnonzero(costs[row]<1e5).tolist() for row in range(len(identities))]
    def search(row,used,total,columns):
        nonlocal best_cost,best_columns
        if total>=best_cost:return
        if row==len(identities):
            best_cost=total;best_columns=columns.copy();return
        for column in sorted(options[row],key=lambda c:costs[row,c]):
            if column not in used:
                search(row+1,used|{column},total+costs[row,column],columns+[column])
    search(0,set(),0.,[])
    rows=range(len(identities));columns=best_columns
    assignments={}
    for row,column in zip(rows,columns):
        index=chosen.get((row,column))
        assignments[identities[row]]=None if index is None else dict(candidate=index,group=int(column))
    real_groups=[v['group'] for v in assignments.values() if v is not None]
    assert len(real_groups)==len(set(real_groups))
    return assignments,costs


def memory_decision(evaluation, identity_margin, config):
    reasons=[]
    if not evaluation['valid']:reasons.append('observation_not_accepted')
    if evaluation['score']<config['memory_score']:reasons.append('match_not_strong_enough')
    if evaluation['appearance']<config['memory_appearance']:reasons.append('appearance_not_strong_enough')
    innovation=evaluation['innovation']
    if innovation is None or innovation['mahalanobis2']>config['memory_mahalanobis2']:reasons.append('motion_not_strong_enough')
    if identity_margin<config['memory_identity_margin']:reasons.append('identity_ambiguous')
    return not reasons,reasons


def clean(value):
    if isinstance(value,np.ndarray):return clean(value.tolist())
    if isinstance(value,np.generic):return clean(value.item())
    if isinstance(value,float) and not np.isfinite(value):return None
    if isinstance(value,dict):return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean(v) for v in value]
    return value
