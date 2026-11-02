"""Geometry-only association audit. Human pair decisions are never model labels."""
import math
import numpy as np

METHODS = ('none', 'rgb', 'aria')
NAMES = {'none': '不补偿', 'rgb': '原 RGB 补偿', 'aria': 'Aria 旋转补偿'}
VALUES = ('yes', 'no', 'unsure')


def warp_box(box, H):
    x1, y1, x2, y2 = box
    corners = np.array([[x1,y1,1], [x2,y1,1], [x2,y2,1], [x1,y2,1]], float)
    q = corners @ np.asarray(H).T
    if not np.isfinite(q).all() or np.any(q[:,2] <= 1e-8):
        return None
    q = q[:,:2] / q[:,2:3]
    return np.r_[q.min(axis=0), q.max(axis=0)]


def box_scores(source, predicted, candidates):
    source, predicted, candidates = map(np.asarray, (source, predicted, candidates))
    diagonal = max(float(np.linalg.norm(source[2:]-source[:2])), 1e-8)
    centers = (candidates[:,:2]+candidates[:,2:])/2
    errors = np.linalg.norm(centers-(predicted[:2]+predicted[2:])/2,axis=1)
    lt, rb = np.maximum(predicted[:2], candidates[:,:2]), np.minimum(predicted[2:], candidates[:,2:])
    intersection = np.maximum(rb-lt,0).prod(axis=1)
    area = (predicted[2:]-predicted[:2]).prod()
    union = area + (candidates[:,2:]-candidates[:,:2]).prod(axis=1)-intersection
    iou = intersection / np.maximum(union,1e-8)
    scores = errors/diagonal
    order = np.argsort(scores,kind='stable').tolist()
    iou_order = np.argsort(-iou,kind='stable').tolist()
    return dict(predicted_box=predicted.tolist(), distances_px=errors.tolist(),
                normalized_distances=scores.tolist(), iou=iou.tolist(), order=order,
                iou_order=iou_order, top1=order[0], tie_top1=bool(len(order)>1 and abs(scores[order[0]]-scores[order[1]])<1e-10))


def pair_key(query_id, candidate):
    return f'{query_id}:{int(candidate)}'


def answers_for(query, labels):
    return {i:labels.get(pair_key(query['id'],i)) for i in range(len(query['candidates']))}


def question_state(query, labels):
    answers = answers_for(query,labels)
    primary = query['review_order']
    for i in primary:
        if answers[i] is None:
            return i, 'primary'
    if any(v == 'yes' for v in answers.values()):
        return None, 'reviewed'
    if any(answers[i] == 'unsure' for i in primary):
        return None, 'uncertain'
    # Corrections: at most three additional candidates by best rank across methods.
    rest = [i for i in range(len(answers)) if i not in primary]
    rest.sort(key=lambda i:(min(query['methods'][m]['order'].index(i) for m in METHODS),i))
    for i in rest[:3]:
        if answers[i] is None:
            return i, 'correction'
        if answers[i] == 'unsure':
            return None, 'uncertain'
    return None, 'no_positive_yet' if len(rest)>3 else 'all_rejected'


def _median(values):
    return float(np.median(values)) if values else None


def summarize(queries, labels):
    """Explicit denominators; no identity inferred from one accepted candidate."""
    groups = {}
    for cohort in ('random','disagreement'):
        selected = [q for q in queries if q['cohort']==cohort]
        n = len(selected)
        result = {'queries':n,'reviewed_pairs':0,'confirmed_positive_queries':0,
                  'multiple_positive_queries':0,'states':{},'methods':{},'paired':{}}
        positives = {}
        for q in selected:
            a = answers_for(q,labels)
            result['reviewed_pairs'] += sum(v is not None for v in a.values())
            yes = [i for i,v in a.items() if v=='yes']
            positives[q['id']] = yes
            result['confirmed_positive_queries'] += bool(yes)
            result['multiple_positive_queries'] += len(yes)>1
            _,state = question_state(q,labels)
            result['states'][state] = result['states'].get(state,0)+1
        for m in METHODS:
            vals = [labels.get(pair_key(q['id'],q['methods'][m]['top1'])) for q in selected]
            yes,no,unsure = (vals.count(v) for v in VALUES)
            ranks, errors, margins, iou_ranks = [],[],[],[]
            known_present_vals = []
            for q,v in zip(selected,vals):
                ps = positives[q['id']]
                if not ps:
                    continue
                method=q['methods'][m]
                known_present_vals.append(v)
                # Multiple accepted boxes: best accepted rank, explicitly not unique GT.
                ranks.append(min(method['order'].index(i)+1 for i in ps))
                iou_ranks.append(min(method['iou_order'].index(i)+1 for i in ps))
                errors.append(min(method['distances_px'][i] for i in ps))
                negatives=[i for i,a in answers_for(q,labels).items() if a=='no']
                if negatives:
                    margins.append(min(method['normalized_distances'][i] for i in negatives)-min(method['normalized_distances'][i] for i in ps))
            kn_yes=known_present_vals.count('yes'); kn_no=known_present_vals.count('no')
            result['methods'][m]={'yes':yes,'no':no,'unsure':unsure,'pending':vals.count(None),
                'judged_accuracy':yes/(yes+no) if yes+no else None,
                'all_sample_lower':yes/n if n else None,'all_sample_upper':(n-no)/n if n else None,
                'confirmed_present_judged_n':kn_yes+kn_no,
                'confirmed_present_accuracy':kn_yes/(kn_yes+kn_no) if kn_yes+kn_no else None,
                'positive_rank_median':_median(ranks),'positive_iou_rank_median':_median(iou_ranks),
                'positive_center_error_px_median':_median(errors),
                'reviewed_negative_margin_median':_median(margins),'reviewed_negative_margin_n':len(margins)}
        for baseline in ('none','rgb'):
            rows=[]
            for q in selected:
                a=labels.get(pair_key(q['id'],q['methods']['aria']['top1']))
                b=labels.get(pair_key(q['id'],q['methods'][baseline]['top1']))
                if a in ('yes','no') and b in ('yes','no'):
                    rows.append((q,a,b))
            gains=sum(a=='yes' and b=='no' for q,a,b in rows)
            losses=sum(a=='no' and b=='yes' for q,a,b in rows)
            delta=(gains-losses)/len(rows) if rows else None
            # Resample five-second blocks, not individual detections. Descriptive on reviewed subset.
            blocks={}
            for q,a,b in rows:
                blocks.setdefault(q['time_block'],[]).append(int(a=='yes')-int(b=='yes'))
            ci=None
            if len(blocks)>=4:
                arrays=list(blocks.values()); rng=np.random.default_rng(20260927)
                samples=[]
                for _ in range(1000):
                    chosen=rng.integers(0,len(arrays),len(arrays))
                    samples.append(np.mean([x for i in chosen for x in arrays[i]]))
                ci=np.quantile(samples,[.025,.975]).tolist()
            result['paired'][baseline]={'n':len(rows),'gains':gains,'losses':losses,
                'accuracy_delta':delta,'time_blocks':len(blocks),'block_bootstrap_95':ci,
                'note':'仅已明确判断的共同子集；部分审核有选择偏差，非全样本结论'}
        groups[cohort]=result
    return groups
