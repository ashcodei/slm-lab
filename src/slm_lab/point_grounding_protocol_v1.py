"""Matched human-attention diagnostics, including priors and image/query controls.

Scores concern the publisher's approximate mouse-attention grid. They do not
measure object-box accuracy, OCR, or broad visual reasoning. Position priors
may be fitted only on original training rows, never validation rows.
"""
from collections import defaultdict
import hashlib
import math


def _key(value):
    return hashlib.sha256(('own-pointer-controls-v1:'+value).encode()).hexdigest()


def fit_position_prior(rows, cells=196, pseudocount=1.):
    if not math.isfinite(pseudocount) or pseudocount <= 0:
        raise ValueError('Positive predeclared prior smoothing required')
    prior = [float(pseudocount)]*cells
    seen = set()
    for row in rows:
        if row['split'] != 'train' or row['optimization_forbidden'] or row['id'] in seen:
            raise ValueError('Only distinct original training queries can fit a position prior')
        seen.add(row['id'])
        total = sum(count for _,count in row['attention_grid_histogram'])
        if total != row['source_point_count'] or total <= 0:
            raise ValueError('Complete original training point counts required')
        for cell,count in row['attention_grid_histogram']:
            if type(cell) is not int or not 0 <= cell < cells or count <= 0:
                raise ValueError('Actual original training cells required')
            prior[cell] += count/total
    if not seen:
        raise ValueError('A source position prior needs actual original training rows')
    return dict(counts=prior, fitted_original_training_queries=len(seen),
        pseudocount=pseudocount, per_query_unit_mass=True, validation_used=False)


def prior_logits(kind, valid, position_prior=None, center_cell=None):
    if not valid or not any(valid):
        raise ValueError('A prior needs actual nonempty image cells')
    if kind == 'uniform':
        counts = [1.]*len(valid)
    elif kind == 'training_position':
        if position_prior is None or len(position_prior['counts']) != len(valid) or position_prior['validation_used']:
            raise ValueError('Only a matching original-training position prior allowed')
        counts = position_prior['counts']
    elif kind == 'center':
        if type(center_cell) is not int or not 0 <= center_cell < len(valid) or not valid[center_cell]:
            raise ValueError('Exact original content center must be a valid image cell')
        # Fixed small smoothing gives this deterministic-location control a
        # finite, explicitly qualified cross-entropy as well as top-1 scores.
        counts = [1e-4]*len(valid); counts[center_cell] = 1.
    else:
        raise ValueError('Unknown predeclared human-attention prior')
    if any(not math.isfinite(c) or c <= 0 for c in counts):
        raise ValueError('Finite positive prior counts required')
    return [math.log(c) for c in counts]


def score(logits, counts, valid, grid=14):
    if (len(logits) != grid*grid or len(counts) != len(logits) or len(valid) != len(logits)
            or not any(valid) or any(not math.isfinite(v) for v in logits)
            or any(not math.isfinite(v) or v < 0 for v in counts)
            or any(c and not v for c,v in zip(counts,valid)) or sum(counts) <= 0):
        raise ValueError('Exact finite predictions and complete original image-grid targets required')
    allowed = [i for i,flag in enumerate(valid) if flag]
    ordered = sorted(allowed, key=lambda i: (-logits[i],i))
    predicted = ordered[0]
    maximum = max(logits[i] for i in allowed)
    log_normalizer = maximum+math.log(sum(math.exp(logits[i]-maximum) for i in allowed))
    total = sum(counts)
    nll = sum(c/total*(log_normalizer-logits[i]) for i,c in enumerate(counts) if c)
    distance = sum(c/total*math.hypot((predicted%grid-i%grid)/grid,
        (predicted//grid-i//grid)/grid) for i,c in enumerate(counts) if c)
    return dict(predicted_cell=predicted, original_sample_count=total,
        soft_target_cross_entropy=nll, top1_observed_attention_cell=counts[predicted] > 0,
        original_sample_mass_at_top1=counts[predicted]/total,
        original_sample_mass_in_top5=sum(counts[i] for i in ordered[:5])/total,
        expected_normalized_grid_center_distance=distance)


def control_pairs(rows, photos):
    """Preseal deterministic wrong photos and disjoint same-photo query swaps."""
    by_photo = defaultdict(list)
    seen = set()
    for row in rows:
        if row['split'] != 'validation' or not row['optimization_forbidden'] or row['id'] in seen:
            raise ValueError('Distinct original validation queries required for diagnostics')
        seen.add(row['id']); by_photo[row['image_id']].append(row)
    keys = sorted(by_photo,key=_key)
    if len(keys) < 2 or any(key not in photos for key in keys):
        raise ValueError('At least two actual validation photo identities required')
    result = []
    for index,key in enumerate(keys):
        aliases = set(photos[key]['source_photo_aliases'])
        other = next((candidate for candidate in keys[index+1:]+keys[:index]
            if not aliases.intersection(photos[candidate]['source_photo_aliases'])),None)
        if other is None:
            raise ValueError('A wrong-image control needs a distinct original scene identity')
        for row in sorted(by_photo[key],key=lambda r:_key(r['id'])):
            cells = {cell for cell,_ in row['attention_grid_histogram']}
            alternatives = [r for r in by_photo[key] if r['id'] != row['id']
                and r['query'].casefold().split() != row['query'].casefold().split()
                and cells.isdisjoint(cell for cell,_ in r['attention_grid_histogram'])]
            swap = min(alternatives,key=lambda r:_key(r['id']))['id'] if alternatives else None
            result.append(dict(id=row['id'], image_id=key, wrong_image_id=other, wrong_query_id=swap))
    return result


METRICS = ('soft_target_cross_entropy','top1_observed_attention_cell','original_sample_mass_at_top1',
    'original_sample_mass_in_top5','expected_normalized_grid_center_distance')


def summarize(observations):
    """Recount per-query and per-photo means on exactly matched control rows."""
    observations = list(observations)
    by_condition = defaultdict(dict)
    for row in observations:
        condition = row['condition']
        if row['id'] in by_condition[condition]:
            raise ValueError('Each actual query/control prediction must be unique')
        if any(not math.isfinite(float(row['scores'][m])) for m in METRICS):
            raise ValueError('Finite actual retained diagnostic scores required')
        by_condition[condition][row['id']] = row
    required = ('proper','blank_image','wrong_image','uniform','center','training_position')
    if any(c not in by_condition for c in required):
        raise ValueError('All predeclared model/image/position controls required')
    proper = by_condition['proper']
    if not proper or any(by_condition[c].keys() != proper.keys() for c in required):
        raise ValueError('Every primary control must use exactly the same original queries')
    if any(row['image_id'] != proper[k]['image_id'] for c in by_condition for k,row in by_condition[c].items() if k in proper):
        raise ValueError('Control scores must retain their original target photo identity')
    swaps = by_condition.get('wrong_query',{})
    if swaps.keys()-proper.keys():
        raise ValueError('Query swaps must be a declared subset of the original queries')
    result = {}
    for condition,rows in by_condition.items():
        images = defaultdict(list)
        for row in rows.values():
            images[row['image_id']].append(row['scores'])
        if not rows:
            continue
        result[condition] = dict(queries=len(rows), original_photos=len(images),
            query_mean={m:sum(r['scores'][m] for r in rows.values())/len(rows) for m in METRICS},
            photo_mean={m:sum(sum(r[m] for r in group)/len(group) for group in images.values())/len(images) for m in METRICS},
            matched_proper_query_mean={m:sum(proper[k]['scores'][m] for k in rows)/len(rows) for m in METRICS})
    return dict(conditions=result, all_primary_controls_exact_same_query_ids=True,
        wrong_query_scope='Only different original phrases with disjoint original attention cells on the same photo.',
        distance_scope='Euclidean distance between 14x14 grid centers; original pointer positions are quantized.',
        object_box_OCR_or_general_visual_reasoning_accuracy=False)
