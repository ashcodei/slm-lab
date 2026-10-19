"""Independent reconstruction of original human pointer supervision.

This reader deliberately does not import the candidate producer's span or
geometry functions. Publisher speech-assisted timestamps are retained; mouse
samples describe attention rather than object boundaries or OCR truth.
"""
from collections import Counter
import hashlib
import math
import re


POLICY_STOP_WORDS = frozenset(
    'a an the in this image picture we can see there is are was were some and or '
    'of at on to with as it that these those also has have had side left right '
    'top bottom background front'.split())


def original_queries(record):
    """Recompute eligible whole utterances by scanning original trace samples."""
    timed, traces = record['timed_caption'], record['traces']
    if (not isinstance(record.get('caption'), str) or not isinstance(timed, list)
            or not isinstance(traces, list)
            or record['caption'].split() != ' '.join(t['utterance'] for t in timed).split()):
        raise ValueError('Original complete transcript and timing must agree')
    for segment in traces:
        previous = -1.
        for point in segment:
            if any(type(point[k]) not in (int, float) or not math.isfinite(point[k]) for k in ('x', 'y', 't')):
                raise ValueError('Every original sample must be finite')
            if point['t'] < 0 or point['t'] < previous:
                raise ValueError('Original sample times must be ordered')
            previous = point['t']
    previous_start = -1.
    result = []
    for index, timed_row in enumerate(timed):
        start, end, query = (timed_row[k] for k in ('start_time', 'end_time', 'utterance'))
        if (any(type(t) not in (int, float) or not math.isfinite(t) for t in (start, end))
                or not 0 <= start <= end or start < previous_start or not isinstance(query, str)):
            raise ValueError('Original utterance times must be finite and ordered')
        previous_start = start
        words = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", query.casefold())
        if not (.15 <= end-start <= 3 and 1 <= len(words) <= 12
                and any(len(w) >= 3 and w not in POLICY_STOP_WORDS for w in words)):
            continue
        choices = []
        for segment_index, segment in enumerate(traces):
            interval = [(i, p) for i, p in enumerate(segment) if start <= p['t'] <= end]
            inside = [(i, p) for i, p in interval if 0 <= p['x'] < 1 and 0 <= p['y'] < 1]
            if len(inside) < 4 or len(inside) < .8*len(interval):
                continue
            if any(max(p[k] for _, p in inside)-min(p[k] for _, p in inside) > .3 for k in ('x', 'y')):
                continue
            choices.append((segment_index, [i for i, _ in inside]))
        if choices:
            segment_index, positions = sorted(choices, key=lambda item: (-len(item[1]), item[0]))[0]
            result.append(dict(timed_utterance_index=index, query=query, start_time=start, end_time=end,
                source_point_refs=[[segment_index, p] for p in positions], source_point_count=len(positions)))
    return result


def image_layout(dimensions, size=448, grid=14):
    """Return actual padded content bounds and cells overlapping real pixels."""
    if (len(dimensions) != 2 or any(type(n) is not int or n < 1 for n in (*dimensions, size, grid))
            or size % grid):
        raise ValueError('Exact positive image/grid dimensions required')
    width, height = dimensions
    # Preserve Pillow's actual division/multiplication order at half pixels.
    resized = (size, round(height/width*size)) if width >= height else (round(width/height*size), size)
    if min(resized) < 1:
        raise ValueError('Source aspect ratio has no pixels after actual resize')
    left, top = (round((size-n)/2) for n in resized)
    right, bottom = left+resized[0], top+resized[1]
    edge = size//grid
    valid = [max(left, x*edge) < min(right, (x+1)*edge)
             and max(top, y*edge) < min(bottom, (y+1)*edge)
             for y in range(grid) for x in range(grid)]
    return (left, top, *resized), valid


def point_counts(record, query, dimensions, size=448, grid=14):
    (left, top, width, height), valid = image_layout(dimensions, size, grid)
    cells = Counter()
    for segment, index in query['source_point_refs']:
        point = record['traces'][segment][index]
        if not 0 <= point['x'] < 1 or not 0 <= point['y'] < 1:
            raise ValueError('Only original in-frame samples supply targets')
        x = math.floor((left + point['x']*width)/(size/grid))
        y = math.floor((top + point['y']*height)/(size/grid))
        cell = y*grid+x
        if not 0 <= cell < grid*grid or not valid[cell]:
            raise ValueError('Original sample maps outside real image pixels')
        cells[cell] += 1
    if sum(cells.values()) != query['source_point_count']:
        raise ValueError('Every original sample must contribute exactly once')
    return [[cell, count] for cell, count in sorted(cells.items())]


def expected_rows(record, raw, source, offset, photo, tokenizer):
    role = photo['split']
    if role not in ('train', 'validation') or record['image_id'] != photo['image_id']:
        raise ValueError('Original source file role and photo identity required')
    result = []
    reviews = 0
    for query in original_queries(record):
        ids = [tokenizer.token_to_id('<bos>'), *tokenizer.encode(query['query']).ids, tokenizer.token_to_id('<eos>')]
        if len(ids) > 34 or any(s in query['query'] for s in ('<pad>', '<bos>', '<eos>')):
            reviews += 1
            continue
        result.append(dict(id=f"localized-pointer:{role}:{record['image_id']}:{record['annotator_id']}:{query['timed_utterance_index']}",
            split=role, group_id=photo['group_id'], image_id=record['image_id'], annotator_id=record['annotator_id'],
            source_dataset_id=record['dataset_id'], **query, query_input_ids=ids,
            attention_grid_histogram=point_counts(record, query, photo['stored_dimensions']), image_size=448, grid_side=14,
            source_record=dict(source=str(source), offset=offset, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()),
            object_box_or_OCR_label=False, external_model_teacher_used=False, human_preferences=False,
            source_photo_license='CC-BY-2.0', annotation_license='CC-BY-4.0', optimization_forbidden=role != 'train'))
    return result, reviews


def require_row(actual, expected):
    if any(actual.get(k) != value for k, value in expected.items()):
        wrong = [k for k, value in expected.items() if actual.get(k) != value]
        raise ValueError('Candidate differs from original source reconstruction: '+', '.join(wrong))


def dense_target(row, dimensions):
    _, valid = image_layout(dimensions, row['image_size'], row['grid_side'])
    target = [0.]*len(valid)
    previous = -1
    for cell, count in row['attention_grid_histogram']:
        if (type(cell) is not int or type(count) is not int or cell <= previous
                or not 0 <= cell < len(valid) or not valid[cell] or count < 1):
            raise ValueError('Distinct ordered actual-image cells and positive original counts required')
        target[cell] = float(count)
        previous = cell
    if sum(target) != row['source_point_count'] or sum(target) < 4:
        raise ValueError('All original attention sample counts must reconcile')
    return target, valid
