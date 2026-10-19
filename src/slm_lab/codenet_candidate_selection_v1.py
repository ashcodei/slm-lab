"""Select a bounded, deterministic set of distinct authors per problem/language."""
import hashlib


def identity_rank(problem, language, user):
    return hashlib.sha256(('codenet-candidates-v1:' + problem + ':' + language + ':' + user).encode()).hexdigest()


def select(rows, problem, per_language=4, maximum_declared_bytes=32768):
    if per_language < 1 or maximum_declared_bytes < 1:
        raise ValueError('Candidate bounds must be positive')
    buckets = {}
    counts = {'rows': 0, 'accepted': 0, 'accepted_within_declared_byte_bound': 0}
    for row in rows:
        counts['rows'] += 1
        if row['problem_id'] != problem:
            raise ValueError('Original metadata problem identity differs')
        if row['status'] != 'Accepted':
            continue
        counts['accepted'] += 1
        size = int(row['code_size'])
        if not 0 < size <= maximum_declared_bytes:
            continue
        counts['accepted_within_declared_byte_bound'] += 1
        language, user = row['language'], row['user_id']
        if not language or not user:
            raise ValueError('Original language/author identity missing')
        rank = identity_rank(problem, language, user)
        bucket = buckets.setdefault(language, {})
        if user in bucket:
            prior = bucket[user]
            # The submission choice is deterministic, not a claimed efficiency ranking.
            if row['submission_id'] < prior['metadata']['submission_id']:
                bucket[user] = {'author_rank': rank, 'metadata': dict(row)}
        elif len(bucket) < per_language:
            bucket[user] = {'author_rank': rank, 'metadata': dict(row)}
        else:
            worst = max(bucket, key=lambda u: (bucket[u]['author_rank'], u))
            if (rank, user) < (bucket[worst]['author_rank'], worst):
                del bucket[worst]
                bucket[user] = {'author_rank': rank, 'metadata': dict(row)}
    chosen = [value for bucket in buckets.values() for value in bucket.values()]
    chosen.sort(key=lambda r: (r['metadata']['language'], r['author_rank'], r['metadata']['submission_id']))
    return chosen, counts
