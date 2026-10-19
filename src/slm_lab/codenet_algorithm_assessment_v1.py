"""Authored held-task correctness/stress cases, never training demonstrations.

Contracts are the retained original CodeNet descriptions. These local cases
extend published examples; they are not the original judges' hidden tests or
a proof of universal correctness, complexity, or pretraining independence.
"""
from bisect import bisect_left, bisect_right
from collections import Counter, deque
from fractions import Fraction
from functools import lru_cache
import heapq
import random
import re

from .codenet_sample_oracles_v1 import check_sample, stagecoach_times, reachable_masses

PROBLEMS = ('p00097', 'p00439', 'p00719', 'p00836', 'p02787', 'p02889', 'p02917',
            'p03087', 'p03095', 'p03409', 'p03476', 'p03557', 'p03601', 'p03746')
INTEGER = re.compile(r'[+-]?[0-9]{1,30}')


@lru_cache(maxsize=2)
def primes(limit):
    result = bytearray(b'\1') * (limit + 1)
    result[:2] = b'\0\0'
    for p in range(2, int(limit**.5) + 1):
        if result[p]:
            result[p*p::p] = b'\0' * ((limit - p*p) // p + 1)
    return result


@lru_cache(maxsize=1)
def subset_counts():
    table = [[0] * 1001 for _ in range(10)]
    table[0][0] = 1
    for value in range(101):
        for count in range(9, 0, -1):
            for total in range(1000, value - 1, -1):
                table[count][total] += table[count - 1][total - value]
    return table


@lru_cache(maxsize=1)
def consecutive_prime_counts():
    ps = [i for i, yes in enumerate(primes(10000)) if yes]
    counts = [0] * 10001
    for first in range(len(ps)):
        total = 0
        for value in ps[first:]:
            total += value
            if total > 10000:
                break
            counts[total] += 1
    return counts


def integers(text):
    values = text.split()
    if not all(INTEGER.fullmatch(value) for value in values):
        raise ValueError('Only bounded integer input tokens required')
    return list(map(int, values))


def lines(values):
    return '\n'.join(map(str, values)) + '\n'


def fuel_answers(n, edges, capacity, queries):
    graph = [[] for _ in range(n)]
    for a, b, cost in edges:
        graph[a - 1].append((b - 1, cost))
        graph[b - 1].append((a - 1, cost))
    reachable = []
    for start in range(n):
        distance = [10**30] * n
        distance[start] = 0
        queue = [(0, start)]
        while queue:
            cost, vertex = heapq.heappop(queue)
            if cost != distance[vertex]:
                continue
            for other, weight in graph[vertex]:
                candidate = cost + weight
                if candidate < distance[other] and candidate <= capacity:
                    distance[other] = candidate
                    heapq.heappush(queue, (candidate, other))
        reachable.append([i for i, cost in enumerate(distance) if i != start and cost <= capacity])
    answers = []
    for start in range(n):
        distance = [-1] * n
        distance[start] = 0
        queue = deque([start])
        while queue:
            vertex = queue.popleft()
            for other in reachable[vertex]:
                if distance[other] == -1:
                    distance[other] = distance[vertex] + 1
                    queue.append(other)
        answers.append([max(-1, cost - 1) for cost in distance])
    return [answers[a - 1][b - 1] for a, b in queries]


def matching(red, blue):
    assigned = [-1] * len(blue)

    def augment(i, seen):
        for j, (x, y) in enumerate(blue):
            if j in seen or not (red[i][0] < x and red[i][1] < y):
                continue
            seen.add(j)
            if assigned[j] == -1 or augment(assigned[j], seen):
                assigned[j] = i
                return True
        return False

    return sum(augment(i, set()) for i in range(len(red)))


def expected_output(problem, text):
    """Construct one valid output; property tasks use validators, not this witness."""
    if problem == 'p00719':
        return lines('Impossible' if value is None else format(float(value), '.10f')
                     for value in stagecoach_times(text))
    v = [] if problem in {'p03087', 'p03095'} else integers(text)
    if problem == 'p00097':
        assert len(v) % 2 == 0 and v[-2:] == [0, 0] and len(v) <= 102
        pairs = list(zip(v[:-2:2], v[1:-2:2]))
        assert all(1 <= n <= 9 and 0 <= s <= 1000 for n, s in pairs)
        answers = [subset_counts()[n][s] for n, s in pairs]
        assert all(value <= 10**10 for value in answers)
        return lines(answers)
    if problem == 'p00439':
        out, position = [], 0
        while v[position:position + 2] != [0, 0]:
            n, k = v[position:position + 2]
            position += 2
            a = v[position:position + n]
            position += n
            assert 1 <= k <= n <= 100000 and len(a) == n and all(-10000 <= x <= 10000 for x in a)
            total = best = sum(a[:k])
            for i in range(k, n):
                total += a[i] - a[i - k]
                best = max(best, total)
            out.append(best)
        assert position + 2 == len(v) and len(out) <= 5
        return lines(out)
    if problem == 'p00836':
        assert v[-1] == 0 and all(2 <= x <= 10000 for x in v[:-1])
        return lines(consecutive_prime_counts()[x] for x in v[:-1])
    if problem == 'p02787':
        h, n = v[:2]
        assert 1 <= h <= 10000 and 1 <= n <= 1000 and len(v) == 2 + 2*n
        spells = list(zip(v[2::2], v[3::2]))
        assert all(1 <= a <= 10000 and 1 <= b <= 10000 for a, b in spells)
        cost = [0] + [10**30] * h
        for health in range(1, h + 1):
            cost[health] = min(cost[max(0, health - a)] + b for a, b in spells)
        return lines([cost[h]])
    if problem == 'p02889':
        n, m, capacity = v[:3]
        assert 2 <= n <= 300 and 0 <= m <= n*(n-1)//2 and 1 <= capacity <= 10**9
        edges = [tuple(v[i:i+3]) for i in range(3, 3 + 3*m, 3)]
        assert len({tuple(sorted((a, b))) for a, b, _ in edges}) == m
        assert all(1 <= a <= n and 1 <= b <= n and a != b and 1 <= c <= 10**9 for a, b, c in edges)
        position = 3 + 3*m
        q = v[position]
        queries = list(zip(v[position+1::2], v[position+2::2]))
        assert 1 <= q <= n*(n-1) and len(v) == position + 1 + 2*q and len(set(queries)) == q
        assert all(1 <= a <= n and 1 <= b <= n and a != b for a, b in queries)
        return lines(fuel_answers(n, edges, capacity, queries))
    if problem == 'p02917':
        n, b = v[0], v[1:]
        assert 2 <= n <= 100 and len(b) == n - 1 and all(0 <= x <= 100000 for x in b)
        return lines([b[0] + b[-1] + sum(min(a, z) for a, z in zip(b, b[1:]))])
    if problem == 'p03087':
        parts = text.split()
        n, q, s = int(parts[0]), int(parts[1]), parts[2]
        assert 2 <= n <= 100000 and 1 <= q <= 100000 and len(s) == n and set(s) <= set('ACGT')
        assert len(parts) == 3 + 2*q
        queries = list(zip(map(int, parts[3::2]), map(int, parts[4::2])))
        assert all(1 <= l < r <= n for l, r in queries)
        prefix = [0] * n
        for i in range(1, n):
            prefix[i] = prefix[i-1] + (s[i-1:i+1] == 'AC')
        return lines(prefix[r-1] - prefix[l-1] for l, r in queries)
    if problem == 'p03095':
        size, s = text.split()
        assert 1 <= int(size) <= 100000 and len(s) == int(size) and all('a' <= c <= 'z' for c in s)
        answer = 1
        for count in Counter(s).values():
            answer = answer * (count + 1) % 1000000007
        return lines([(answer - 1) % 1000000007])
    if problem == 'p03409':
        n = v[0]
        points = list(zip(v[1::2], v[2::2]))
        assert 1 <= n <= 100 and len(v) == 1 + 4*n
        assert len({x for x, _ in points}) == len({y for _, y in points}) == 2*n
        assert all(0 <= x < 2*n and 0 <= y < 2*n for x, y in points)
        return lines([matching(points[:n], points[n:])])
    if problem == 'p03476':
        q = v[0]
        queries = list(zip(v[1::2], v[2::2]))
        assert 1 <= q <= 100000 and len(v) == 1 + 2*q
        assert all(1 <= l <= r <= 100000 and l % 2 == r % 2 == 1 for l, r in queries)
        sieve = primes(100000)
        prefix = [0] * 100001
        for x in range(1, 100001):
            prefix[x] = prefix[x - 1] + bool(x % 2 and sieve[x] and sieve[(x+1)//2])
        return lines(prefix[r] - prefix[l-1] for l, r in queries)
    if problem == 'p03557':
        n = v[0]
        assert 1 <= n <= 100000 and len(v) == 1 + 3*n and all(1 <= x <= 10**9 for x in v[1:])
        a, b, c = sorted(v[1:1+n]), v[1+n:1+2*n], sorted(v[1+2*n:])
        return lines([sum(bisect_left(a, value) * (n - bisect_right(c, value)) for value in b)])
    if problem == 'p03601':
        a, b, c, d, e, limit = v
        assert 1 <= a < b <= 30 and 1 <= c < d <= 30 and 1 <= e <= 100 and 100*a <= limit <= 3000
        water, sugar = reachable_masses(100*a, 100*b, limit), reachable_masses(c, d, limit)
        masses = [i for i, yes in enumerate(sugar) if yes]
        best, witness = Fraction(-1), None
        for w in range(1, limit + 1):
            if water[w]:
                s = masses[bisect_right(masses, min(limit - w, e*w//100)) - 1]
                density = Fraction(s, w + s)
                if density > best:
                    best, witness = density, (w+s, s)
        return ' '.join(map(str, witness)) + '\n'
    if problem == 'p03746':
        n, m = v[:2]
        edges = list(zip(v[2::2], v[3::2]))
        assert 2 <= n <= 100000 and 1 <= m <= 100000 and len(v) == 2 + 2*m
        assert len(set(edges)) == m and all(1 <= a < b <= n for a, b in edges)
        graph = [set() for _ in range(n+1)]
        for a, b in edges:
            graph[a].add(b)
            graph[b].add(a)
        reached, stack = {1}, [1]
        while stack:
            for other in graph[stack.pop()]:
                if other not in reached:
                    reached.add(other)
                    stack.append(other)
        assert len(reached) == n
        path, used = deque([1]), {1}
        for side in [False, True]:
            while True:
                end = path[0] if side else path[-1]
                other = next((i for i in sorted(graph[end]) if i not in used), None)
                if other is None:
                    break
                (path.appendleft if side else path.append)(other)
                used.add(other)
        return str(len(path)) + '\n' + ' '.join(map(str, path)) + '\n'
    raise ValueError('Unknown source-qualified algorithm task')


def check_output(problem, text, expected, actual):
    if problem in {'p00719', 'p03601'}:
        return check_sample(problem, text, expected, actual)['passed']
    if problem == 'p03746':
        try:
            values = integers(actual)
            k, path = values[0], values[1:]
            n, m, *edges = integers(text)
            if not 2 <= k <= n or k != len(path) or len(set(path)) != k or any(not 1 <= x <= n for x in path):
                return False
            pairs = {tuple(edges[i:i+2]) for i in range(0, len(edges), 2)}
            if any(tuple(sorted(pair)) not in pairs for pair in zip(path, path[1:])):
                return False
            used, ends = set(path), {path[0], path[-1]}
            return all((a not in ends or b in used) and (b not in ends or a in used) for a, b in pairs)
        except (ValueError, IndexError):
            return False
    try:
        return integers(actual) == integers(expected)
    except ValueError:
        return False


def authored_inputs(problem, seed=731906):
    """Fixed constraints-valid edge/random cases plus separately labelled stress."""
    rng = random.Random(seed + int(problem[1:]))
    out = []

    def add(text, stress=False):
        out.append({'kind': 'stress' if stress else 'functional', 'input': text})

    def pairs_text(pairs):
        return ''.join(' '.join(map(str, row)) + '\n' for row in pairs)

    if problem == 'p00097':
        add(pairs_text([(1, 0), (1, 100), (2, 1), (9, 35), (9, 36), (9, 450), (9, 1000)]) + '0 0\n')
        add(pairs_text([(rng.randint(1, 9), rng.randint(0, 1000)) for _ in range(50)]) + '0 0\n', True)
    elif problem == 'p00439':
        for a, k in [([-8], 1), ([-9, -2, -8], 2), ([10000, -10000, 10000], 3), ([5, 0, -7, 2], 1)]:
            add(f'{len(a)} {k}\n' + lines(a) + '0 0\n')
        add('100000 50000\n' + lines(rng.randint(-10000, 10000) for _ in range(100000)) + '0 0\n', True)
    elif problem == 'p00719':
        add('1 2 0 1 2\n10\n0 0 0 0 0\n')
        add('1 3 2 1 3\n2\n1 2 1\n2 3 1\n0 0 0 0 0\n')
        add('2 3 3 1 3\n1 10\n1 2 2\n2 3 20\n1 3 30\n0 0 0 0 0\n')
        for n, m in [(3, 6), (4, 7), (8, 30)]:
            edges = [(i, i+1, rng.randint(1, 100)) for i in range(1, m)]
            for i in range(1, m):
                for j in range(i+2, m+1):
                    if rng.random() < .15:
                        edges.append((i, j, rng.randint(1, 100)))
            add(f'{n} {m} {len(edges)} 1 {m}\n' + ' '.join(str(rng.randint(1, 10)) for _ in range(n)) + '\n'
                + pairs_text(edges) + '0 0 0 0 0\n', n == 8)
    elif problem == 'p00836':
        add(lines([2, 3, 4, 5, 20, 41, 53, 9973, 10000, 0]))
        add(lines(list(range(2, 10001)) + [0]), True)
    elif problem == 'p02787':
        for h, spells in [(1, [(10000, 10000)]), (17, [(5, 4), (9, 7)]), (10, [(3, 1)]), (12, [(6, 7), (4, 5), (1, 2)])]:
            add(f'{h} {len(spells)}\n' + pairs_text(spells))
        for _ in range(5):
            spells = [(rng.randint(1, 30), rng.randint(1, 40)) for _ in range(6)]
            add(f'{rng.randint(1, 60)} 6\n' + pairs_text(spells))
        add('10000 1000\n' + pairs_text([(i, rng.randint(1, 10000)) for i in range(1, 1001)]), True)
    elif problem == 'p02889':
        for n, edges, cap in [(2, [], 1), (3, [(1,2,5),(2,3,5)], 5), (4, [(1,2,2),(2,3,2),(3,4,2)], 4),
                               (3, [(1,2,6),(2,3,2),(1,3,10)], 5)]:
            qs = [(a,b) for a in range(1,n+1) for b in range(1,n+1) if a != b]
            add(f'{n} {len(edges)} {cap}\n' + pairs_text(edges) + f'{len(qs)}\n' + pairs_text(qs))
        n, cap = 300, 10**9
        edges = [(i, i+1, 400000000) for i in range(1,n)]
        qs = [(a,b) for a in range(1,n+1) for b in range(1,n+1) if a != b]
        add(f'{n} {len(edges)} {cap}\n' + pairs_text(edges) + f'{len(qs)}\n' + pairs_text(qs), True)
    elif problem == 'p02917':
        for b in [[0], [100000], [1,100000,1], [0,0,0], [0,100000,0,100000], [rng.randint(0,100000) for _ in range(99)]]:
            add(f'{len(b)+1}\n' + ' '.join(map(str,b)) + '\n')
    elif problem == 'p03087':
        for s in ['AC', 'CA', 'AAAA', 'CACAC', 'ACGTAC']:
            qs = [(l,r) for l in range(1,len(s)) for r in range(l+1,len(s)+1)]
            add(f'{len(s)} {len(qs)}\n{s}\n' + pairs_text(qs))
        qs = [(1,100000)] * 50000
        for _ in range(50000):
            l = rng.randint(1,99999)
            qs.append((l,rng.randint(l+1,100000)))
        add('100000 100000\n' + 'AC'*50000 + '\n' + pairs_text(qs), True)
    elif problem == 'p03095':
        for s in ['a','aa','abab','abcdefghijklmnopqrstuvwxyz','zzzzzzzzzz', 'abacabadabacaba']:
            add(f'{len(s)}\n{s}\n')
        for s in ['a'*100000, ('abcdefghijklmnopqrstuvwxyz'*3847)[:100000]]:
            add(f'{len(s)}\n{s}\n', True)
    elif problem == 'p03409':
        for n in [1,2,4,5,100]:
            xs,ys = list(range(2*n)),list(range(2*n))
            rng.shuffle(xs);rng.shuffle(ys)
            add(f'{n}\n' + pairs_text(list(zip(xs,ys))), n==100)
        add('3\n0 0\n1 1\n2 2\n3 3\n4 4\n5 5\n')
        add('3\n3 3\n4 4\n5 5\n0 0\n1 1\n2 2\n')
    elif problem == 'p03476':
        qs = [(1,1),(3,3),(5,5),(7,7),(1,99999),(99999,99999),(2017,2017)]
        add(f'{len(qs)}\n' + pairs_text(qs))
        qs = [(1,99999)] * 50000
        for _ in range(50000):
            a,b=sorted([rng.randrange(50000),rng.randrange(50000)])
            qs.append((2*a+1,2*b+1))
        add('100000\n' + pairs_text(qs), True)
    elif problem == 'p03557':
        for n, a, b, c in [(1,[1],[1],[1]), (1,[1],[2],[3]), (3,[1]*3,[2]*3,[2]*3),
                             (3,[1]*3,[2]*3,[3]*3), (5,[5,4,3,2,1],[2,3,4,5,6],[3,4,5,6,7])]:
            add(f'{n}\n' + pairs_text([a,b,c]))
        add('100000\n' + pairs_text([[1]*100000,[2]*100000,[3]*100000]), True)
        add('100000\n' + pairs_text([[rng.randint(1,10**9) for _ in range(100000)] for _ in range(3)]), True)
    elif problem == 'p03601':
        for v in [(1,2,1,2,1,100),(1,2,29,30,1,3000),(1,2,1,2,100,3000),
                  (29,30,1,2,100,3000),(1,2,7,11,17,321)]:
            add(' '.join(map(str,v))+'\n',v[2]==1 and v[4]==100)
        for _ in range(5):
            a,b=sorted(rng.sample(range(1,8),2));c,d=sorted(rng.sample(range(1,31),2))
            v=(a,b,c,d,rng.randint(1,100),rng.randint(100*a,3000))
            add(' '.join(map(str,v))+'\n')
    elif problem == 'p03746':
        for n, edges in [(2,[(1,2)]), (5,[(i,i+1) for i in range(1,5)]+[(1,5)]),
                         (8,[(1,i) for i in range(2,9)]), (7,[(i,j) for i in range(1,7) for j in range(i+1,8)])]:
            add(f'{n} {len(edges)}\n'+pairs_text(edges))
        n=100000
        for edges in [[(i,i+1) for i in range(1,n)],[(1,i) for i in range(2,n+1)]]:
            add(f'{n} {len(edges)}\n'+pairs_text(edges), True)
    else:
        raise ValueError('Unknown source-qualified algorithm task')
    return out
