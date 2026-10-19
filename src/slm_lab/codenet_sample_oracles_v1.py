"""Source-grounded sample qualifications; no generic literal reward oracle."""
from decimal import Decimal, localcontext
from fractions import Fraction
import re


SUPPORTED = {'p00719', 'p01411', 'p01853', 'p03601'}
REAL = re.compile(r'[+-]?(?:\d{1,32}(?:\.\d{0,32})?|\.\d{1,32})(?:[eE][+-]?\d{1,3})?')


def mean_median_gap(values):
    ordered = sorted(values)
    n = len(values)
    if not n:
        raise ValueError('Nonempty score distribution required')
    median = Fraction(ordered[n // 2]) if n % 2 else Fraction(ordered[n // 2 - 1] + ordered[n // 2], 2)
    return abs(Fraction(sum(values), n) - median)


def maximum_mean_median_gap(n, maximum):
    """Exact optimum for the original bounded integer-score problem.

    For odd n=2k+1, at most k scores on either side of the median contribute
    positively to the gap, giving k*m/n. For even n=2k, bounding the two
    central scores gives (k-1)*m/n. Both are achieved by k+1 zero scores
    followed by the remaining scores equal to m (n=1 is degenerate).
    """
    if not 1 <= n <= 100 or not 1 <= maximum <= 100:
        raise ValueError('Outside original p01853 input bounds')
    return Fraction((n // 2 if n % 2 else n // 2 - 1) * maximum, n)


def stagecoach_times(sample_input):
    """Exact subset-ticket shortest paths for the original p00719 constraints."""
    tokens = sample_input.split()
    if len(tokens) > 100000 or any(not re.fullmatch(r'\d{1,3}', token) for token in tokens):
        raise ValueError('Original p00719 input differs')
    values, position, results = list(map(int, tokens)), 0, []
    while position + 5 <= len(values):
        n, m, roads, start, finish = values[position:position + 5]
        position += 5
        if (n, m, roads, start, finish) == (0, 0, 0, 0, 0):
            if position != len(values):
                raise ValueError('Unexpected data after p00719 terminator')
            return results
        if not (1 <= n <= 8 and 2 <= m <= 30 and 0 <= roads <= m * (m - 1) // 2
                and 1 <= start <= m and 1 <= finish <= m and start != finish):
            raise ValueError('Outside original p00719 bounds')
        if position + n + 3 * roads > len(values):
            raise ValueError('Incomplete p00719 dataset')
        tickets = values[position:position + n]
        position += n
        if any(not 1 <= ticket <= 10 for ticket in tickets):
            raise ValueError('Outside original p00719 ticket bounds')
        graph, seen = [[] for _ in range(m)], set()
        for _ in range(roads):
            a, b, distance = values[position:position + 3]
            position += 3
            key = tuple(sorted((a, b)))
            if not (1 <= a <= m and 1 <= b <= m and a != b and 1 <= distance <= 100) or key in seen:
                raise ValueError('Invalid original p00719 road')
            seen.add(key)
            graph[a - 1].append((b - 1, distance))
            graph[b - 1].append((a - 1, distance))
        states = [{} for _ in range(1 << n)]
        states[0][start - 1] = Fraction(0)
        for mask, cities in enumerate(states):
            for city, cost in cities.items():
                for destination, distance in graph[city]:
                    for index, horses in enumerate(tickets):
                        if not mask & (1 << index):
                            next_state = states[mask | (1 << index)]
                            candidate = cost + Fraction(distance, horses)
                            if destination not in next_state or candidate < next_state[destination]:
                                next_state[destination] = candidate
        arrivals = [state[finish - 1] for state in states if finish - 1 in state]
        results.append(min(arrivals) if arrivals else None)
    raise ValueError('Missing original p00719 terminator')


def reachable_masses(first, second, limit):
    reachable = [False] * (limit + 1)
    reachable[0] = True
    for mass in range(1, limit + 1):
        reachable[mass] = ((mass >= first and reachable[mass - first]) or
                           (mass >= second and reachable[mass - second]))
    return reachable


def check_sugar_water(sample_input, actual_output):
    """Check reachability, solubility, capacity and exact optimal sugar density."""
    inputs = sample_input.split()
    if len(inputs) != 6 or any(not re.fullmatch(r'\d{1,4}', token) for token in inputs):
        raise ValueError('Original p03601 input differs')
    a, b, c, d, e, limit = map(int, inputs)
    if not (1 <= a < b <= 30 and 1 <= c < d <= 30 and 1 <= e <= 100 and 100 * a <= limit <= 3000):
        raise ValueError('Outside original p03601 bounds')
    water, sugar = reachable_masses(100 * a, 100 * b, limit), reachable_masses(c, d, limit)
    largest, current = [], 0
    for mass in range(limit + 1):
        if sugar[mass]:
            current = mass
        largest.append(current)
    best = Fraction(0)
    for mass in range(1, limit + 1):
        if water[mass]:
            dissolved = largest[min(limit - mass, e * mass // 100)]
            best = max(best, Fraction(dissolved, mass + dissolved))
    tokens = actual_output.split()
    valid = len(tokens) == 2 and all(re.fullmatch(r'\d{1,4}', token) for token in tokens)
    total, dissolved = map(int, tokens) if valid else (0, 0)
    pure_water = total - dissolved
    valid = (valid and 0 < pure_water <= total <= limit and 0 <= dissolved <= limit
             and water[pure_water] and sugar[dissolved] and 100 * dissolved <= e * pure_water)
    density = Fraction(dissolved, total) if valid else None
    return {'oracle': 'p03601_original_reachable_soluble_optimal_density',
            'passed': density is not None and density == best,
            'actual_density': str(density) if density is not None else None, 'optimal_density': str(best)}


def check_sample(problem_id, sample_input, sample_output, actual_output):
    if problem_id not in SUPPORTED:
        raise ValueError('No source-qualified checker for this problem')
    if problem_id == 'p03601':
        return check_sugar_water(sample_input, actual_output)
    if problem_id == 'p00719':
        truth = stagecoach_times(sample_input)
        lines = actual_output.strip().splitlines()
        checks = []
        for expected, actual in zip(truth, lines):
            actual = actual.strip()
            if expected is None:
                checks.append(actual == 'Impossible')
            elif REAL.fullmatch(actual):
                checks.append(abs(Fraction(Decimal(actual)) - expected) <= Fraction(1, 1000))
            else:
                checks.append(False)
        return {'oracle': 'p00719_exact_ticket_subset_paths_absolute_error_at_most_0.001',
                'passed': len(lines) == len(truth) and all(checks),
                'exact_reference_times': [str(value) if value is not None else None for value in truth]}
    if problem_id == 'p01411':
        # Original Output paragraph explicitly permits absolute error <= 10^-6.
        expected, actual = sample_output.split(), actual_output.split()
        valid = len(expected) == len(actual) == 1 and REAL.fullmatch(expected[0]) and REAL.fullmatch(actual[0])
        with localcontext() as context:
            context.prec = 50
            difference = abs(Decimal(actual[0]) - Decimal(expected[0])) if valid else None
        return {'oracle': 'p01411_original_absolute_error_at_most_1e-6',
                'passed': difference is not None and difference <= Decimal('0.000001'),
                'absolute_error': str(difference) if difference is not None else None}
    inputs = sample_input.split()
    if len(inputs) != 2 or any(not re.fullmatch(r'\d{1,3}', token) for token in inputs):
        raise ValueError('Original p01853 input differs')
    n, maximum = map(int, inputs)
    optimum = maximum_mean_median_gap(n, maximum)
    tokens = actual_output.split()
    valid = len(tokens) == n and all(re.fullmatch(r'[+-]?\d{1,4}', token) for token in tokens)
    values = list(map(int, tokens)) if valid else []
    valid = valid and all(0 <= value <= maximum for value in values)
    gap = mean_median_gap(values) if valid else None
    return {'oracle': 'p01853_original_optimal_bounded_score_distribution',
            'passed': gap is not None and gap == optimum,
            'actual_gap': str(gap) if gap is not None else None, 'optimal_gap': str(optimum)}
