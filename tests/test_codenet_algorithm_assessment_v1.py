from fractions import Fraction
import heapq
from itertools import combinations, product, permutations
import random

import pytest

from slm_lab.codenet_algorithm_assessment_v1 import (PROBLEMS, authored_inputs, check_output,
    expected_output, fuel_answers, matching, subset_counts)


@pytest.mark.parametrize('problem', PROBLEMS)
def test_all_authored_inputs_respect_original_bounds_and_have_valid_oracle_witnesses(problem):
    cases = authored_inputs(problem)
    assert cases and all(c['kind'] in {'functional', 'stress'} for c in cases)
    for case in cases:
        expected = expected_output(problem, case['input'])
        assert len(case['input'].encode()) <= 4 * 1024**2
        assert len(expected.encode()) <= 1024**2
        assert check_output(problem, case['input'], expected, expected)
        assert not check_output(problem, case['input'], expected, 'not an answer')


def test_distinct_subset_dp_matches_direct_combinations_including_zero():
    for size in range(1, 4):
        for total in range(13):
            truth = sum(sum(values) == total for values in combinations(range(total+1), size))
            assert subset_counts()[size][total] == truth


def test_window_oracle_matches_explicit_sums_with_all_negative_inputs():
    rng = random.Random(41)
    for _ in range(100):
        a = [rng.randint(-10, 10) for _ in range(rng.randint(1, 10))]
        k = rng.randint(1, len(a))
        text = f'{len(a)} {k}\n' + '\n'.join(map(str,a)) + '\n0 0\n'
        assert int(expected_output('p00439',text)) == max(sum(a[i:i+k]) for i in range(len(a)-k+1))


def test_stagecoach_subset_oracle_matches_enumerated_paths_and_ticket_orders():
    tickets = [1, 3, 7]
    edges = [(1,2,4),(2,3,11),(3,4,17),(1,4,80),(1,3,40),(2,4,50)]
    distances = {frozenset((a,b)):w for a,b,w in edges}
    truth = []
    for count in range(3):
        for interior in permutations([2,3],count):
            path = (1,) + interior + (4,)
            for order in permutations(tickets,len(path)-1):
                truth.append(sum(Fraction(distances[frozenset((a,b))],t) for (a,b),t in zip(zip(path,path[1:]),order)))
    text='3 4 6 1 4\n1 3 7\n'+'\n'.join(' '.join(map(str,e)) for e in edges)+'\n0 0 0 0 0\n'
    result=expected_output('p00719',text)
    assert abs(Fraction(result.strip())-min(truth)) < Fraction(1,1000000)
    assert not check_output('p00719',text,result,'Impossible')


def naive_prime(n):
    return n >= 2 and all(n % d for d in range(2,n))


def test_prime_sum_oracle_matches_trial_division_and_explicit_ranges():
    primes=[x for x in range(2,101) if naive_prime(x)]
    for n in range(2,101):
        truth=sum(sum(primes[a:b])==n for a in range(len(primes)) for b in range(a+1,len(primes)+1))
        assert int(expected_output('p00836',f'{n}\n0\n'))==truth


def test_damage_dp_matches_independent_shortest_path_search():
    rng=random.Random(43)
    for _ in range(40):
        h=rng.randint(1,30)
        spells=[(rng.randint(1,20),rng.randint(1,20)) for _ in range(4)]
        distance=[10**9]*(h+1);distance[0]=0;queue=[(0,0)]
        while queue:
            cost,done=heapq.heappop(queue)
            if distance[done]!=cost:continue
            for amount,price in spells:
                other=min(h,done+amount)
                if cost+price<distance[other]:
                    distance[other]=cost+price;heapq.heappush(queue,(cost+price,other))
        text=f'{h} 4\n'+'\n'.join(f'{a} {b}' for a,b in spells)+'\n'
        assert int(expected_output('p02787',text))==distance[h]


def test_fuel_oracle_matches_expanded_remaining_fuel_state_search():
    rng=random.Random(44)
    for _ in range(25):
        n,capacity=rng.randint(2,5),rng.randint(1,6)
        edges=[(a,b,rng.randint(1,8)) for a in range(1,n) for b in range(a+1,n+1) if rng.random()<.6]
        queries=[(a,b) for a in range(1,n+1) for b in range(1,n+1) if a!=b]
        expected=[]
        graph=[[] for _ in range(n)]
        for a,b,w in edges:graph[a-1].append((b-1,w));graph[b-1].append((a-1,w))
        for start,finish in queries:
            distance={(start-1,capacity):0};queue=[(0,start-1,capacity)]
            while queue:
                count,vertex,fuel=heapq.heappop(queue)
                if distance[vertex,fuel]!=count:continue
                options=[(vertex,capacity,count+1)]
                options.extend((other,fuel-w,count) for other,w in graph[vertex] if fuel>=w)
                for other,remaining,cost in options:
                    if cost<distance.get((other,remaining),10**9):
                        distance[other,remaining]=cost;heapq.heappush(queue,(cost,other,remaining))
            expected.append(min((c for (v,f),c in distance.items() if v==finish-1),default=-1))
        assert fuel_answers(n,edges,capacity,queries)==expected


def test_adjacent_bound_optimum_matches_full_small_feasible_assignments():
    for b in product(range(3),repeat=3):
        truth=max(sum(a) for a in product(range(3),repeat=4)
                  if all(max(a[i],a[i+1])<=b[i] for i in range(3)))
        assert int(expected_output('p02917','4\n'+' '.join(map(str,b))+'\n'))==truth


def test_range_oracle_matches_direct_substring_scans():
    for n in range(2,7):
        queries=[(a,b) for a in range(1,n) for b in range(a+1,n+1)]
        for chars in product('AC',repeat=n):
            s=''.join(chars)
            text=f'{n} {len(queries)}\n{s}\n'+'\n'.join(f'{a} {b}' for a,b in queries)+'\n'
            truth=[s[a-1:b].count('AC') for a,b in queries]
            assert list(map(int,expected_output('p03087',text).split()))==truth


def test_unique_letter_subsequence_formula_matches_position_subsets():
    for n in range(1,7):
        for chars in product('abc',repeat=n):
            s=''.join(chars)
            truth=0
            for mask in range(1,1<<n):
                selected=[s[i] for i in range(n) if mask>>i&1]
                truth+=len(selected)==len(set(selected))
            assert int(expected_output('p03095',f'{n}\n{s}\n'))==truth


def test_matching_augmentations_match_exhaustive_partial_assignment():
    rng=random.Random(47)
    for n in range(1,6):
        for _ in range(8):
            xs,ys=list(range(2*n)),list(range(2*n));rng.shuffle(xs);rng.shuffle(ys)
            red,blue=list(zip(xs[:n],ys[:n])),list(zip(xs[n:],ys[n:]))
            def brute(i,used):
                if i==n:return 0
                return max([brute(i+1,used)]+[1+brute(i+1,used|{j}) for j in range(n) if j not in used
                    and red[i][0]<blue[j][0] and red[i][1]<blue[j][1]])
            assert matching(red,blue)==brute(0,set())


def test_prime_range_prefix_matches_independent_trial_division():
    queries=[(a,b) for a in range(1,50,2) for b in range(a,50,2)]
    truth=[sum(naive_prime(x) and naive_prime((x+1)//2) for x in range(a,b+1,2)) for a,b in queries]
    text=f'{len(queries)}\n'+'\n'.join(f'{a} {b}' for a,b in queries)+'\n'
    assert list(map(int,expected_output('p03476',text).split()))==truth


def test_triple_count_matches_cartesian_products_with_ties():
    rng=random.Random(48)
    for _ in range(50):
        n=rng.randint(1,6);a,b,c=[[rng.randint(1,6) for _ in range(n)] for _ in range(3)]
        text=f'{n}\n'+'\n'.join(' '.join(map(str,row)) for row in [a,b,c])+'\n'
        assert int(expected_output('p03557',text))==sum(x<y<z for x,y,z in product(a,b,c))


def test_sugar_property_accepts_multiple_optima_and_rejects_unreachable_equal_density():
    text='1 2 1 2 100 1000\n'
    assert check_output('p03601',text,'200 100','200 100')
    assert check_output('p03601',text,'200 100','400 200')
    assert not check_output('p03601',text,'200 100','300 150')
    rng=random.Random(49)
    for _ in range(20):
        c,d=sorted(rng.sample(range(1,11),2));limit=rng.randint(100,400);e=rng.randint(1,100)
        water={100*a+200*b for a in range(5) for b in range(3) if 0<100*a+200*b<=limit}
        sugar={c*a+d*b for a in range(limit//c+1) for b in range(limit//d+1) if c*a+d*b<=limit}
        truth=max(Fraction(s,w+s) for w in water for s in sugar if w+s<=limit and 100*s<=e*w)
        total,s=map(int,expected_output('p03601',f'1 2 {c} {d} {e} {limit}\n').split())
        assert Fraction(s,total)==truth


def test_maximal_path_properties_accept_reversal_but_reject_incomplete_or_invalid_paths():
    text='5 4\n1 2\n2 3\n3 4\n4 5\n'
    for answer in ['5\n1 2 3 4 5\n','5\n5 4 3 2 1\n']:
        assert check_output('p03746',text,'',answer)
    for answer in ['2\n1 2','5\n1 2 3 4 4','5\n1 2 3 4 6','4\n1 2 3','3\n1 3 5']:
        assert not check_output('p03746',text,'',answer)
