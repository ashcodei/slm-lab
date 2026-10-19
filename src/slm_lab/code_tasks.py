"""Private test cases and reference implementations for a local coding assessment.
These tasks are outside the training collectors. The model sees specifications only.
"""
import random

# name, arguments, specification, intended scaling, Python reference, JS reference
TASKS=[
('sum_even','xs','Return the sum of all even integers in xs. Empty input returns 0.','linear',
 'def sum_even(xs):\n    return sum(x for x in xs if x%2==0)',
 'function sum_even(xs) { let s=0; for (const x of xs) if (x%2===0) s+=x; return s; }'),
('count_runs','xs','Return the number of contiguous runs of equal integers in xs. Empty input returns 0.','linear',
 'def count_runs(xs):\n    return sum(i==0 or x!=xs[i-1] for i,x in enumerate(xs))',
 'function count_runs(xs) { let n=0; for(let i=0;i<xs.length;i++) if(i===0||xs[i]!==xs[i-1]) n++; return n; }'),
('prefix_sums','xs','Return a list of inclusive prefix sums of xs. Empty input returns an empty list.','linear',
 'def prefix_sums(xs):\n    out=[]; s=0\n    for x in xs:\n        s+=x; out.append(s)\n    return out',
 'function prefix_sums(xs) { let s=0; const out=[]; for(const x of xs) {s+=x;out.push(s);} return out; }'),
('first_duplicate','xs','Return the first integer whose second occurrence is encountered while scanning xs from left to right. Return null/None when none repeats.','linear',
 'def first_duplicate(xs):\n    seen=set()\n    for x in xs:\n        if x in seen: return x\n        seen.add(x)\n    return None',
 'function first_duplicate(xs) { const seen=new Set(); for(const x of xs) {if(seen.has(x)) return x;seen.add(x);} return null; }'),
('count_target_pairs','xs, target','Count pairs of distinct indices i<j such that xs[i]+xs[j]==target. Count duplicate values at different indices separately.','linear',
 'def count_target_pairs(xs,target):\n    seen={}; count=0\n    for x in xs:\n        count+=seen.get(target-x,0); seen[x]=seen.get(x,0)+1\n    return count',
 'function count_target_pairs(xs,target) {const seen=new Map();let count=0;for(const x of xs){count+=seen.get(target-x)||0;seen.set(x,(seen.get(x)||0)+1);}return count;}'),
('lower_bound','xs, target','xs is sorted in ascending order. Return the first index with xs[index]>=target, or xs length when none exists.','logarithmic',
 'def lower_bound(xs,target):\n    lo=0;hi=len(xs)\n    while lo<hi:\n        mid=(lo+hi)//2\n        if xs[mid]<target:lo=mid+1\n        else:hi=mid\n    return lo',
 'function lower_bound(xs,target){let lo=0,hi=xs.length;while(lo<hi){let mid=Math.floor((lo+hi)/2);if(xs[mid]<target)lo=mid+1;else hi=mid;}return lo;}'),
('max_window_sum','xs, k','Return the largest sum of a contiguous window of exactly k integers. For k<=0 or k>xs length return null/None. Negative elements are allowed.','linear',
 'def max_window_sum(xs,k):\n    if k<=0 or k>len(xs):return None\n    s=sum(xs[:k]);best=s\n    for i in range(k,len(xs)):\n        s+=xs[i]-xs[i-k];best=max(best,s)\n    return best',
 'function max_window_sum(xs,k){if(k<=0||k>xs.length)return null;let s=0;for(let i=0;i<k;i++)s+=xs[i];let best=s;for(let i=k;i<xs.length;i++){s+=xs[i]-xs[i-k];best=Math.max(best,s);}return best;}'),
('stable_unique','xs','Return the distinct integers of xs in first-occurrence order. Do not sort the result.','linear',
 'def stable_unique(xs):\n    seen=set();out=[]\n    for x in xs:\n        if x not in seen:seen.add(x);out.append(x)\n    return out',
 'function stable_unique(xs){const seen=new Set(),out=[];for(const x of xs)if(!seen.has(x)){seen.add(x);out.push(x);}return out;}'),
('max_subarray','xs','Return the largest sum of a nonempty contiguous subarray. For empty xs return 0. For all-negative input return its largest element.','linear',
 'def max_subarray(xs):\n    if not xs:return 0\n    cur=best=xs[0]\n    for x in xs[1:]:cur=max(x,cur+x);best=max(best,cur)\n    return best',
 'function max_subarray(xs){if(!xs.length)return 0;let cur=xs[0],best=cur;for(let i=1;i<xs.length;i++){cur=Math.max(xs[i],cur+xs[i]);best=Math.max(best,cur);}return best;}'),
('rotate_left','xs, k','Return xs rotated left by k positions. Negative k rotates right. Return a new list; empty xs remains empty.','linear',
 'def rotate_left(xs,k):\n    if not xs:return []\n    k%=len(xs);return xs[k:]+xs[:k]',
 'function rotate_left(xs,k){if(!xs.length)return [];k=((k%xs.length)+xs.length)%xs.length;return xs.slice(k).concat(xs.slice(0,k));}'),
('is_balanced','s','Return whether parentheses, brackets, and braces in s are correctly balanced and nested. Ignore every other character. Empty string is balanced.','linear',
 'def is_balanced(s):\n    stack=[];pairs={")":"(","]":"[","}":"{"}\n    for c in s:\n        if c in "([{":stack.append(c)\n        elif c in pairs:\n            if not stack or stack.pop()!=pairs[c]:return False\n    return not stack',
 'function is_balanced(s){const stack=[],pairs={")":"(","]":"[","}":"{"};for(const c of s){if("([{ ".trim().includes(c))stack.push(c);else if(c in pairs){if(!stack.length||stack.pop()!==pairs[c])return false;}}return stack.length===0;}'),
('longest_unique','s','Return the length of the longest contiguous substring whose characters are all distinct. Inputs use Unicode BMP characters. Empty string returns 0.','linear',
 'def longest_unique(s):\n    last={};start=best=0\n    for i,c in enumerate(s):\n        start=max(start,last.get(c,-1)+1);last[c]=i;best=max(best,i-start+1)\n    return best',
 'function longest_unique(s){const last=new Map();let start=0,best=0;for(let i=0;i<s.length;i++){let c=s[i];start=Math.max(start,(last.has(c)?last.get(c):-1)+1);last.set(c,i);best=Math.max(best,i-start+1);}return best;}')]


def inputs(name,large=False):
    rng=random.Random('hidden-tests-'+name)
    if large:
        result=[]
        for n in [2048,8192,32768]:
            if name=='lower_bound':args=[list(range(n)),n-1]
            elif name=='longest_unique':args=[''.join(chr(1024+i) for i in range(n))]
            elif name=='is_balanced':args=['('*n+')'*n]
            else:
                xs=list(range(n)) if name in {'first_duplicate','stable_unique'} else [rng.randrange(-100,101) for _ in range(n)]
                args=[xs]
                if name=='count_target_pairs':args.append(1000000)
                if name=='max_window_sum':args.append(n//2)
                if name=='rotate_left':args.append(-n//3)
            result.append(args)
        return result
    result=[]
    if name in {'longest_unique','is_balanced'}:
        strings=['','a','aaaa','abcabcbb','([{}])','([)]','hello (world) []','()'*50]+[
            ''.join(rng.choice('abc()[]{}') for _ in range(rng.randrange(60))) for _ in range(50)]
        return [[s] for s in strings]
    arrays=[[],[0],[1],[-4],[-2,-1,-3],[1,2,1,3,2],[0,0,0]]+[
        [rng.randrange(-20,21) for _ in range(rng.randrange(50))] for _ in range(60)]
    for xs in arrays:
        if name=='lower_bound':xs=sorted(xs)
        extras=[[]]
        if name in {'count_target_pairs','lower_bound'}:extras=[[-40],[0],[40],[rng.randrange(-20,21)]]
        if name=='max_window_sum':extras=[[0],[1],[len(xs)],[len(xs)+1],[max(1,len(xs)//2)]]
        if name=='rotate_left':extras=[[-31],[-1],[0],[1],[31]]
        result.extend([[xs]+extra for extra in extras])
    return result


def reference(task):
    namespace={};exec(task[4],namespace);return namespace[task[0]]


def request(task,language,code,reference_mode=False):
    fn=reference(task)
    return {'language':language,'code':code,'function':task[0],
        'cases':[{'args':args,'expected':fn(*args)} for args in inputs(task[0])],
        'stress':[{'args':args,'expected':fn(*args)} for args in inputs(task[0],True)],
        'repetitions':50 if task[0]=='lower_bound' else 1,'reference_mode':reference_mode}
