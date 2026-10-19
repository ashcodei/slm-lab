"""Independent bounded answers and strict serialization for selected RG tasks.

The question's actual inputs must agree with the source metadata. Published
solution fields are never oracle inputs. Model answers are compared as text;
they are never evaluated as Python. This is our explicit canonical protocol,
not a claim to reproduce upstream fractional or cascade scoring.
"""
import ast
from collections import Counter
import re

CONFIGS={
    'count_bits':dict(min_n=1,max_n=65535),
    'gcd':dict(min_numbers=2,max_numbers=4,min_value=1,max_value=1000),
    'lcm':dict(min_numbers=2,max_numbers=3,min_value=1,max_value=100),
    'chain_sum':dict(min_terms=2,max_terms=6,min_digits=1,max_digits=3),
    'bitwise_arithmetic':dict(difficulty=2),
    'products':dict(min_terms=2,max_terms=3,min_digits=1,max_digits=3),
    'base_conversion':dict(min_base=2,max_base=36,min_value=0,max_value=65535),
    'binary_matrix':dict(min_n=2,max_n=7,p_zero=.25),
    'rotate_matrix':dict(min_n=2,max_n=7,min_rotations=0,max_rotations=8),
    'ransom_note':dict(min_note_length=1,max_note_length=10,min_magazine_length=2,max_magazine_length=30),
    'isomorphic_strings':dict(min_string_length=2,max_string_length=12),
    'rotten_oranges':dict(min_n=2,max_n=9,p_oranges=.85,p_rotten=.2),
}


def expression(text):
    tree=ast.parse(text,mode='eval')
    if len(list(ast.walk(tree)))>128:raise ValueError('Bounded arithmetic expression required')
    def visit(node):
        if isinstance(node,ast.Constant) and type(node.value) is int:value=node.value
        elif isinstance(node,ast.UnaryOp) and type(node.op) in (ast.USub,ast.UAdd):
            value=visit(node.operand);value=-value if isinstance(node.op,ast.USub) else value
        elif isinstance(node,ast.BinOp) and type(node.op) in (ast.Add,ast.Sub,ast.Mult,ast.LShift,ast.RShift):
            left,right=visit(node.left),visit(node.right)
            if isinstance(node.op,(ast.LShift,ast.RShift)) and not 0<=right<=32:raise ValueError('Explicit bounded shift required')
            if isinstance(node.op,ast.Add):value=left+right
            elif isinstance(node.op,ast.Sub):value=left-right
            elif isinstance(node.op,ast.Mult):value=left*right
            elif isinstance(node.op,ast.LShift):value=left<<right
            else:value=left>>right
        else:raise ValueError('Only literal bounded arithmetic is allowed; no names, calls or attributes')
        if abs(value).bit_length()>512:raise ValueError('Bounded arithmetic values required')
        return value
    return visit(tree.body)


def matrix_text(matrix):return '\n'.join(' '.join(str(value) for value in row) for row in matrix)


def matrix_inputs(entry):
    lines=entry['question'].rstrip().splitlines();rows=[]
    for line in reversed(lines):
        if not re.fullmatch(r'\d+(?: +\d+)*',line):break
        rows.append([int(v) for v in line.split()])
    rows.reverse();n=len(rows)
    if not 1<=n<=30 or any(len(row)!=n for row in rows) or rows!=entry['metadata']['matrix']:
        raise ValueError('Every complete square matrix in the actual question must match the declared input')
    return rows


def expected(entry):
    question=entry['question'];meta=entry['metadata'];task=meta['source_dataset']
    if task=='count_bits':
        match=re.fullmatch(r'How many 1 bits are there in the binary representation of the number (\d+)\?',question)
        if not match or int(match[1])!=meta['number']:raise ValueError('Question number must match')
        n=int(match[1]);result=0
        while n:n,remainder=divmod(n,2);result+=remainder
        return str(result)
    if task in ('gcd','lcm'):
        match=re.search(r'these numbers: (\d+(?:, \d+)*)',question)
        if not match:raise ValueError('The actual complete positive integer list is required')
        numbers=[int(v) for v in match[1].split(', ')]
        if numbers!=meta['numbers'] or not numbers or any(v<1 or v>1000 for v in numbers):raise ValueError('Question operands must match their bounded inputs')
        if task=='gcd':
            value=numbers[0]
            for other in numbers[1:]:
                while other:value,other=other,value%other
            return str(value)
        exponents={}
        for number in numbers:
            divisor=2;current={}
            while divisor*divisor<=number:
                while number%divisor==0:current[divisor]=current.get(divisor,0)+1;number//=divisor
                divisor+=1
            if number>1:current[number]=current.get(number,0)+1
            for prime,count in current.items():exponents[prime]=max(count,exponents.get(prime,0))
        result=1
        for prime,count in exponents.items():result*=prime**count
        return str(result)
    if task in ('chain_sum','products','bitwise_arithmetic'):
        text=meta['problem' if task=='bitwise_arithmetic' else 'expression']
        if text not in question:raise ValueError('Every literal arithmetic input must appear unchanged in the question')
        value=expression(text)
        return hex(value) if task=='bitwise_arithmetic' else str(value)
    if task=='base_conversion':
        source,target=meta['source_base'],meta['target_base'];text=meta['source_repr']
        if not 2<=source<=36 or not 2<=target<=36 or source==target:raise ValueError('Explicit distinct bounded bases required')
        def base_name(number):return {2:'binary',16:'hexadecimal'}.get(number,'base-'+str(number))
        suffix='Now, convert the '+base_name(source)+' number '+text+' to '+base_name(target)+'\n'
        if not question.endswith(suffix):raise ValueError('The actual question must contain those exact bases and source numeral')
        value=int(text,source)
        if value!=meta['decimal_value'] or value<0:raise ValueError('Exact nonnegative positional source value required')
        digits='0123456789abcdefghijklmnopqrstuvwxyz';result=''
        while value:value,remainder=divmod(value,target);result=digits[remainder]+result
        return result or '0'
    if task in ('binary_matrix','rotate_matrix','rotten_oranges'):
        matrix=matrix_inputs(entry);n=len(matrix)
        if task=='binary_matrix':
            if any(v not in (0,1) for row in matrix for v in row):raise ValueError('Binary cells required')
            zeros=[(r,c) for r in range(n) for c in range(n) if matrix[r][c]==0]
            if not zeros:raise ValueError('At least one zero is required')
            # Closed-form Manhattan distances, independent of upstream BFS.
            return matrix_text([[min(abs(r-a)+abs(c-b) for a,b in zeros) for c in range(n)] for r in range(n)])
        if task=='rotate_matrix':
            match=re.search(r'by (\d+) degrees clockwise:',question)
            if not match or int(match[1])!=90*meta['num_rotations']:raise ValueError('Actual declared rotation must match')
            turns=meta['num_rotations']%4;result=[[0]*n for _ in range(n)]
            for r in range(n):
                for c in range(n):
                    out_r,out_c=(r,c) if turns==0 else (c,n-1-r) if turns==1 else (n-1-r,n-1-c) if turns==2 else (n-1-c,r)
                    result[out_r][out_c]=matrix[r][c]
            return matrix_text(result)
        if any(v not in (0,1,2) for row in matrix for v in row):raise ValueError('Original orange/empty cells required')
        grid=[row[:] for row in matrix];minutes=0
        while any(v==1 for row in grid for v in row):
            changes=[(r,c) for r in range(n) for c in range(n) if grid[r][c]==1 and any(
                0<=r+dr<n and 0<=c+dc<n and grid[r+dr][c+dc]==2 for dr,dc in [(1,0),(-1,0),(0,1),(0,-1)])]
            if not changes:return '-1'
            for r,c in changes:grid[r][c]=2
            minutes+=1
        return str(minutes)
    if task=='ransom_note':
        match=re.search(r'Ransom note: ([a-z]*)\nMagazine: ([a-z]*)\n$',question)
        if not match or [match[1],match[2]]!=[meta['ransom_note'],meta['magazine']]:raise ValueError('Both exact question strings required')
        needed,available=Counter(match[1]),Counter(match[2])
        return str(all(available[letter]>=count for letter,count in needed.items()))
    if task=='isomorphic_strings':
        words=question.rstrip().splitlines()[-1].split()
        if words!=meta['words'] or len(words)!=2 or len(words[0])!=len(words[1]):raise ValueError('Both exact equal-length strings required')
        first,second=words
        return str(all((first[i]==first[j])==(second[i]==second[j]) for i in range(len(first)) for j in range(i)))
    raise ValueError('Every selected task requires an explicit independently implemented oracle')


def format_instruction(task):
    if task in ('binary_matrix','rotate_matrix'):return 'Use one row per line and single spaces between integer entries; do not use brackets or extra text.'
    if task in ('ransom_note','isomorphic_strings'):return 'Reply only True or False, with that capitalization.'
    if task=='bitwise_arithmetic':return 'Reply only a lowercase hexadecimal integer with a 0x prefix, or -0x for a negative result; omit unnecessary leading zeroes.'
    if task=='base_conversion':return 'Reply only lowercase digits 0-9 and a-z in the target base, without a base prefix or unnecessary leading zeroes.'
    return 'Reply only the base-10 integer, without unnecessary leading zeroes or extra text.'


def prompt(entry):return entry['question']+'\n'+format_instruction(entry['metadata']['source_dataset'])


def strict_reward(answer,entry):
    wanted=expected(entry)
    if entry['answer']!=wanted:raise ValueError('The independently calculated label differs from the public generator; no reward admission')
    return float(isinstance(answer,str) and answer.strip()==wanted)
