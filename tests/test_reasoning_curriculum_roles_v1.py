"""Actual input groups, gold independence and whole-group exclusion controls."""
from collections import Counter
import copy
import json
from pathlib import Path

import pytest

from slm_lab.reasoning_curriculum_roles_v1 import assignments,group_id,role


def entry(task,question,**meta):
    return dict(question=question,answer='unused',metadata=dict(source_dataset=task,**meta))


@pytest.mark.parametrize('task',['gcd','lcm'])
def test_integer_list_permutations_share_one_role(task):
    a=entry(task,'these numbers: 12, 18, 24',numbers=[12,18,24])
    b=entry(task,'these numbers: 24, 12, 18',numbers=[24,12,18])
    c=entry(task,'these numbers: 12, 18, 25',numbers=[12,18,25])
    assert group_id(a)==group_id(b) and role(group_id(a))==role(group_id(b))
    assert group_id(a)!=group_id(c)


def test_arithmetic_permutation_and_operand_order_controls():
    def arithmetic(task,text):return entry(task,'Calculate '+text,**{'problem' if task=='bitwise_arithmetic' else 'expression':text})
    assert group_id(arithmetic('chain_sum','12 - 8 + 3'))==group_id(arithmetic('chain_sum','3 + 12 - 8'))
    assert group_id(arithmetic('products','-2 * 3 * 4'))==group_id(arithmetic('products','4 * 2 * -3'))
    assert group_id(arithmetic('bitwise_arithmetic','0x02 + 0x03'))==group_id(arithmetic('bitwise_arithmetic','3 + 2'))
    assert group_id(arithmetic('bitwise_arithmetic','2 << 3'))!=group_id(arithmetic('bitwise_arithmetic','3 << 2'))


def test_character_renaming_permutation_and_symmetric_isomorphism_groups():
    def ransom(note,magazine):return entry('ransom_note','Ransom note: '+note+'\nMagazine: '+magazine+'\n',ransom_note=note,magazine=magazine)
    assert group_id(ransom('aab','abca'))==group_id(ransom('yxx','xyxz'))
    assert group_id(ransom('aab','abca'))!=group_id(ransom('aab','abc'))
    def iso(a,b):return entry('isomorphic_strings',a+' '+b,words=[a,b])
    assert group_id(iso('aba','cfc'))==group_id(iso('xyx','efe'))==group_id(iso('cfc','aba'))
    assert group_id(iso('aba','cfc'))!=group_id(iso('aba','cff'))


def test_grid_symmetries_and_rotation_modulo_groups():
    def matrix(task,grid,turns=0):
        text='Rotate by '+str(turns*90)+' degrees clockwise:\n' if task=='rotate_matrix' else 'Grid:\n'
        text+='\n'.join(' '.join(map(str,row)) for row in grid)
        return entry(task,text,matrix=grid,num_rotations=turns)
    assert group_id(matrix('binary_matrix',[[0,1],[1,1]]))==group_id(matrix('binary_matrix',[[1,0],[1,1]]))
    assert group_id(matrix('rotten_oranges',[[2,1],[0,1]]))==group_id(matrix('rotten_oranges',[[0,2],[1,1]]))
    assert group_id(matrix('rotate_matrix',[[1,2],[3,4]],1))==group_id(matrix('rotate_matrix',[[3,1],[4,2]],5))
    assert group_id(matrix('rotate_matrix',[[1,2],[3,4]],1))!=group_id(matrix('rotate_matrix',[[1,2],[3,4]],2))


def test_gold_and_generator_seed_do_not_influence_roles_and_an_overlap_excludes_every_equivalent_input():
    a=entry('count_bits','How many 1 bits are there in the binary representation of the number 17?',number=17)
    b=copy.deepcopy(a);b['answer']='different gold';b['metadata'].update(solution='different',source_index=999)
    assert group_id(a)==group_id(b)
    rows=[dict(id='a',original_entry=a),dict(id='b',original_entry=b)]
    regular=assignments(rows,dict(a=[],b=[]))
    assert regular[0]['representative'] and not regular[1]['representative']
    assert regular[0]['split']==regular[1]['split']==role(group_id(a))
    excluded=assignments(rows,dict(a=[],b=[dict(protected_id='held:test')]))
    assert all(r['split']=='excluded_protected_text_overlap' for r in excluded)
    with pytest.raises(ValueError):assignments(rows,dict(a=[]))


def test_complete_real3072_population_has_every_task_and_disjoint_input_groups():
    rows=list(map(json.loads,Path('data/public-reasoning-generator-qualified-candidates-v3/entries.jsonl').read_text().splitlines()))
    assert len(rows)==3072
    result=assignments(rows,{r['id']:[] for r in rows})
    groups={split:{r['group_id'] for r in result if r['split']==split} for split in ['train','validation']}
    assert groups['train'] and groups['validation'] and not groups['train']&groups['validation']
    assert len({r['group_id'] for r in result})<2988
    assert Counter(r['original_entry']['metadata']['source_dataset'] for r in rows)==Counter({task:256 for task in {
        'count_bits','gcd','lcm','chain_sum','bitwise_arithmetic','products','base_conversion','binary_matrix',
        'rotate_matrix','ransom_note','isomorphic_strings','rotten_oranges'}})
