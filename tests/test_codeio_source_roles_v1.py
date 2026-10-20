import copy

import pytest

from slm_lab.codeio_source_roles_v1 import FIELDS,assign,digest,verify


def test_transitive_repository_question_and_body_components_match_an_independent_graph_walk():
    rows=[]
    for i,values in enumerate([('r1','q1','b1'),('r1','q2','b2'),('r2','q2','b3'),('r3','q3','b3'),('r4','q4','b4')]):
        rows.append(dict(original_position=i,**{k:digest(v) for k,v in zip(FIELDS,values)}))
    remaining=set(range(len(rows)));groups=[]
    while remaining:
        group={min(remaining)};changed=True
        while changed:
            joined={j for j in remaining if any(rows[j][f]==rows[i][f] for i in group for f in FIELDS)}
            changed=not joined<=group;group|=joined
        groups.append(group);remaining-=group
    assert groups==[{0,1,2,3},{4}]
    roles=assign(rows)
    assert {frozenset(i for i,r in roles.items() if r['component_sha256']==g) for g in {r['component_sha256'] for r in roles.values()}}=={frozenset(g) for g in groups}
    assert assign(list(reversed(rows)))==roles and verify(rows,roles)['declared_cross_role_intersections']==0
    broken=copy.deepcopy(roles);broken[0]['source_role']='reserved' if roles[0]['source_role']=='train' else 'train'
    with pytest.raises(ValueError):verify(rows,broken)


def test_bad_source_id_or_incomplete_grouping_cannot_receive_a_role():
    row=dict(original_position=0,**{f:digest(f) for f in FIELDS})
    for rows in [[row,row],[dict(row,original_position=True)],[dict(row,question_group_sha256='short')]]:
        with pytest.raises(ValueError):assign(rows)
