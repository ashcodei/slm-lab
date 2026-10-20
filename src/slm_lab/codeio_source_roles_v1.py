"""Fix complete repository/question/body components before source execution."""
import hashlib
import json
import re

FIELDS=('repository_group_sha256','question_group_sha256','whole_program_group_sha256')
SEED='CodeIO-repository-question-whole-body-v1:2026-10-20'
HEX=re.compile('[0-9a-f]{64}')


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def assign(records):
    parents={};first={}
    def find(p):
        while parents[p]!=p:parents[p]=parents[parents[p]];p=parents[p]
        return p
    def union(a,b):
        a,b=find(a),find(b);parents[max(a,b)]=min(a,b)
    for r in records:
        p=r['original_position']
        if type(p) is not int or p<0 or p in parents:raise ValueError('Every distinct original source position is required')
        parents[p]=p
        for field in FIELDS:
            value=r[field]
            if not isinstance(value,str) or HEX.fullmatch(value) is None:raise ValueError('Each declared source grouping needs a complete SHA256')
            key=(field,value)
            if key in first:union(p,first[key])
            else:first[key]=p
    groups={}
    for p in sorted(parents):groups.setdefault(find(p),[]).append(p)
    roles={}
    for members in groups.values():
        group=digest(members);reserved=int(digest([SEED,group]),16)%10==0
        for p in members:roles[p]=dict(source_role='reserved' if reserved else 'train',
            component_sha256=group,component_size=len(members),optimization_forbidden=reserved,role_fixed_before_fresh_execution=True)
    return roles


def verify(records,roles):
    if roles!=assign(records):raise ValueError('Original fixed source components and complete assignment must match')
    owners={}
    for row in records:
        role=roles[row['original_position']]['source_role']
        for field in FIELDS:
            key=(field,row[field])
            if key in owners and owners[key]!=role:raise ValueError('A declared source component crossed roles')
            owners[key]=role
    return dict(original_rows=len(records),components=len({r['component_sha256'] for r in roles.values()}),
        train_rows=sum(r['source_role']=='train' for r in roles.values()),reserved_rows=sum(r['source_role']=='reserved' for r in roles.values()),
        largest_component=max(r['component_size'] for r in roles.values()),declared_cross_role_intersections=0,
        semantic_or_historical_independence_certified=False)
