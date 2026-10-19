"""Load only reviewed unchanged public generators in a temporary namespace.

The upstream root/family/coaching initializers also import unrelated tasks and
optional packages. Narrow namespace packages expose the actual selected source
modules and coaching classes without installing packages or editing upstream.
Use only in an isolated CPU data-preparation process, never a neural worker.
"""
from contextlib import contextmanager
import hashlib
import importlib
import json
from pathlib import Path
import sys
import types

REVISION='49b07130b3fcd12f2d064bba7c43869543a0e7e7'
ARITHMETIC=('count_bits','gcd','lcm','chain_sum','bitwise_arithmetic','products')
ALGORITHMIC=('base_conversion','binary_matrix','rotate_matrix','ransom_note','isomorphic_strings','rotten_oranges')


def verify(folder):
    folder=Path(folder).resolve();receipt=json.loads((folder/'collection.json').read_text())
    if receipt['revision']!=REVISION or receipt['selected_source_tasks']!=list(ARITHMETIC+ALGORITHMIC):
        raise ValueError('Exact reviewed public source revision and selected task inventory required')
    result={str(folder/'collection.json'):hashlib.sha256((folder/'collection.json').read_bytes()).hexdigest()}
    for row in receipt['files']:
        path=folder/'upstream'/row['path'];raw=path.read_bytes()
        if (hashlib.sha256(raw).hexdigest()!=row['sha256'] or len(raw)!=row['bytes']
                or hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()!=row['git_blob_sha1']):
            raise ValueError('Every exact original Git blob and SHA256 must match before source execution')
        result[str(path)]=row['sha256']
    return result


@contextmanager
def load(folder):
    folder=Path(folder).resolve();verify(folder)
    if any(name=='reasoning_gym' or name.startswith('reasoning_gym.') for name in sys.modules):
        raise ValueError('A fresh isolated process namespace is required')
    previous=sys.dont_write_bytecode;sys.dont_write_bytecode=True
    try:
        for name in ['reasoning_gym','reasoning_gym.arithmetic','reasoning_gym.algorithmic','reasoning_gym.coaching']:
            package=types.ModuleType(name);package.__package__=name
            package.__path__=[str(folder/'upstream'/Path(*name.split('.')))];sys.modules[name]=package
        attributes=importlib.import_module('reasoning_gym.coaching.attributes')
        base=importlib.import_module('reasoning_gym.coaching.base_curriculum')
        coaching=sys.modules['reasoning_gym.coaching'];coaching.BaseCurriculum=base.BaseCurriculum
        for name in ['AttributeDefinition','RangeAttributeDefinition','ScalarAttributeDefinition']:
            setattr(coaching,name,getattr(attributes,name))
        factory=importlib.import_module('reasoning_gym.factory')
        for category,names in [('arithmetic',ARITHMETIC),('algorithmic',ALGORITHMIC)]:
            for name in names:importlib.import_module('reasoning_gym.'+category+'.'+name)
        if set(factory.DATASETS)!=set(ARITHMETIC+ALGORITHMIC):
            raise ValueError('Only the twelve reviewed original generators may register')
        yield factory
        verify(folder)
    finally:
        for name in list(sys.modules):
            if name=='reasoning_gym' or name.startswith('reasoning_gym.'):del sys.modules[name]
        sys.dont_write_bytecode=previous
