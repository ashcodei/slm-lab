"""Bind actually executed revised lineage and source-factory control logs."""
from .corpus import file_hash


def bind_tests(files,root):
    for name,count in [('public-pretraining-lineage-drivers-v1-final2-tests',16),('public-pretraining-human-policy-v3-tests',2),
        ('public-pretraining-broad-policy-v3-tests',2),('public-pretraining-machine-policy-v2-tests',2)]:
        path=root/'reports'/(name+'.log')
        if str(count)+' passed' not in path.read_text():raise ValueError('All revised lineage and actual factory/corruption controls must pass')
        files[str(path)]=file_hash(path)
    for name in ['test_public_pretraining_lineage_drivers_v1.py','test_public_pretraining_human_policy_v3.py',
        'test_broad_anchored_policy_v3.py','test_public_pretraining_machine_policy_v2.py']:
        path=root/'tests'/name
        files[str(path)]=file_hash(path)
    path=root/'reports/public-pretraining-lineage-file-map-v1.json'
    files[str(path)]=file_hash(path)
