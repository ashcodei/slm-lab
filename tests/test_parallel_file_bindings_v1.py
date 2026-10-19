import os
import pytest
from slm_lab.corpus import file_hash
from slm_lab.parallel_file_bindings_v1 import ParallelFileBindings


def test_parallel_admission_hashes_every_file_and_rejects_same_size_preserved_timestamp_runtime_change(tmp_path):
    files=[]
    for i in range(7):
        p=tmp_path/f'{i}.py';p.write_text('first');files.append(p)
    bindings=ParallelFileBindings({str(p):file_hash(p) for p in files});bindings.check(full=True)
    p=files[2];stat=p.stat();p.write_text('wrong');os.utime(p,ns=(stat.st_atime_ns,stat.st_mtime_ns))
    with pytest.raises(ValueError,match='changed'):bindings.check()
