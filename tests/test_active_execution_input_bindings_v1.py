from pathlib import Path

import pytest

from slm_lab.active_execution_input_bindings_v1 import input_bindings,merge_bindings
from slm_lab.corpus import file_hash
from slm_lab.expanded_execution_corpus_v1 import ExpandedExecutionCorpus


def test_all_current_pack_files_and_reward_ledgers_are_bound_without_unrelated_old_training_states():
    root=Path(__file__).resolve().parents[1];corpus=ExpandedExecutionCorpus(root);files=input_bindings(root,corpus)
    for pack in corpus.packs.values():
        assert all(files[str(pack.folder/name)]==sha for name,sha in pack.manifest['files_sha256'].items())
    for name in ['reports/bounded-code-probes-v1/observations.jsonl',
                 'reports/causal-code-rewards-v2/observations.jsonl',
                 'data/apps-execution-rewards-v1/catalog.jsonl']:
        assert files[str(root/name)]==file_hash(root/name)
    assert str(root/'reports/fast-joint-step1024-readback-v1/resume-step1024.pt') not in files
    assert sum(Path(p).stat().st_size for p in files)<4*1024**3


def test_two_versions_of_one_actual_runtime_input_reject_instead_of_overwriting(tmp_path):
    path=tmp_path/'runtime-input';path.write_text('original');sha=file_hash(path)
    files={str(path):sha};merge_bindings(files,{str(path):sha})
    with pytest.raises(ValueError,match='Conflicting'):
        merge_bindings(files,{str(path):'0'*64})
    assert files[str(path)]==sha
