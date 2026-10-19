from copy import deepcopy
import gzip
import io
import json

import numpy as np
import pytest

from slm_lab.public_billion_candidate_v1 import CandidateWriter
from slm_lab.public_billion_readback_v1 import Rebuilder, budget_key, qualify, verify_record
from slm_lab.public_pretraining_intake_v2 import candidate
from test_public_pretraining_intake_v1 import fixture


def retained(tmp_path):
    tokenizer, overlap, source, original = fixture()
    source = dict(source, sha256='a'*64)
    original = dict(original, language='GO', path='src/main.go', code='package main\nfunc twice(n int) int { return n * 2 }\n')
    entry, _ = candidate(original, source, tokenizer, overlap)
    entry.update(original_parquet_row_ordinal=7, original_source_file_sha256=source['sha256'])
    writer = CandidateWriter(tmp_path/'candidate', {'code:Go':10**9})
    assert writer.append(entry) == 'whole_original_document_retained_unadmitted'
    writer.close()
    with gzip.open(tmp_path/'candidate/whole-original-documents.jsonl.gz', 'rt') as stream:
        record = json.loads(next(stream))
    proof, reason = qualify(original, source, tokenizer, overlap)
    assert reason is None
    raw = (tmp_path/'candidate'/record['native_file']).read_bytes()
    return original, source, record, proof, raw, tokenizer, overlap


def test_independent_original_GO_source_all_fields_native_mask_and_rights_rebuild(tmp_path):
    original, source, record, proof, raw, _, _ = retained(tmp_path)
    assert budget_key('code', original) == 'code:Go'
    assert verify_record(original, source, 7, record, proof, raw, 0) == len(proof['ids'])-1
    assert record['original_publisher_row']['language'] == 'GO'
    assert np.frombuffer(raw, dtype='<u2').tolist() == proof['ids']


def test_independent_reader_rejects_source_native_role_mask_and_attribution_corruption(tmp_path):
    original, source, record, proof, raw, _, _ = retained(tmp_path)
    for key, value in [('original_parquet_row_ordinal', 8), ('role', 'not-train'), ('target_mask', 'clipped'),
                       ('original_public_URL', 'https://wrong.invalid'), ('native_token_offset', 1)]:
        with pytest.raises(ValueError):
            verify_record(original, source, 7, dict(record, **{key:value}), proof, raw, 0)
    wrong = deepcopy(record);wrong['original_publisher_row']['code'] += '// changed\n'
    with pytest.raises(ValueError):
        verify_record(original, source, 7, wrong, proof, raw, 0)
    with pytest.raises(ValueError, match='native original token'):
        verify_record(original, source, 7, record, proof, raw[:-2]+b'\x04\x00', 0)


def test_independent_qualification_rechecks_license_syntax_and_actual_consumed_held_text(tmp_path):
    original, source, _, _, _, tokenizer, overlap = retained(tmp_path)
    assert qualify(dict(original, license='gpl-3.0'), source, tokenizer, overlap)[1] == 'per_file_license_not_in_declared_allowlist'
    assert qualify(dict(original, language='Python', code='def invalid(\n'+'# text\n'*10), source, tokenizer, overlap)[1] == 'whole_Python3_syntax_qualification'
    class RejectBody:
        def literal(self, text):return ['held actual body']
        def lexical_copies(self, text):return []
    assert qualify(original, source, tokenizer, RejectBody())[1] == 'declared_held_stem_or_lexical_copy_signal'


def test_actual_rebuilder_reconciles_source_ledger_native_tail_and_whole_last_document_quota(tmp_path):
    original, source, record, proof, raw, tokenizer, overlap = retained(tmp_path)
    assert proof['role'] == 'train'
    native = dict(train=io.BytesIO(raw), valid=io.BytesIO())
    rebuild = Rebuilder({'code:Go':1}, tokenizer, overlap, iter([record]), native)
    ledger = dict(family='code', source_file=1, original_row_ordinal=7,
        disposition='whole_original_document_retained_unadmitted', accepted_id=record['id'])
    assert rebuild.observe(original, source, 7, 1, ledger) == ledger['disposition']
    assert rebuild.family_full('code')
    # Once the train quota fills, the producer explicitly skips further rows.
    later = dict(ledger, original_row_ordinal=8, disposition='outside_declared_or_already_satisfied_training_quota', accepted_id=None)
    rebuild.observe(original, source, 8, 1, later)
    result = rebuild.finish()
    assert result['actual_train_targets'] == len(proof['ids'])-1 > 1
    assert rebuild.starts['train'] == [0] and rebuild.observed == 2
    broken = Rebuilder({'code:Go':1}, tokenizer, overlap, iter([record]), dict(train=io.BytesIO(raw+b'\x02\x00'), valid=io.BytesIO()))
    broken.observe(original, source, 7, 1, ledger)
    with pytest.raises(ValueError, match='No unobserved'):
        broken.finish()
