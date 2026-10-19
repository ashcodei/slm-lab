"""Rebuild full original source qualification independently of its intake/writer."""
import ast
from collections import Counter
import hashlib
import json
from urllib.parse import quote, urlsplit

import numpy as np

LANGUAGES = {'Python', 'C', 'C++', 'Java', 'JavaScript', 'TypeScript', 'Go', 'Rust'}
LICENSES = {'mit', 'apache-2.0', 'bsd-2-clause', 'bsd-3-clause', 'isc', 'cc0-1.0', 'unlicense', 'zlib', '0bsd'}


def budget_key(family, row):
    language = row.get('language')
    language = 'Go' if language == 'GO' else language
    return 'code:'+language if family == 'code' and isinstance(language, str) else (
        'code:unsupported_publisher_language' if family == 'code' else family)


def qualify(original, source, tokenizer, overlap):
    family = source['family']
    text = original.get('code' if family == 'code' else 'text')
    if not isinstance(text, str) or not text.strip() or '\0' in text:
        return None, 'empty_nontext_or_NUL'
    raw = text.encode('utf-8')
    if not 32 <= len(raw) <= 1024**2:
        return None, 'whole_document_byte_bound'
    language = 'Go' if original.get('language') == 'GO' else original.get('language')
    if family == 'code':
        license = original.get('license')
        if not isinstance(license, str) or license.casefold() not in LICENSES:
            return None, 'per_file_license_not_in_declared_allowlist'
        if language not in LANGUAGES:
            return None, 'outside_declared_programming_languages'
        repo, path = original.get('repo_name'), original.get('path')
        if not isinstance(repo, str) or repo.count('/') != 1 or not isinstance(path, str) or not path:
            return None, 'missing_original_repository_or_path'
        if any(marker in (repo+'/'+path).casefold() for marker in ['humaneval', 'human_eval', 'mbpp', 'gsm8k', 'mmlu']):
            return None, 'recognizable_held_benchmark_path'
        if language == 'Python':
            try:
                ast.parse(text)
            except (SyntaxError, ValueError, MemoryError, RecursionError):
                return None, 'whole_Python3_syntax_qualification'
    else:
        url = original.get('url')
        if not isinstance(url, str) or urlsplit(url).scheme not in {'http', 'https'} or not urlsplit(url).netloc:
            return None, 'missing_original_public_page_URL'
    values = tokenizer.encode(text, add_special_tokens=False).ids
    if not values or any(type(v) is not int or not 4 <= v < 32768 for v in values):
        return None, 'reserved_boundary_inside_original_text'
    if tokenizer.decode(values, skip_special_tokens=False) != text:
        return None, 'own_BPE_not_lossless_for_whole_original_text'
    if overlap.literal(text) or overlap.lexical_copies(text):
        return None, 'declared_held_stem_or_lexical_copy_signal'
    sha = hashlib.sha256(raw).hexdigest()
    return dict(sha256=sha, ids=[1, *values, 2], role='valid' if int(sha[-8:], 16)%100 == 0 else 'train',
        language=language if family == 'code' else 'English', key=budget_key(family, original)), None


def verify_record(original, source, ordinal, record, proof, native, expected_offset):
    """Check freshly re-fetched original fields and exact consumed native bytes."""
    ids = proof['ids']
    role = proof['role']
    family = source['family']
    field = 'code' if family == 'code' else 'text'
    if (record['original_publisher_row'] != json.loads(json.dumps(original, default=str))
            or record['original_text_publisher_field'] != field
            or record['original_parquet_row_ordinal'] != ordinal
            or record['source_repository'] != source['repo'] or record['source_revision'] != source['revision']
            or record['source_parquet'] != source['path'] or record['original_source_file_sha256'] != source['sha256']
            or record['family'] != family or record['language'] != proof['language']
            or record['original_UTF8_sha256'] != proof['sha256'] or record['id'] != 'public-source:'+proof['sha256']
            or record['role'] != role or record['native_file'] != role+'.tokens.u16'
            or record['native_token_offset'] != expected_offset or record['native_token_length'] != len(ids)
            or record['complete_causal_targets'] != len(ids)-1 or record['complete_context_tokens'] != len(ids)
            or record['target_mask'] != 'Every original token after native BOS, including EOS; BOS is not a target'
            or record['optimization_admitted'] is not False or record['human_preferences'] is not False
            or record['execution_reward'] is not False
            or record['whole_document_preserved_without_clipping'] is not True
            or record['own_BPE_decodes_entire_original_UTF8_text_exactly'] is not True
            or record['source_quality_filter_is_not_human_or_execution_correctness_label'] is not True
            or record['source_semantic_and_all_history_independence_not_certified'] is not True):
        raise ValueError('Whole original publisher fields, fixed role, source identity and native mask differ')
    raw = np.asarray(ids, dtype='<u2').tobytes()
    if native != raw or hashlib.sha256(native).hexdigest() != record['native_little_u16_sha256']:
        raise ValueError('Every actually consumed native original token and EOS byte must rebuild')
    if family == 'code':
        if (record['original_publisher_language'] != original['language']
                or record['publisher_language_normalization'] != dict(original=original['language'], canonical=proof['language'],
                    original_source_fields_and_program_bytes_unchanged=True)):
            raise ValueError('Original publisher GO spelling must be retained separately from canonical Go')
        expected = 'https://github.com/'+original['repo_name']+'/blob/HEAD/'+quote(original['path'].lstrip('/'), safe='/')
        if (record['original_public_URL'] != expected or record['rights'] != dict(
                publisher_per_file_license=original['license'], repository=original['repo_name'], path=original['path'],
                repository_HEAD_URL_is_attribution_not_verified_current_file_bytes=True,
                dataset_catalog_license_does_not_replace_per_file_license=True, license_text_not_independently_retrieved=True)):
            raise ValueError('Original per-file rights and repository/path attribution differ')
    elif (record['original_public_URL'] != original['url'] or record['rights'] != dict(
            dataset_license=source['dataset_license'], underlying_page_rights_not_relicensed=True, CommonCrawl_terms_apply=True)):
        raise ValueError('Original page URL and underlying-rights qualifications differ')
    return len(ids)-1


class Rebuilder:
    """Replay all source dispositions, quotas and actual accepted native bytes."""
    def __init__(self, quotas, tokenizer, overlap, documents, native):
        self.quotas = dict(quotas)
        self.tokenizer, self.overlap = tokenizer, overlap
        self.documents, self.native = documents, native
        self.train, self.valid, self.examples = Counter(), Counter(), Counter()
        self.offsets = dict(train=0, valid=0)
        self.starts = dict(train=[], valid=[])
        self.seen = set()
        self.dispositions = Counter()
        self.observed = 0

    def full(self, key):
        return key not in self.quotas or self.train[key] >= self.quotas[key]

    def family_full(self, family):
        keys = [k for k in self.quotas if k == family or k.startswith(family+':')]
        return bool(keys) and all(self.full(k) for k in keys)

    def observe(self, original, source, ordinal, source_number, ledger):
        key = budget_key(source['family'], original)
        proof = None
        if self.full(key):
            reason = 'outside_declared_or_already_satisfied_training_quota'
        else:
            proof, reason = qualify(original, source, self.tokenizer, self.overlap)
            if proof is not None:
                reason = ('exact_original_body_duplicate_within_new_candidate' if proof['sha256'] in self.seen
                          else 'whole_original_document_retained_unadmitted')
        expected_id = None
        if reason == 'whole_original_document_retained_unadmitted':
            record = next(self.documents, None)
            if record is None:
                raise ValueError('Every accepted original source needs its complete retained record')
            role = proof['role']
            offset = self.offsets[role]
            native = self.native[role].read(len(proof['ids'])*2)
            targets = verify_record(original, source, ordinal, record, proof, native, offset)
            self.starts[role].append(offset)
            self.offsets[role] += len(proof['ids'])
            self.seen.add(proof['sha256'])
            (self.train if role == 'train' else self.valid)[key] += targets
            self.examples[(key, role)] += 1
            expected_id = 'public-source:'+proof['sha256']
        if ledger != dict(family=source['family'], source_file=source_number, original_row_ordinal=ordinal,
                          disposition=reason, accepted_id=expected_id):
            raise ValueError('Every original source-prefix disposition and exact accepted identity must rebuild')
        self.dispositions[reason] += 1
        self.observed += 1
        return reason

    def summary(self):
        return dict(train_target_tokens_by_quota={k:self.train[k] for k in self.quotas},
            valid_target_tokens_by_quota={k:self.valid[k] for k in self.quotas},
            actual_train_targets=sum(self.train.values()), actual_valid_targets=sum(self.valid.values()),
            original_documents={k:{r:self.examples[(k,r)] for r in ['train','valid']} for k in self.quotas},
            native_context_token_counts=dict(self.offsets),
            all_training_quotas_satisfied=all(self.full(k) for k in self.quotas),
            whole_final_document_may_exceed_quota_no_target_clipping=True, optimization_admitted=False)

    def finish(self):
        if next(self.documents, None) is not None or any(stream.read(1) for stream in self.native.values()):
            raise ValueError('No unobserved original document or native token bytes may remain')
        if not all(self.full(k) for k in self.quotas):
            raise ValueError('All original declared training quotas must finish')
        return self.summary()
