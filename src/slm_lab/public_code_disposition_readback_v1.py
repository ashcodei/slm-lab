"""Rebuild frozen full-shard dispositions without trusting pass booleans."""
import hashlib
import json
import re
import struct

from .held_text_overlap_v1 import normalize
from .public_code_instruction_screen_v1 import parse_whole, literal, UnsupportedExample
from .public_synthetic_code_screen_policy_v1 import published_all_pass, body_fingerprint, supervision_scope
import ast


def row_sha(row):
    return hashlib.sha256(json.dumps(row, sort_keys=True, ensure_ascii=True).encode()).hexdigest()


def verify_observation(parsed, observed):
    fixed = dict(expected_outputs_sent_to_worker=False, whole_original_tests=len(parsed['cases']),
        published_tests_are_synthetic=True, independent_semantic_correctness_not_certified=True,
        program_sha256=hashlib.sha256(parsed['program'].encode()).hexdigest(),
        code_bytes_unchanged_after_fence_extraction=True,
        OS_confinement='Landlock and seccomp before interpreter startup')
    if any(observed.get(k) != v or type(observed.get(k)) is not type(v) for k, v in fixed.items()):
        raise ValueError('Complete original execution identity and confined-test scope differ')
    if any(type(observed.get(k)) is not bool for k in ['all_cases_observed', 'all_published_tests_passed']):
        raise ValueError('Actual observation booleans required')
    execution = observed['execution']
    if not isinstance(execution.get('status'), str) or re.fullmatch(r'[0-9a-f]{64}', execution['stdout_sha256']) is None:
        raise ValueError('A bounded original execution receipt is required')
    status = observed['status']
    if observed['all_cases_observed']:
        if status not in {'all_published_tests_pass', 'published_test_mismatch'} or execution['status'] != 'completed':
            raise ValueError('Complete returned values require completed original execution')
        values = observed['actual_value_literals']
        if not isinstance(values, list) or len(values) != len(parsed['cases']):
            raise ValueError('Every original test needs its actual returned literal')
        actual = [literal(ast.parse(value, mode='eval').body) for value in values]
        expected = [literal(ast.parse(case['expected_literal'], mode='eval').body) for case in parsed['cases']]
        flags = [a == b for a, b in zip(actual, expected)]
        if (observed['test_passes'] != flags or any(type(v) is not bool for v in observed['test_passes'])
                or observed['passed_tests'] != sum(flags) or type(observed['passed_tests']) is not int
                or observed['all_published_tests_passed'] != all(flags)
                or (status == 'all_published_tests_pass') != all(flags)):
            raise ValueError('Published-test pass decisions do not match the raw actual values')
    elif (observed['all_published_tests_passed']
            or status not in {'runtime_resource_or_confinement_abstention', 'unsupported_or_incomplete_raw_output_abstention'}
            or (status == 'runtime_resource_or_confinement_abstention') == (execution['status'] == 'completed')):
        raise ValueError('A resource/output abstention must not certify returned values or correctness')


def rebuild(row, position, overlap, tokenizer, veto, bad_ast, prior, pilot=None):
    rebuilt = dict(original_shard_position=position, source_id=row.get('id'), original_entry_sha256=row_sha(row),
        source_domain=row.get('domain'), scope=supervision_scope(), screen_pass_candidate=False)
    native = None
    parsed = None
    if not all(isinstance(row.get(k), str) and bool(row[k]) for k in ['id', 'input', 'output']):
        rebuilt['status'] = 'malformed_whole_source_abstention'
    elif not published_all_pass(row):
        rebuilt['status'] = 'provider_not_complete_all_pass_exclusion'
    else:
        rebuilt.update(protected_overlap=overlap,
            input_only_group_sha256=hashlib.sha256(normalize(row['input']).encode()).hexdigest())
        if overlap:
            rebuilt['status'] = 'protected_input_overlap_exclusion'
        else:
            prompt = 'User: ' + row['input'] + '\nAssistant:'
            head = tokenizer.encode(prompt, add_special_tokens=False).ids
            response = tokenizer.encode(row['output'], add_special_tokens=False).ids
            ids = [1] + head + response + [2]
            rebuilt.update(complete_context_tokens=len(ids), response_EOS_targets=len(response) + 1)
            if len(ids) > 4096 or any(not 4 <= value < 32768 for value in ids[1:-1]):
                rebuilt['status'] = 'whole_native_length_or_special_token_abstention'
            else:
                if (tokenizer.decode(head, skip_special_tokens=False) != prompt
                        or tokenizer.decode(response, skip_special_tokens=False) != row['output']):
                    raise ValueError('Complete original prompt and response must round-trip without normalization')
                mask = [False] * (1 + len(head)) + [True] * (len(response) + 1)
                if len(mask) != len(ids) or sum(mask) != len(response) + 1 or not mask[-1] or ids[-1] != 2:
                    raise ValueError('Whole original response and genuine EOS must form the full contiguous target mask')
                native = dict(native_input_u16le_sha256=hashlib.sha256(struct.pack('<' + 'H' * len(ids), *ids)).hexdigest(),
                    native_target_mask_u8_sha256=hashlib.sha256(bytes(mask)).hexdigest(),
                    complete_context_tokens=len(ids), response_EOS_targets=sum(mask), target_start=1 + len(head))
                try:
                    parsed = parse_whole(row)
                except UnsupportedExample as error:
                    rebuilt.update(status='unsupported_whole_test_set_abstention', reason=str(error))
                else:
                    fingerprint = body_fingerprint(parsed['program'])
                    rebuilt.update(complete_original_tests=len(parsed['cases']), program_AST_without_docstrings_sha256=fingerprint)
                    if row_sha(row) == veto['original_entry_sha256'] or fingerprint == bad_ast:
                        rebuilt.update(status='known_independent_semantic_veto', veto_evidence=veto['evidence_path'])
                    else:
                        observed = prior['actual_whole_execution']
                        verify_observation(parsed, observed)
                        reused = prior['execution_reused_from_exact_fresh_pilot_readback']
                        if type(reused) is not bool:
                            raise ValueError('Pinned pilot execution reuse must be explicit')
                        if reused and (pilot is None or pilot.get('fresh_whole_program_execution') != observed):
                            raise ValueError('An exact source-bound full fresh pilot observation is required for reuse')
                        if reused and not observed['all_published_tests_passed']:
                            raise ValueError('Only fully passing exact pilot observations may have been reused')
                        rebuilt.update(actual_whole_execution=observed,
                            execution_reused_from_exact_fresh_pilot_readback=reused,
                            status='weak_SFT_screen_pass_candidate' if observed['all_published_tests_passed'] else observed['status'],
                            screen_pass_candidate=observed['all_published_tests_passed'])
    if rebuilt != prior:
        raise ValueError('A frozen original whole-row disposition does not reconstruct exactly at position ' + str(position))
    return rebuilt, native, parsed
