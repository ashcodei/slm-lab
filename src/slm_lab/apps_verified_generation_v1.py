"""Unchanged original train programs with complete finite source-case evidence.

This is supervised generation data. Execution evidence is neither a proof of
general correctness/efficiency nor genuine human preference feedback.
"""
import hashlib
import json

import numpy as np

from .apps_published_cases_v2 import contract, compare


def generation_example(source, question, starter, program, original_tests, observed, tokenizer):
    # Role rejection must precede touching code, test cases, or the tokenizer.
    if source.get('split') != 'train' or source.get('publisher_split') != 'train':
        raise ValueError('Only admitted original publisher-train programs can enter generation training')
    cases = contract(original_tests)
    if (observed.get('id') != source['id'] or observed.get('admitted_role') != 'train'
            or observed.get('publisher_role') != 'train'
            or observed.get('status') != 'all_finite_published_cases_pass'
            or observed.get('all_cases_observed') is not True
            or observed.get('all_finite_published_cases_pass') is not True
            or observed.get('syntax_valid') is not True
            or observed.get('mode') != cases['mode']
            or observed.get('total_cases') != cases['total_cases']
            or observed.get('passed_cases') != cases['total_cases']
            or observed.get('original_tests_sha256') != cases['original_tests_sha256']
            or source['original_tests_sha256'] != cases['original_tests_sha256']):
        raise ValueError('Exact complete passing original training-case observation required')
    raw = observed.get('raw_results', {}).get('results', [])
    if (len(raw) != cases['total_cases']
            or any(not compare(result.get('value'), expected, cases['mode'])
                   for result, expected in zip(raw, cases['outputs']))
            or observed.get('case_passes') != [True] * cases['total_cases']):
        raise ValueError('Every retained raw result must pass its complete original source case')
    program_sha = hashlib.sha256(program.encode()).hexdigest()
    if (program_sha != source['original_source_body_sha256']
            or program_sha != observed.get('original_program_sha256')
            or hashlib.sha256(question.encode()).hexdigest() != source['original_question_sha256']
            or source['response'] != program):
        raise ValueError('Unchanged complete original human program and question required')
    interface = ('Use the original required function or class interface.' if starter or cases['function']
                 else 'Read standard input and write standard output as specified.')
    prompt = ('Write a Python solution to the following problem. ' + interface +
              ' Return only the complete code.\n\n' + question)
    if starter:
        prompt += '\n\nOriginal starter code:\n' + starter
    if source['prompt'] != prompt:
        raise ValueError('Exact original native generation prompt required')
    prefix = [1] + tokenizer.encode('User: ' + prompt + '\nAssistant:').ids
    body = tokenizer.encode(program).ids + [2]
    ids = prefix + body
    if (len(ids) > 4096 or any(marker in prompt or marker in program for marker in ['<pad>', '<bos>', '<eos>'])
            or hashlib.sha256(np.asarray(ids, dtype='<u2').tobytes()).hexdigest() != source['complete_input_ids_sha256']
            or len(body) != source['target_tokens']):
        raise ValueError('Every original complete token, target mask and context budget must match')
    row = dict(source, id=source['id'] + ':verified-generation-v1', original_source_program_id=source['id'],
        task='original_problem_to_complete_original_human_code', execution_verified=True,
        verification_scope='All finite original published training cases under the declared local stdlib-prelude runtime',
        annotation_origin='unchanged_original_human_solution_with_complete_finite_source_case_evidence',
        complete_original_cases=cases['total_cases'], original_target_program_bytes_modified=False,
        held_evaluation_cases_in_prompt=False, machine_execution_feedback=True,
        human_preferences=False, genuine_RLHF_labels=False, external_model_teacher_used=False,
        model_generated_examples=False, algorithmic_correctness_or_efficiency_proven=False,
        dataset_admitted_to_training=False)
    return row, ids, [-100] * len(prefix) + body
