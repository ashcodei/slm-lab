"""Reconstruct the exact device-observer protocol without inventing fields."""
import hashlib
import struct

from .code_quality import instruction_code
from .decoded_native_code_reexecution_v1 import reconstruct
from .native_code_action_readback_v1 import prompt_for

FIELDS = {'case_position', 'original_input_token_ids', 'actual_generated_token_ids',
    'original_input_ids_sha256', 'input_pixels_sha256', 'requested_native_action_budget',
    'actual_native_action_count', 'actual_stopped_on_EOS', 'inference_device_type',
    'inference_device_index', 'inference_precision', 'greedy_native_output_tensor_returned_unchanged',
    'generated_program_not_repaired'}


def validate(event, tokenizer, *, prompt_ids, budget, case_position):
    if (tokenizer.get_vocab_size() != 32768 or tokenizer.truncation is not None or tokenizer.padding is not None
            or [tokenizer.token_to_id(t) for t in ['<pad>', '<bos>', '<eos>']] != [0, 1, 2]):
        raise ValueError('The original complete own tokenizer and boundary convention are required')
    ids = event.get('actual_generated_token_ids')
    if (set(event) != FIELDS or not isinstance(prompt_ids, list) or not prompt_ids or prompt_ids[0] != 1
            or any(type(x) is not int or not 0 <= x < 32768 for x in prompt_ids)
            or budget not in [512, 1024] or len(prompt_ids) + budget > 4096
            or type(event['case_position']) is not int or event['case_position'] != case_position
            or event['original_input_token_ids'] != prompt_ids
            or event['original_input_ids_sha256'] != hashlib.sha256(struct.pack('<' + 'H' * len(prompt_ids), *prompt_ids)).hexdigest()
            or event['input_pixels_sha256'] is not None
            or type(event['requested_native_action_budget']) is not int or event['requested_native_action_budget'] != budget
            or not isinstance(ids, list) or not 1 <= len(ids) <= budget
            or any(type(x) is not int or not 0 <= x < 32768 for x in ids)
            or type(event['actual_native_action_count']) is not int or event['actual_native_action_count'] != len(ids)
            or 2 in ids[:-1] or type(event['actual_stopped_on_EOS']) is not bool
            or event['actual_stopped_on_EOS'] != (ids[-1] == 2) or len(ids) < budget and ids[-1] != 2
            or event['inference_device_type'] != 'cuda' or event['inference_device_index'] != 0
            or event['inference_precision'] != 'bf16'
            or event['greedy_native_output_tensor_returned_unchanged'] is not True
            or event['generated_program_not_repaired'] is not True):
        raise ValueError('Every actual device action, prompt hash, frame, native budget and first-EOS stop must reconstruct')
    return dict(actual_complete_prompt_sha256=event['original_input_ids_sha256'],
        actual_action_ids_sha256=hashlib.sha256(struct.pack('<' + 'H' * len(ids), *ids)).hexdigest(),
        actual_native_action_count=len(ids), actual_stopped_on_EOS=ids[-1] == 2,
        original_raw_decoded_response=tokenizer.decode(ids), source_role='evaluation', optimization_forbidden=True,
        original_device_observer_does_not_record_action_probabilities=True,
        no_action_probabilities_invented_or_certified=True, no_new_generation_or_program_repair=True)


def local(case, frame, event, tokenizer, *, case_position):
    prompt = [1] + tokenizer.encode(prompt_for(case, frame)).ids
    proof = validate(event, tokenizer, prompt_ids=prompt, budget=512, case_position=case_position)
    if proof.pop('original_raw_decoded_response') != case['raw_response']:
        raise ValueError('Actual original actions must decode to the unchanged complete raw code response')
    _, program = reconstruct(case, frame)
    return dict(proof, actual_program_sha256=hashlib.sha256(program.encode()).hexdigest())


def source(row, cohort, event, tokenizer, *, case_position):
    proof = validate(event, tokenizer, prompt_ids=cohort['original_prompt_ids'], budget=1024, case_position=case_position)
    raw = proof.pop('original_raw_decoded_response')
    if (cohort['source_role'] != 'validation' or cohort['optimization_forbidden'] is not True
            or row['source_role'] != 'validation' or row['original_source_id'] != cohort['original_source_id']
            or row['raw_response'] != raw or row['actual_event'] != event
            or row['program'] != instruction_code(raw)):
        raise ValueError('Every unchanged source-validation program must retain its original held role and public prompt')
    return dict(proof, actual_program_sha256=hashlib.sha256(row['program'].encode()).hexdigest(),
        source_pack_public_examples_and_answers_visible_in_prompt=True,
        source_pack_not_certified_historically_independent=True)
