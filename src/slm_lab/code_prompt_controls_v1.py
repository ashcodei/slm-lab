"""Counterfactual prompts for known public training algorithms, not fresh tasks."""
from copy import deepcopy
import hashlib

from .code_quality import instruction_code
from .verified_code import renamed

MARKER = '\n\nPublished behavior examples:\n'
ENDING = 'Return the complete function and any necessary helpers.'


def without_examples(row):
    if row['prompt'].count(MARKER) != 1 or not row['prompt'].endswith(ENDING):
        raise ValueError('Expected the exact admitted machine-added example block')
    result = deepcopy(row)
    result['prompt'] = row['prompt'].split(MARKER)[0] + '\n\n' + ENDING
    result['prompt_condition'] = 'original_signature_description_without_published_examples'
    return result


def new_function_name(row):
    result = without_examples(row)
    old = row['function']; new = 'task_' + hashlib.sha256(row['id'].encode()).hexdigest()[:10]
    signature = 'def ' + old + '('
    if result['prompt'].count(signature) != 1:
        raise ValueError('Original function signature is not unique in prompt')
    result['prompt'] = result['prompt'].replace(signature, 'def ' + new + '(', 1)
    result['response'] = '```python\n' + renamed(instruction_code(row['response']), old, new) + '\n```'
    result['function'] = new
    result['original_function'] = old
    result['prompt_condition'] = 'unseen_function_identifier_description_without_published_examples'
    return result
