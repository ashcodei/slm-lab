import copy
import json
from pathlib import Path

import pytest
from tokenizers import Tokenizer

from slm_lab.device_native_code_action_readback_v1 import local, source

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / 'reports/full-function-repair-comparison-GPU-v2'


def lines(path):
    with path.open('rb') as stream:
        return [json.loads(line) for line in stream]


def fixtures():
    token = Tokenizer.from_file(str(ROOT / 'checkpoints/full-function-repair-comparison-GPU-v2/tokenizer.json'))
    report = json.loads((STUDY / 'baseline/code-completion-shard-0.raw.json').read_text())
    event = lines(STUDY / 'baseline/code-completion-actions-0.jsonl')[0]
    return token, report['results'][0], event


def test_all96_real_closed_local_action_sequences_and_all30_source_deliveries_reconstruct():
    token, _, _ = fixtures()
    cohort = json.loads((ROOT / 'reports/source-function-validation-runtime-CPU-v1/all15-fixed-native-validation-cases.json').read_text())
    local_count = source_count = 0
    for profile in ['baseline', 'plain_repeat']:
        for rank in range(4):
            for frame in ['completion', 'instruction']:
                report = json.loads((STUDY / profile / f'code-{frame}-shard-{rank}.raw.json').read_text())
                events = lines(STUDY / profile / f'code-{frame}-actions-{rank}.jsonl')
                assert len(events) == len(report['results']) == 6
                for position, (row, event) in enumerate(zip(report['results'], events)):
                    proof = local(row, frame, event, token, case_position=position)
                    assert proof['no_action_probabilities_invented_or_certified']
                    local_count += 1
            report = json.loads((STUDY / profile / f'source-validation-shard-{rank}.json').read_text())
            events = lines(STUDY / profile / f'source-validation-actions-{rank}.jsonl')
            assert len(report['results']) == len(events)
            for i, (row, event) in enumerate(zip(report['results'], events)):
                source(row, cohort[row['position']], event, token, case_position=i)
                source_count += 1
    assert (local_count, source_count) == (96, 30)


@pytest.mark.parametrize('case', ['prompt', 'hash', 'action', 'EOS_flag', 'case_position', 'budget',
    'probabilities_invented', 'device', 'precision', 'hidden_image', 'repair', 'raw_response'])
def test_tampering_cannot_be_called_an_original_native_delivery(case):
    token, row, event = fixtures()
    row, event = copy.deepcopy(row), copy.deepcopy(event)
    if case == 'prompt':
        event['original_input_token_ids'][2] += 1
    elif case == 'hash':
        event['original_input_ids_sha256'] = 'a' * 64
    elif case == 'action':
        event['actual_generated_token_ids'][0] = 2
    elif case == 'EOS_flag':
        event['actual_stopped_on_EOS'] = not event['actual_stopped_on_EOS']
    elif case == 'case_position':
        event['case_position'] = 1
    elif case == 'budget':
        event['requested_native_action_budget'] = 256
    elif case == 'probabilities_invented':
        event['selected_action_log_probabilities_not_retained'] = True
    elif case == 'device':
        event['inference_device_type'] = 'cpu'
    elif case == 'precision':
        event['inference_precision'] = 'fp32'
    elif case == 'hidden_image':
        event['input_pixels_sha256'] = 'b' * 64
    elif case == 'repair':
        event['generated_program_not_repaired'] = False
    else:
        row['raw_response'] += '\n'
    with pytest.raises(ValueError):
        local(row, 'completion', event, token, case_position=0)
