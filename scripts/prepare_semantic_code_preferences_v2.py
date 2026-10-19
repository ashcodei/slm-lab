"""Wait for the verified live source diagnostic, then prepare execution contrasts."""
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import torch
from tokenizers import Tokenizer

from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.evaluation_provenance import capture
from slm_lab.model import ModelConfig
from slm_lab.semantic_code_preferences_v2 import prepare
from slm_lab.training import Encoder

ROOT = Path(__file__).resolve().parents[1]


def main():
    reports = ROOT / 'reports/semantic-code-preferences-v2'
    if reports.exists():
        raise ValueError('Inspect existing preparation; no automatic replacement')
    reports.mkdir()
    launch = json.loads((ROOT / 'reports/exposed-code-learning-after-instruction-launch-v1.json').read_text())
    files = list(dict.fromkeys([ROOT / name for name in capture()['files_sha256']] + [
        Path(__file__), ROOT / 'src/slm_lab/semantic_code_preferences_v2.py',
        ROOT / 'src/slm_lab/instruction_generalization_trial_v3.py',
        ROOT / 'src/slm_lab/verified_code_diagnostics_v2.py',
        ROOT / 'scripts/assess_exposed_code_learning.py']))
    runtime = capture(files)
    if (file_hash(ROOT / 'scripts/assess_exposed_code_learning.py') != launch['source_sha256']
            or file_hash(ROOT / 'src/slm_lab/verified_code_diagnostics_v2.py') != launch['diagnostic_module_sha256']):
        raise ValueError('Original diagnostic actor source differs from its launch receipt')

    def guard():
        if capture(files) != runtime:
            raise ValueError('Bound preparation/generation implementation changed')

    def status(message, **fields):
        atomic_json(reports / 'status.json', {'status': message,
            'updated_at': datetime.now(timezone.utc).isoformat(),
            'trained': False, 'goal_achieved': False, **fields})

    def original_actor_live():
        process = Path('/proc') / str(launch['pid'])
        try:
            stat = (process / 'stat').read_text().split(') ', 1)[1].split()
            argv = (process / 'cmdline').read_bytes().split(b'\0')[:-1]
        except FileNotFoundError:
            return False
        return (stat[0] != 'Z' and stat[19] == launch['start_ticks']
                and argv == [part.encode() for part in launch['argv']])

    try:
        diagnostic = ROOT / 'reports/exposed-code-learning-after-instruction-v1'
        complete = 'exposed-source learning comparison completed; broader capability unverified'
        while True:
            guard()
            current = json.loads((diagnostic / 'status.json').read_text())
            if current['status'] == complete:
                break
            if not original_actor_live():
                raise ValueError('Original diagnostic actor is terminal or missing without completed paired evidence')
            status('waiting for verified live paired source diagnostic',
                   preceding_pid=launch['pid'], preceding_start_ticks=launch['start_ticks'],
                   preceding_status=current, original_actor_live_verified=True)
            time.sleep(15)
        trial = ROOT / 'reports/instruction-generalization-probe-v3'
        plan = json.loads((trial / 'plan.json').read_text())
        trained = json.loads((trial / 'training.json').read_text())
        comparison = json.loads((diagnostic / 'comparison.json').read_text())
        checkpoint = Path(trained['candidate_checkpoint'])
        if (comparison['candidate_checkpoint_sha256'] != trained['candidate_checkpoint_sha256']
                or file_hash(checkpoint) != trained['candidate_checkpoint_sha256']
                or trained['completed_steps'] != 1500):
            raise ValueError('Completed own instruction policy differs')
        source = Path(plan['input_manifest'])
        if file_hash(source) != plan['input_manifest_sha256']:
            raise ValueError('Original admitted instruction pools differ')
        metadata = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
        tokenizer = checkpoint.parent / 'tokenizer.json'
        if (file_hash(tokenizer) != plan['tokenizer_sha256']
                or metadata['tokenizer_sha256'] != plan['tokenizer_sha256']
                or metadata['steps'] != 1500 or metadata['planned_steps'] != 1500
                or metadata['data_sha256'] != plan['input_manifest_sha256']):
            raise ValueError('Actual checkpoint/source/tokenizer learning identities differ')
        encoder = Encoder(Tokenizer.from_file(str(tokenizer)), ModelConfig(**metadata['config']), source.parent, 448)
        del metadata
        output = ROOT / 'data/semantic-code-preferences-v2'
        status('reexecuting own runnable errors and original public positives')
        receipt = prepare(source.parent / 'verified_python.jsonl', source.parent / 'exposures.jsonl',
                          diagnostic / 'after.json', checkpoint, output, encoder)
        guard()
        atomic_json(reports / 'preparation.json', {'data_manifest': str(output / 'manifest.json'),
            'data_manifest_sha256': file_hash(output / 'manifest.json'),
            'train_examples': receipt['train_examples'], 'excluded': receipt['excluded'],
            'source_runtime': runtime, 'tokenizer_sha256': plan['tokenizer_sha256'],
            'preceding_comparison_sha256': file_hash(diagnostic / 'comparison.json'),
            'trained': False, 'human_feedback': False, 'goal_achieved': False})
        status('own-code wrong-result preference preparation completed; learning untested',
               train_examples=receipt['train_examples'], data_ready=receipt['data_ready'])
    except Exception as error:
        status('semantic preference preparation stopped; requires inspection', error=str(error))
        raise


if __name__ == '__main__':
    main()
