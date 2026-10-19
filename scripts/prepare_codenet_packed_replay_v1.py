"""Admit and index all original older/systems packed code using CPU only."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil

from tokenizers import Tokenizer

from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.codenet_packed_replay_v1 import FAMILY, PackedReplayCorpus, prepare_packed_replay
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.evaluation_provenance import capture
from slm_lab.systems_variety_training_admission import admit


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'data/codenet-packed-pretraining-replay-v1'
REPORTS = ROOT / 'reports/codenet-packed-pretraining-replay-v1'
CAP = 32 * 1024**2
RESERVE = 26 * 1024**3


def status(message, **fields):
    atomic_json(REPORTS / 'status.json', dict(status=message,
        updated_at=datetime.now(timezone.utc).isoformat(), goal_achieved=False, **fields))


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU-only original packed replay preparation required')
    if OUTPUT.exists() or REPORTS.exists():
        raise ValueError('Retain existing original preparation; no implicit restart')
    if shutil.disk_usage(ROOT).free < RESERVE + 2 * CAP:
        raise ValueError('Fund32MiB source indices/reports each above unchanged26GiB reserve')
    token = ROOT / 'data/tokenizer-corpus/tokenizer.json'
    paths = [Path(__file__), token, ROOT / 'configs/model.json',
        ROOT / 'tests/test_codenet_packed_replay_v1.py', ROOT / 'reports/codenet-packed-replay-v1-focused-tests.json']
    paths += [ROOT / 'src/slm_lab' / (name + '.py') for name in ['codenet_packed_replay_v1',
        'codenet_checkpoint_v1', 'systems_variety_training_admission', 'systems_variety_admission',
        'systems_variety', 'wikipedia_corpus']]
    paths += [ROOT / 'data/systems-variety-code-continuation-v1' / name
        for name in ['manifest.json', 'new-systems.jsonl', 'packed/corpus.json', 'new-packed/corpus.json']]
    paths.append(ROOT / 'data/broad-code-continuation-v1/packed/corpus.json')
    bindings = {str(path): file_hash(path) for path in paths}
    bindings.update({str(ROOT / path): digest for path, digest in capture()['files_sha256'].items()})
    guard = FileBindings(bindings)
    REPORTS.mkdir()
    status('reconstructing original systems archives/files/licenses and every original new packed token')
    try:
        packed_root, receipt = admit(ROOT)
        guard.check(full=True)
        status('original systems and earlier packed bytes admitted; indexing complete causal windows including tails',
            new_train_tokens=receipt['new_train_tokens'], all_train_tokens=receipt['combined_train_tokens'])
        tokenizer = Tokenizer.from_file(str(token))
        config = json.loads((ROOT / 'configs/model.json').read_text())
        if tokenizer.get_vocab_size() != config['vocab_size']:
            raise ValueError('Actual own tokenizer vocabulary must match original model')
        manifest = prepare_packed_replay(OUTPUT, packed_root, receipt, token,
            config['max_text_length'], config['vocab_size'])
        reader = PackedReplayCorpus(OUTPUT)
        size = sum(path.stat().st_size for path in OUTPUT.iterdir() if path.is_file())
        if size > CAP or reader.size(FAMILY) != manifest['splits']['train']['windows']:
            raise ValueError('Complete bounded source index/population differs')
        guard.check(full=True)
        atomic_json(REPORTS / 'results.json', dict(completed_at=datetime.now(timezone.utc).isoformat(),
            source_files_sha256=bindings, manifest_sha256=file_hash(OUTPUT / 'manifest.json'),
            splits=manifest['splits'], stored_artifact_bytes=size, original_source_bytes_copied=0,
            original_new_systems_files_reconstructed=receipt['source_receipt']['original_files_reconstructed'],
            all_original_new_systems_text_and_EOS_reencoded=receipt['every_complete_original_text_and_EOS_reencoded_and_matched'],
            original_prior_code_train_tokens_bound=receipt['all_prior_code_train_tokens_rehearsed'],
            source_file_group_overlap=receipt['source_file_group_overlap'], independent_index_audit_pending=True,
            original_running_curriculum_changed=False, neural_training_started=False,
            registered_in_production_runner=False, goal_achieved=False))
        status('all original packed replay windows indexed; independent token/mask audit pending',
            splits=manifest['splits'], stored_artifact_bytes=size)
        print(json.dumps(dict(splits=manifest['splits'], stored_artifact_bytes=size)), flush=True)
    except Exception as error:
        status('original packed replay preparation stopped; retain evidence and inspect', error=str(error))
        raise


if __name__ == '__main__':
    main()
