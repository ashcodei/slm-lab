"""Retain a current own foundation snapshot and assess audited visual questions."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import time

import torch
from slm_lab.clevr_evaluation import evaluate
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.evaluation_provenance import capture
from slm_lab.masked_visual_pretraining import process_identity

ROOT = Path(__file__).resolve().parents[1]


def prepare(source, directory):
    if directory.exists(): raise ValueError('Retain existing baseline; never overwrite or silently restart')
    if shutil.disk_usage(ROOT).free < 35 * 1024**3: raise ValueError('Need35 GiB free before retaining this model')
    metadata = torch.load(source, map_location='cpu', weights_only=True, mmap=True)
    if (metadata['stage'] != 'pretrain' or metadata['steps'] != 3000 or metadata['planned_steps'] != 17746
            or metadata['world_size'] != 4 or metadata['trainable'] != 'language' or metadata['truncated_sequences'] != 0):
        raise ValueError('Source is not the expected completed3000-update own language snapshot')
    metrics = metadata['metrics']
    if ([item['step'] for item in metrics] != list(range(1, 3001))
            or any(item['global_target_tokens'] != 262080 for item in metrics)):
        raise ValueError('Interim snapshot training targets/updates are incomplete')
    tokenizer = ROOT / 'data/tokenizer-corpus/tokenizer.json'
    if file_hash(tokenizer) != metadata['tokenizer_sha256']: raise ValueError('Own tokenizer differs from snapshot')
    digest = file_hash(source); directory.mkdir(parents=True)
    kept = {key: value for key, value in metadata.items() if key not in {'optimizer', 'reference_model', 'torch_rng_state', 'cuda_rng_state'}}
    temporary = directory / 'model.pt.tmp'; torch.save(kept, temporary); temporary.replace(directory / 'model.pt')
    shutil.copyfile(tokenizer, directory / 'tokenizer.json')
    retained = torch.load(directory / 'model.pt', map_location='cpu', weights_only=True, mmap=True)
    if set(retained['model']) != set(metadata['model']) or retained['config'] != metadata['config']:
        raise ValueError('Retained snapshot model/configuration differs')
    for name, value in metadata['model'].items():
        if not torch.equal(value, retained['model'][name]) or not torch.isfinite(value).all():
            raise ValueError('Retained model tensor differs or is nonfinite: ' + name)
    if file_hash(source) != digest: raise ValueError('Original periodic snapshot changed during retention')
    receipt = {'prepared_at': datetime.now(timezone.utc).isoformat(), 'source_checkpoint': str(source),
               'source_checkpoint_sha256': digest, 'retained_checkpoint_sha256': file_hash(directory / 'model.pt'),
               'tokenizer_sha256': metadata['tokenizer_sha256'], 'exact_model_tensors': len(metadata['model']),
               'steps': 3000, 'planned_steps': 17746, 'actual_target_exposures': 786240000,
               'goal_achieved': False, 'source_training_completed': False,
               'scope': 'Exact model-only export of completed periodic snapshot; active curriculum continues separately.'}
    atomic_json(directory / 'preparation.json', receipt)
    return receipt


def main():
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'checkpoints/extended-curriculum-v1/language-pretraining/step-00003000.pt')
    parser.add_argument('--directory', type=Path, default=ROOT / 'checkpoints/clevr-foundation-baseline-v1')
    parser.add_argument('--reports', type=Path, default=ROOT / 'reports/clevr-foundation-baseline-v1')
    parser.add_argument('--wait-audit-pid', type=int, required=True)
    parser.add_argument('--device', default='cuda:2')
    args = parser.parse_args(); args.reports.mkdir(parents=True, exist_ok=True)
    files = [ROOT / name for name in capture()['files_sha256']]
    files.extend([Path(__file__), ROOT / 'src/slm_lab/clevr_evaluation.py', ROOT / 'scripts/audit_clevr_inputs.py'])
    loaded = capture(files)

    def status(message, **values):
        atomic_json(args.reports / 'status.json', {'status': message, 'updated_at': datetime.now(timezone.utc).isoformat(),
                                                  'goal_achieved': False, **values})

    try:
        pid = args.wait_audit_pid; identity = process_identity(pid)
        if identity is None or b'scripts/audit_clevr_inputs.py' not in Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0'):
            raise ValueError('Original-input auditor is not the verified live process')
        status('retaining own3000-update language snapshot', source=str(args.source))
        receipt = prepare(args.source.resolve(), args.directory.resolve())
        status('waiting for verified live original-input auditor', audit_pid=pid, audit_start_ticks=identity,
               retained_checkpoint_sha256=receipt['retained_checkpoint_sha256'])
        audit_path = ROOT / 'reports/clevr-original-input-audit-v1.json'
        while not audit_path.exists():
            if process_identity(pid) != identity: raise ValueError('Original-input auditor ended without passing report; inspect before evaluation')
            if capture(files) != loaded: raise ValueError('Baseline/evaluation sources changed while waiting')
            time.sleep(15)
        data = ROOT / 'data/clevr-reasoning-v1'; audit = json.loads(audit_path.read_text())
        if (audit.get('passed') is not True or audit['manifest_sha256'] != file_hash(data / 'manifest.json')
                or audit['audit_source_sha256'] != file_hash(ROOT / 'scripts/audit_clevr_inputs.py')):
            raise ValueError('Original-input audit does not match the finalized dataset/implementation')
        for name, digest in audit['source_files_sha256'].items():
            if file_hash(ROOT / name) != digest: raise ValueError('Audited data/model implementation changed')
        for name, digest in audit['files_sha256'].items():
            if file_hash(data / name) != digest: raise ValueError('Audited source questions/scenes changed')
        if capture(files) != loaded: raise ValueError('Baseline/evaluation sources changed before inference')
        checkpoint = args.directory / 'model.pt'
        if file_hash(checkpoint) != receipt['retained_checkpoint_sha256']: raise ValueError('Retained model changed while waiting')
        status('evaluating original held-out visual questions with actual/blank/wrong images', device=args.device)
        report = evaluate(checkpoint, data / 'valid.jsonl', args.reports / 'visual-reasoning.json', args.device)
        atomic_json(args.reports / 'summary.json', {'checkpoint_sha256': report['checkpoint_sha256'],
                    'input_audit_sha256': file_hash(audit_path), 'overall': report['overall'], 'by_task': report['by_task'],
                    'snapshot_steps': 3000, 'goal_achieved': False,
                    'scope': 'Interim own foundation baseline; no new CLEVR training or policy promotion, no independent final capability claim.'})
        status('interim original visual reasoning baseline completed; broader capability unverified', examples=len(report['results']))
    except Exception as error:
        status('interim visual assessment failed; inspect without restarting live work', error=str(error))
        raise


if __name__ == '__main__': main()
