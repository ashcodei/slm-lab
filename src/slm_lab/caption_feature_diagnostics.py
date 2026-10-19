"""Inspect own frozen caption representations; geometry is not semantic capability."""
import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path

import torch
from torch.nn import functional as F
from tokenizers import Tokenizer

from .corpus import file_hash
from .diverse_images import atomic_json, jsonl
from .evaluation_provenance import capture, seal
from .training import Encoder, load_checkpoint
from .vision_transfer import language_digest

ROOT = Path(__file__).resolve().parents[2]


def geometry(features):
    values = features.detach().cpu().double()
    if values.ndim != 2 or len(values) < 2 or not torch.isfinite(values).all():
        raise ValueError('Need multiple finite caption vectors')
    normalized = F.normalize(values, dim=-1)
    similarity = normalized @ normalized.T
    mask = ~torch.eye(len(values), dtype=torch.bool)
    pairs = similarity[mask]
    centered = normalized - normalized.mean(0)
    eigenvalues = torch.linalg.eigvalsh(centered @ centered.T).clamp_min(0)
    total = float(eigenvalues.sum())
    if total <= 1e-12:
        rank = 0.; first = 0.
    else:
        probabilities = eigenvalues / total
        positive = probabilities[probabilities > 0]
        rank = float(torch.exp(-(positive * positive.log()).sum()))
        first = float(probabilities[-1])
    return {'vectors': len(values), 'dimensions': values.shape[1],
            'mean_off_diagonal_cosine': float(pairs.mean()),
            'cosine_p05': float(torch.quantile(pairs, .05)), 'cosine_p95': float(torch.quantile(pairs, .95)),
            'normalized_centered_total_variance': total / len(values),
            'centered_effective_rank': rank, 'largest_centered_variance_share': first,
            'mean_feature_norm': float(values.norm(dim=-1).mean())}


@torch.no_grad()
def assess(checkpoint, data, output, count=512, device='cuda'):
    files = [ROOT / name for name in capture()['files_sha256']] + [Path(__file__), ROOT / 'src/slm_lab/vision_transfer.py']
    provenance = capture(files); data_digest = file_hash(data)
    torch.set_num_threads(1)
    model, meta = load_checkpoint(checkpoint, device, training_state=False); model.eval()
    tokenizer_path = Path(checkpoint).parent / 'tokenizer.json'
    if file_hash(tokenizer_path) != meta['tokenizer_sha256']:
        raise ValueError('Own tokenizer changed')
    tokenizer = Tokenizer.from_file(str(tokenizer_path)); encoder = Encoder(tokenizer, model.config, Path(data).parent)
    rows = sorted(list(jsonl(data)), key=lambda row: hashlib.sha256(row['id'].encode()).digest())[:count]
    if len(rows) != count or len({row['group_id'] for row in rows}) != count:
        raise ValueError('Expected distinct hash-selected source captions')
    before = language_digest(model.state_dict())
    collected = {name: [] for name in ['last_eos', 'last_content', 'mean_content']}
    amp = lambda: torch.autocast('cuda', dtype=torch.bfloat16) if device.startswith('cuda') else nullcontext()
    for start in range(0, len(rows), 8):
        subset = rows[start:start + 8]
        batch = encoder.batch([{**row, 'text': row['response'], 'image': None} for row in subset], None, device)
        with amp():
            hidden = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                           return_hidden=True, return_logits=False)['hidden']
        lengths = batch['attention_mask'].sum(1)
        for index, length in enumerate(lengths.tolist()):
            if length < 3 or int(batch['input_ids'][index, length - 1]) != encoder.eos:
                raise ValueError('Caption sequence does not end in own EOS')
            collected['last_eos'].append(hidden[index, length - 1].float().cpu())
            collected['last_content'].append(hidden[index, length - 2].float().cpu())
            collected['mean_content'].append(hidden[index, 1:length - 1].float().mean(0).cpu())
    if encoder.truncated or language_digest(model.state_dict()) != before:
        raise ValueError('Caption diagnostic truncated content or changed own language/critic weights')
    report = {'checkpoint_sha256': file_hash(checkpoint), 'data_sha256': data_digest,
              'caption_ids': [row['id'] for row in rows], 'examples': count,
              'pooling': {name: geometry(torch.stack(values)) for name, values in collected.items()},
              'exact_language_and_critic_preserved': True, 'goal_achieved': False,
              'scope': 'Frozen train-caption feature geometry; no native retrieval/OCR/QA or model promotion claim.',
              'limitations': ['Representation diversity is not evidence of semantic accuracy.',
                             'Alternative pooling is measured without training a vision adapter against it.',
                             'Existing own two-thousand-update language snapshot; later foundation/instruction stages may change this geometry.']}
    seal(report, provenance, files=files, data=data, digest=data_digest, device=device, tokenizer_sha256=meta['tokenizer_sha256'])
    atomic_json(Path(output), report)
    print(json.dumps({key: value for key, value in report.items() if key not in {'caption_ids', 'evaluation_provenance'}}, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True); parser.add_argument('--data', required=True)
    parser.add_argument('--output', required=True); parser.add_argument('--examples', type=int, default=512)
    parser.add_argument('--device', default='cuda'); args = parser.parse_args()
    if args.examples < 2: parser.error('At least two captions required')
    assess(args.checkpoint, args.data, args.output, args.examples, args.device)


if __name__ == '__main__':
    main()
