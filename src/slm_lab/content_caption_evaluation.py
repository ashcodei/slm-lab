"""Matched retrieval readout using own frozen content-mean caption features."""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer

from .content_caption_training import content_mean
from .corpus import file_hash
from .data import read_jsonl, stable_id
from .diverse_images import atomic_json
from .evaluation_provenance import capture, seal, pixels_sha256
from .training import Encoder, load_checkpoint
from .vision_evaluation import retrieval
from .vision_transfer import language_digest

ROOT = Path(__file__).resolve().parents[2]


@torch.no_grad()
def evaluate(checkpoint, data, output, device='cuda', count=128):
    files = [ROOT / name for name in capture()['files_sha256']] + [Path(__file__),
             ROOT / 'src/slm_lab/content_caption_training.py', ROOT / 'src/slm_lab/vision_transfer.py']
    provenance = capture(files); data_digest = file_hash(data)
    torch.set_num_threads(1); model, meta = load_checkpoint(checkpoint, device, training_state=False); model.eval()
    tokenizer = Path(checkpoint).parent / 'tokenizer.json'
    if file_hash(tokenizer) != meta['tokenizer_sha256']: raise ValueError('Own tokenizer changed')
    tok = Tokenizer.from_file(str(tokenizer)); encoder = Encoder(tok, model.config, Path(data).parent, 448)
    rows = read_jsonl(data)
    if len(rows) < count: raise ValueError('Insufficient held-out caption cases')
    selected = [rows[i * len(rows) // count] for i in range(count)]
    if len({row['group_id'] for row in selected}) != count: raise ValueError('Repeated held-out image group')
    before = language_digest(model.state_dict())
    images = []; texts = []; cases = []
    amp = lambda: torch.autocast('cuda', dtype=torch.bfloat16) if device.startswith('cuda') else nullcontext()
    for row in selected:
        pixels = encoder.pixels(row)[None].to(device)
        batch = encoder.batch([{**row, 'text': row['response'], 'image': None}], None, device)
        with amp():
            image = model.vision(pixels).mean(1)
            hidden = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                           return_hidden=True, return_logits=False)['hidden']
            text = content_mean(hidden, batch['attention_mask'])
        images.append(image); texts.append(text)
        cases.append({'id': row['id'], 'group_id': row['group_id'], 'input_pixels_sha256': pixels_sha256(pixels),
                      'caption_sha256': stable_id(row['response'])})
    if encoder.truncated or language_digest(model.state_dict()) != before:
        raise ValueError('Content readout truncated captions or changed frozen language weights')
    ids = torch.tensor([int(stable_id(row['response'])[:15], 16) for row in selected], device=device)
    report = {'checkpoint_sha256': file_hash(checkpoint), 'data_sha256': data_digest, 'image_size': 448,
              'examples': count, 'caption_retrieval': retrieval(torch.cat(images), torch.cat(texts), ids), 'cases': cases,
              'caption_pooling': 'content mean excluding BOS/EOS/padding', 'exact_language_and_critic_preserved': True,
              'goal_achieved': False, 'limitations': ['New content-feature readout, separate from the default EOS-feature metric.',
                                                    'Bounded reused development set; no native OCR/QA/general capability claim.']}
    seal(report, provenance, files=files, data=data, digest=data_digest, device=device, tokenizer_sha256=meta['tokenizer_sha256'])
    atomic_json(Path(output), report)
    print(json.dumps({key: value for key, value in report.items() if key not in {'cases', 'evaluation_provenance'}}, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['checkpoint', 'data', 'output']: parser.add_argument('--' + name, required=True)
    parser.add_argument('--device', default='cuda'); args = parser.parse_args()
    evaluate(args.checkpoint, args.data, args.output, args.device)


if __name__ == '__main__':
    main()
