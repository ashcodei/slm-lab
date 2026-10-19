"""All prepared held-out word crops, with declared batched pixel-only inference."""
from collections import Counter
from contextlib import nullcontext
from pathlib import Path

import torch

from .byte_word_reader import HEIGHT, WIDTH, collapse, pixels
from .diverse_images import atomic_json
from .evaluation_provenance import pixels_sha256
from .ocr_evaluation import normalized, transcription_scores


def counterparts(rows):
    if len(rows) < 2 or len({row['id'] for row in rows}) != len(rows): raise ValueError('Full reading evaluation requires unique original word cases')
    families = {}
    for row in rows: families.setdefault(row['byte_family'], []).append(row)
    selected = {}
    for family, items in families.items():
        for index, row in enumerate(items):
            for other in items[index + 1:] + items[:index]:
                if other['group_id'] != row['group_id'] and normalized(other['response']) != normalized(row['response']):
                    selected[row['id']] = other['id']; break
            else: raise ValueError('No different-photo/different-text counterpart in family: ' + family)
    return selected


def summarize(rows):
    if not rows: raise ValueError('Cannot summarize an empty reading assessment')
    result = {'examples': len(rows), 'original_groups': len({row['group_id'] for row in rows})}
    for name in ['proper', 'blank', 'wrong']:
        characters = sum(row[name + '_scores']['reference_characters'] for row in rows)
        if characters <= 0: raise ValueError('No actual reference characters')
        result[name + '_exact'] = sum(row[name + '_scores']['exact_match'] for row in rows)
        result[name + '_case_insensitive_exact'] = sum(row[name + '_scores']['case_insensitive_exact_match'] for row in rows)
        result[name + '_character_error'] = sum(row[name + '_scores']['character_edits'] for row in rows) / characters
        result[name + '_valid_utf8'] = sum(row[name + '_valid_utf8'] for row in rows) / len(rows)
    result['proper_correct_with_neither_blank_nor_wrong_match'] = sum(row['proper_scores']['exact_match'] and not row['blank_scores']['exact_match'] and not row['wrong_scores']['exact_match'] for row in rows)
    return result


@torch.inference_mode()
def evaluate(model, rows, output, device='cuda:2', batch_size=32):
    if not isinstance(batch_size, int) or batch_size < 1: raise ValueError('Full word evaluation needs a positive fixed batch size')
    rows = sorted(rows, key=lambda row: row['id']); others = counterparts(rows); generated = {}; inputs_sha = {}
    amp = lambda: torch.autocast('cuda', dtype=torch.bfloat16) if str(device).startswith('cuda') else nullcontext()
    model.eval()
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]; actual = [pixels(row) for row in batch]
        for row, value in zip(batch, actual): inputs_sha[row['id']] = pixels_sha256(value)
        with amp(): logits = model(torch.stack(actual).to(device))
        if not torch.isfinite(logits).all(): raise ValueError('Nonfinite full word-reading logits')
        for row, path in zip(batch, logits.argmax(-1).cpu().tolist()): generated[row['id']] = collapse(path)
        if start % (batch_size * 10) == 0: print('Actual held-out word crops inferred', min(len(rows), start + len(batch)), len(rows), flush=True)
    with amp(): blank_logits = model(torch.zeros(1, 3, HEIGHT, WIDTH, device=device))
    if not torch.isfinite(blank_logits).all(): raise ValueError('Nonfinite blank word-reader output')
    blank = collapse(blank_logits.argmax(-1)[0].cpu().tolist()); results = []
    for row in rows:
        other = others[row['id']]
        result = {'id': row['id'], 'group_id': row['group_id'], 'family': row['byte_family'], 'expected': row['response'],
                  'stored_image_sha256': row['stored_image_sha256'], 'input_pixels_sha256': inputs_sha[row['id']],
                  'wrong_image_id': other, 'wrong_pixels_sha256': inputs_sha[other]}
        for name, (text, valid) in [('proper', generated[row['id']]), ('blank', blank), ('wrong', generated[other])]:
            result[name + '_generated'] = text; result[name + '_valid_utf8'] = valid
            result[name + '_scores'] = transcription_scores(row['response'], text)
        results.append(result)
    report = {'overall': summarize(results), 'by_family': {family: summarize([row for row in results if row['family'] == family])
              for family in sorted({row['family'] for row in results})}, 'results': results,
              'proper_batch_size': batch_size, 'input_height': HEIGHT, 'input_width': WIDTH, 'goal_achieved': False,
              'control_generation': 'Each actual proper crop inferred once in fixed-size batches; one deterministic gray blank output and actual counterpart outputs reused explicitly.',
              'scope': 'Every prepared validation word crop retained; standalone recognition component, not complete-photo or native-policy question answering.',
              'limitations': ['Existing development crop split; not an independent final broad benchmark.',
                              'Batch32 convolution/FP32 recurrent kernels may round differently than the earlier single-crop assessment; matched case results are retained separately.',
                              'Controls retain original targets and choose different source groups and original transcripts; no transcript enters the model.',
                              'Multiple real crops can share one held-out parent; word count is distinct from photo count.']}
    atomic_json(Path(output), report)
    return report
