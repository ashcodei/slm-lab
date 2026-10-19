"""Audited lettering supplement, rendered locally from public text and fonts.

These are synthetic images with exact rendering labels, not real photographs or
human feedback. They supplement the existing real-image reading curriculum.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import hashlib
import io
import json
from pathlib import Path
import random
import re
import string

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

from .corpus import file_hash
from .diverse_images import atomic_json, jsonl
from .ocr_evaluation import TASKS
from .wikipedia_corpus import protected_questions

ALPHABET = string.ascii_letters + string.digits
HELD_OUT_FONTS = {'LiberationSans-Italic.ttf', 'LiberationSerif-Italic.ttf'}


def text_group(text):
    return hashlib.sha256(' '.join(text.casefold().split()).encode()).hexdigest()


def text_split(text):
    # Single letters bootstrap character coverage in training. Validation tests
    # novel combinations rather than withholding the entire character alphabet.
    if len(text) == 1:
        return 'train'
    return 'valid' if int(text_group(text)[:8], 16) % 10 == 0 else 'train'


def public_words(path):
    words = {}
    for row in jsonl(path):
        if row.get('license') != 'Public-domain-US' or row.get('source') != 'Project Gutenberg':
            continue
        origin = {key: row[key] for key in ['id', 'group_id', 'title', 'source_url', 'license']}
        for word in re.findall(r'\b[A-Za-z]{2,18}\b', row['text']):
            words.setdefault(word.casefold(), {'word': word, 'origin': origin})
    if len(words) < 100:
        raise ValueError('Insufficient documented public-domain word sources')
    return [words[key] for key in sorted(words)]


def render(text, font_path, seed):
    if not text or len(text) > 32 or any(ch not in ALPHABET + ' .,:;!?+-=/()' for ch in text):
        raise ValueError('Rendering accepts short supported ASCII text only')
    rng = random.Random(seed)
    font = ImageFont.truetype(str(font_path), rng.randint(36, 64))
    left, top, right, bottom = font.getbbox(text)
    margin = rng.randint(9, 20)
    width, height = right - left + 2 * margin, bottom - top + 2 * margin
    if width > 1024:
        raise ValueError('Rendered line exceeds the readable crop width')
    light = rng.randint(205, 255)
    dark = rng.randint(0, 50)
    background, foreground = ((light,) * 3, (dark,) * 3)
    if rng.random() < .25:
        background, foreground = foreground, background
    image = Image.new('RGB', (width, height), background)
    ImageDraw.Draw(image).text((margin - left, margin - top), text, font=font, fill=foreground)
    angle = rng.uniform(-6, 6)
    image = image.rotate(angle, Image.Resampling.BICUBIC, expand=True, fillcolor=background)
    if rng.random() < .25:
        image = image.filter(ImageFilter.GaussianBlur(rng.uniform(.15, .55)))
    noise = rng.randint(0, 5)
    if noise:
        values = np.asarray(image, dtype=np.int16)
        generator = np.random.default_rng(seed)
        values += generator.integers(-noise, noise + 1, values.shape, dtype=np.int16)
        image = Image.fromarray(np.clip(values, 0, 255).astype(np.uint8))
    return image, {'font_pixels': font.size, 'rotation_degrees': angle,
                   'background': background, 'foreground': foreground, 'noise': noise}


def pixel_hash(image, size):
    pixels = ImageOps.pad(image, (size, size), method=Image.Resampling.BICUBIC, color=(128, 128, 128))
    return hashlib.sha256(pixels.tobytes()).hexdigest()


def prepare(output, books, questions, real_validation, fonts, font_license,
            train_count=50000, valid_count=2000, seed=1729):
    if min(train_count, valid_count) < 1:
        raise ValueError('Both requested splits must be nonempty')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'prepare.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _prepare(output, Path(books), Path(questions), Path(real_validation),
                        Path(fonts), Path(font_license), train_count, valid_count, seed)


def _prepare(output, books, questions, real_validation, fonts, font_license,
             train_count, valid_count, seed):
    font_paths = sorted(fonts.glob('Liberation*.ttf'))
    train_fonts = [p for p in font_paths if p.name not in HELD_OUT_FONTS]
    valid_fonts = [p for p in font_paths if p.name in HELD_OUT_FONTS]
    if not train_fonts or len(valid_fonts) != len(HELD_OUT_FONTS):
        raise ValueError('Expected Liberation training and held-out fonts are missing')
    notice = font_license.read_text()
    if 'GNU General Public License v.2' not in notice or 'does not by itself cause' not in notice:
        raise ValueError('Expected Liberation font notice/exception is missing; inspect before use')
    configuration = {
        'version': 1, 'seed': seed, 'train_count': train_count, 'valid_count': valid_count,
        'books_sha256': file_hash(books), 'questions_sha256': file_hash(questions),
        'real_validation_sha256': file_hash(real_validation),
        'fonts': {p.name: file_hash(p) for p in font_paths},
        'font_license_sha256': file_hash(font_license),
    }
    configuration_path = output / 'configuration.json'
    if configuration_path.exists():
        if json.loads(configuration_path.read_text()) != configuration:
            raise ValueError('Partial lettering curriculum has a different configuration')
    else:
        atomic_json(configuration_path, configuration)
    manifest_path = output / 'manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest['configuration'] != configuration:
            raise ValueError('Completed lettering curriculum has a different configuration')
        for name, digest in manifest['files_sha256'].items():
            if file_hash(output / name) != digest:
                raise ValueError('Completed lettering records changed')
        return manifest
    images = output / 'images'
    images.mkdir(exist_ok=True)
    (output / 'font-license.txt').write_text(notice)
    words = public_words(books)
    matcher = protected_questions(questions)
    protected = {text_group(row['response']) for row in jsonl(real_validation) if row.get('task') in TASKS}
    counts = Counter()
    kinds = {'train': Counter(), 'valid': Counter()}
    groups = {'train': set(), 'valid': set()}
    pixels = {size: {'train': set(), 'valid': set()} for size in [224, 448]}
    seen = set()
    journal = output / 'collected.jsonl'
    # Commit records after image files; discard a partial final line on resume.
    raw = journal.read_bytes() if journal.exists() else b''
    raw = raw[:raw.rfind(b'\n') + 1]
    journal.write_bytes(raw)
    last_attempt = -1
    for line in raw.splitlines():
        row = json.loads(line)
        if file_hash(output / row['image']) != row['stored_image_sha256']:
            raise ValueError('Previously committed rendered image changed')
        split = row['split']; counts[split] += 1; kinds[split][row['render_kind']] += 1
        groups[split].add(row['group_id']); last_attempt = max(last_attempt, row['attempt'])
        for size in pixels:
            pixels[size][split].add(row[f'model_pixel_sha256_{size}'])
        seen.add((split, row['stored_image_sha256']))
    budgets = {'train': train_count, 'valid': valid_count}
    attempt = last_attempt + 1
    with journal.open('a') as target:
        while any(counts[split] < budget for split, budget in budgets.items()):
            if attempt > 100 * (train_count + valid_count):
                raise ValueError('Unable to satisfy lettering split budgets; inspect exclusions')
            rng = random.Random(seed + attempt)
            chance = rng.random(); origin = None
            if chance < .05:
                text = rng.choice(ALPHABET); kind = 'character'
            elif chance < .40:
                text = ''.join(rng.choices(ALPHABET + '-+/', k=rng.randint(2, 16))); kind = 'random_string'
            elif chance < .90:
                record = rng.choice(words); text = record['word']; origin = record['origin']; kind = 'public_word'
                text = rng.choice([text, text.lower(), text.upper()])
            else:
                records = rng.sample(words, 2); text = ' '.join(r['word'] for r in records)
                origin = [r['origin'] for r in records]; kind = 'public_word_pair'
            split = text_split(text); group = text_group(text)
            current = attempt; attempt += 1
            if counts[split] >= budgets[split] or len(text) > 24:
                continue
            if split == 'train' and group in protected:
                continue
            if list(matcher.iter(' '.join(text.casefold().split()))):
                continue
            font = rng.choice(train_fonts if split == 'train' else valid_fonts)
            image, style = render(text, font, seed + current)
            hashes = {size: pixel_hash(image, size) for size in pixels}
            if any(hashes[size] in pixels[size]['valid' if split == 'train' else 'train'] for size in pixels):
                continue
            buffer = io.BytesIO(); image.save(buffer, format='PNG'); encoded = buffer.getvalue()
            digest = hashlib.sha256(encoded).hexdigest()
            if (split, digest) in seen:
                continue
            identifier = hashlib.sha256(f'{seed}:{current}'.encode()).hexdigest()
            path = images / (identifier + '.png')
            path.write_bytes(encoded)
            row = {'id': 'rendered-' + identifier, 'group_id': 'rendered-text:' + group,
                   'prompt': 'Read the text in this image.', 'response': text,
                   'image': str(path.relative_to(output)), 'task': 'word_ocr', 'domain': 'rendered_text',
                   'source': 'Locally rendered public text and programmatically sampled characters',
                   'label_origin': 'Exact string passed to the font renderer; synthetic, not human feedback',
                   'public_text_origin': origin, 'render_kind': kind, 'split': split,
                   'font': font.name, 'font_sha256': configuration['fonts'][font.name],
                   'font_notice': 'font-license.txt', 'style': style,
                   'image_size': list(image.size), 'stored_image_sha256': digest, 'attempt': current,
                   **{f'model_pixel_sha256_{size}': value for size, value in hashes.items()}}
            target.write(json.dumps(row, ensure_ascii=True) + '\n'); target.flush()
            counts[split] += 1; kinds[split][kind] += 1; groups[split].add(row['group_id']); seen.add((split, digest))
            for size in pixels:
                pixels[size][split].add(hashes[size])
            if sum(counts.values()) % 1000 == 0:
                atomic_json(output / 'progress.json', {'status': 'rendering lettering supplement',
                                                      'splits': dict(counts), 'attempts': attempt})
    if groups['train'] & groups['valid']:
        raise ValueError('Rendered text groups overlap')
    for size in pixels:
        if pixels[size]['train'] & pixels[size]['valid']:
            raise ValueError('Rendered input pixels overlap')
    with (output / 'train.jsonl').open('w') as train, (output / 'valid.jsonl').open('w') as valid:
        for row in jsonl(journal):
            (train if row['split'] == 'train' else valid).write(json.dumps(row, ensure_ascii=True) + '\n')
    manifest = {
        'created_at': datetime.now(timezone.utc).isoformat(), 'configuration': configuration,
        'splits': {split: {'examples': counts[split], 'text_groups': len(groups[split]),
                          'by_kind': dict(kinds[split])} for split in budgets},
        'train_valid_text_group_overlap': 0, 'train_valid_exact_input_pixel_overlap': {'224': 0, '448': 0},
        'held_out_font_names': sorted(HELD_OUT_FONTS), 'protected_real_ocr_answer_groups': len(protected),
        'files_sha256': {name: file_hash(output / name) for name in ['train.jsonl', 'valid.jsonl', 'font-license.txt']},
        'limitations': ['Rendered text is a lettering supplement, not real-scene visual competence.',
                       'Public-domain status is the recorded US Gutenberg catalog status.',
                       'Public text sources were already in the text corpus; this is not a novel knowledge benchmark.',
                       'Short supported ASCII only; handwriting, multilingual scripts and complex pages remain pending.'],
    }
    atomic_json(manifest_path, manifest)
    atomic_json(output / 'progress.json', {'status': 'lettering supplement prepared; not trained', 'splits': dict(counts)})
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True); parser.add_argument('--books', required=True)
    parser.add_argument('--questions', required=True); parser.add_argument('--real-validation', required=True)
    parser.add_argument('--fonts', default='/usr/share/fonts/truetype/liberation')
    parser.add_argument('--font-license', default='/usr/share/doc/fonts-liberation/copyright')
    parser.add_argument('--train-count', type=int, default=50000); parser.add_argument('--valid-count', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=1729)
    args = parser.parse_args()
    result = prepare(args.output, args.books, args.questions, args.real_validation, args.fonts,
                     args.font_license, args.train_count, args.valid_count, args.seed)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
