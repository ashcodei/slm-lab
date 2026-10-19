"""Deterministic, evaluation-only AI2D inputs with original choices intact.

Opaque ABC geometry follows Ai2's Molmo data helper at commit793fa387:
https://github.com/allenai/molmo/blob/793fa387edfd6fd0f5b21eb8e0a7620a1f3799e1/olmo/hf_datasets/ai2d.py
That helper is Apache2 licensed. We explicitly use pinned Liberation Sans,
retain every original option (including duplicates/null placeholders), and
claim no pixel/protocol identity with published Arial/pruned-option scores.
No question, answer, text transcription or graph relation enters the renderer.
"""
import hashlib
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

PROTOCOL = 'AI2D-all-original-test-questions-opaque-ABC-LiberationSans-original-options-native448-v1'


def opaque_labels(image, annotations, font_path):
    result = image.convert('RGB').copy(); draw = ImageDraw.Draw(result); trace = []
    for key, box in annotations['text'].items():
        rectangle = box['rectangle']; label = box['replacementText']
        if (len(rectangle) != 2 or any(len(point) != 2 for point in rectangle)
                or any(type(v) not in [int, float] or not math.isfinite(v) for point in rectangle for v in point)
                or not isinstance(label, str)):
            raise ValueError('Every original finite rectangle and literal publisher replacement label required')
        (x1, y1), (x2, y2) = rectangle
        if x2 < x1 or y2 < y1: raise ValueError('Ordered original ABC rectangle required')
        if y1 == y2:
            trace.append(dict(id=key, rectangle=rectangle, replacementText=label, source_zero_height_box_skipped=True)); continue
        draw.rectangle([x1, y1, x2, y2], fill=(25, 249, 255))
        height = y2 - y1; size = 22; font = ImageFont.truetype(str(font_path), size)
        bounds = draw.textbbox((100, 100), label, font=font)
        if height > 4:
            while bounds[3] - bounds[1] > height:
                size -= 1
                if size < 1: raise ValueError('Original label cannot fit its declared positive height')
                font = ImageFont.truetype(str(font_path), size); bounds = draw.textbbox((100, 100), label, font=font)
        x = (x1 + x2) / 2 - (bounds[2] - bounds[0]) / 2
        y = y1 - (bounds[1] - 100) + (height - (bounds[3] - bounds[1])) / 2
        draw.text((x, y), label, fill='black', font=font)
        trace.append(dict(id=key, rectangle=rectangle, replacementText=label, font_size=size,
                          source_zero_height_box_skipped=False))
    return result, trace


def native_pixels(image):
    image = ImageOps.pad(image.convert('RGB'), (448, 448), method=Image.Resampling.BICUBIC, color=(128, 128, 128))
    rgb = np.asarray(image, dtype=np.uint8).copy()
    tensor = np.ascontiguousarray((rgb.astype(np.float32) / 127.5 - 1).transpose(2, 0, 1))
    return image, hashlib.sha256(rgb.tobytes()).hexdigest(), hashlib.sha256(tensor.tobytes()).hexdigest()


def native_prompt(question, choices):
    if (not isinstance(question, str) or not question.strip() or not isinstance(choices, list)
            or not 2 <= len(choices) <= 26 or any(not isinstance(choice, str) for choice in choices)):
        raise ValueError('Whole original question and every original option in original order required')
    options = '\n'.join(chr(65 + i) + '. ' + choice for i, choice in enumerate(choices))
    return 'User: Use the image to answer this multiple-choice question.\n' + question + '\n' + options + '\nAnswer with one letter only.\nAssistant:'


def source_gold(raw):
    value = raw['correctAnswer']
    if type(value) is int: index = value
    elif isinstance(value, str) and value.isdigit(): index = int(value)
    else: raise ValueError('Original integer or digit-string answer index required')
    if not 0 <= index < len(raw['answerTexts']): raise ValueError('Original answer index must address the unmodified original choices')
    return index
