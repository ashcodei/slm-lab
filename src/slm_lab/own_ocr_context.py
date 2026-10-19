"""Bounded structured context from our own image predictions, without target hints."""
import json
import math

from .text_localization_metrics import box


def prompt(question, predictions, tokenizer, max_text_length=4096, generation_tokens=32, max_words=64):
    """Return a raw user prompt for Encoder.prompt, not a target-aware answer.

    This is an optional interface for a measured native-model experiment, not
    an already validated production policy or a replacement for image input.
    Only pixel-derived text/boxes/scores enter the serialized context.
    """
    if not isinstance(question, str) or not question.strip(): raise ValueError('Original question must be nonempty text')
    if not isinstance(max_words, int) or max_words < 0 or not isinstance(generation_tokens, int) or generation_tokens < 1:
        raise ValueError('Invalid declared own-OCR context/generation budget')
    selected = []
    for prediction in predictions:
        if prediction.get('valid_utf8') is not True: continue
        text = prediction.get('text')
        if not isinstance(text, str) or not text.strip(): continue
        rectangle = box(prediction['box']); confidence = float(prediction['score'])
        if not math.isfinite(confidence) or not 0 <= confidence <= 1: raise ValueError('Invalid predicted text-box confidence')
        selected.append({'text': text, 'box': list(rectangle), 'confidence': confidence})
    selected.sort(key=lambda value: (-value['confidence'], value['box'], value['text']))
    available = len(selected); selected = selected[:max_words]
    def build(values):
        # Avoid turning literal text in an image into tokenizer control tokens.
        encoded = json.dumps(values, ensure_ascii=False, separators=(',', ':')).replace('<', '\\u003c').replace('>', '\\u003e')
        return question + '\nText predicted by our own image reader (may contain errors):\n' + encoded + '\nAnswer briefly.'
    while True:
        result = build(selected); formatted = 'User: ' + result + '\nAssistant:'
        # Preserve the complete question, BOS and the declared output budget.
        tokens = tokenizer.encode(formatted).ids
        if len(tokens) + 1 + generation_tokens <= max_text_length: break
        if not selected: raise ValueError('Complete question/empty context exceeds declared model budget')
        selected.pop()
    return {'prompt': result, 'predicted_words_supplied': len(selected), 'valid_predicted_words_available': available,
            'predicted_words_dropped_for_limit': available - len(selected), 'prompt_tokens_with_BOS': len(tokens) + 1,
            'generation_tokens': generation_tokens, 'selected_context': selected,
            'reference_answers_or_annotations_supplied': False,
            'scope': 'Structured context interface only; native-model usefulness and image grounding require actual matched evaluation.'}
