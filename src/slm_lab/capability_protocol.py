"""The complete local capability assessment commands for model handoffs."""
from pathlib import Path

from .alignment_selection import REPORT_FILES


def specifications(root):
    root = Path(root)
    records = [
        ('code', 'slm_lab.code_quality', ['--prompt-mode', 'completion']),
        ('code_instruction', 'slm_lab.code_quality', ['--prompt-mode', 'instruction']),
        ('knowledge', 'slm_lab.knowledge_evaluation', ['--data', str(root / 'data/knowledge-eval-v1/questions.jsonl')]),
        ('math', 'slm_lab.math_evaluation', ['--data', str(root / 'data/math-reasoning-v2/test.jsonl')]),
        ('vision', 'slm_lab.vision_evaluation', ['--data', str(root / 'data/alignment-curriculum-v1/contrastive.valid.jsonl')]),
        ('ocr', 'slm_lab.ocr_evaluation', ['--data', str(root / 'data/alignment-curriculum-v1/images.valid.jsonl')]),
        ('objects', 'slm_lab.object_evaluation', ['--data', str(root / 'data/object-qa-v1/valid.jsonl')]),
        ('public_code', 'slm_lab.public_code_benchmark', ['--data', str(root / 'data/public-code-assessment-v1/questions.jsonl')]),
        ('cpp', 'slm_lab.cpp_evaluation', []),
    ]
    if len(records) != len(REPORT_FILES) or {name for name, _, _ in records} != set(REPORT_FILES):
        raise ValueError('Incomplete or duplicated handoff capability command registry')
    return [(name, module, REPORT_FILES[name], extra) for name, module, extra in records]
