"""Native same-policy question experiment using our own predicted photo text."""
from datetime import datetime, timezone
from pathlib import Path

from slm_lab.diverse_images import atomic_json
from slm_lab.own_ocr_question_evaluation import evaluate

ROOT = Path(__file__).resolve().parents[1]


def main():
    reports = ROOT / 'reports/own-ocr-questions-v1'
    if reports.exists(): raise ValueError('Preserve existing own-OCR question experiment; inspect before resuming')
    reports.mkdir()
    def status(message, **values):
        atomic_json(reports / 'status.json', {'status': message, 'updated_at': datetime.now(timezone.utc).isoformat(), 'goal_achieved': False, **values})
    try:
        status('generating original native questions with own pixel-predicted text contexts', original_cases=128, modes=['plain', 'empty_context', 'own_context'])
        result = evaluate(reports / 'reading.json', device='cuda:3')
        status('own-OCR native question comparison completed; broad/frontier capability unverified', overall=result['overall'])
    except Exception as error:
        status('own-OCR native question experiment failed; inspect actual evidence before resuming', error=str(error)); raise


if __name__ == '__main__': main()
